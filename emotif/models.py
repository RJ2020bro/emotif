"""Training, montage augmentation, evaluation and export for montage-agnostic affect models."""
import datetime
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

from . import artifact
from . import asr as asr_mod
from . import face as FACE
from . import features as F
from . import montage as M

TARGETS = ("valence", "arousal")
FORMAT = "emotif-model-v2"


@dataclass
class SubjectData:
    subject: str
    channels: list
    zde: np.ndarray        # (W, C, 5) per-subject z-scored DE
    labels: dict           # target -> (W,) float with nan for missing
    groups: np.ndarray     # (W,) global trial index
    dataset: str
    keys: list = None      # (W,) (trial_id, start sample) of each window, for pairing with other analyses
    face: np.ndarray = None  # (W, C, 3) facial-activity indices relative to the subject's median (face.py)


@dataclass
class Cleaning:
    """Signal-cleaning pipeline applied before features (the app uses the same steps live)."""
    asr: bool = True                 # Artifact Subspace Reconstruction, calibrated per subject
    asr_k: float = asr_mod.DEFAULT_K
    detector: bool = True            # mask channels the learned artifact detector flags
    reject_uv: float = 0.0           # reject window if a usable channel exceeds this p-p (0 = off)
    bad_uv: float = F.BAD_UV         # channel unusable in a window above this p-p
    max_cal_sec: float = 600.0

    def describe(self):
        parts = ["filter"]
        if self.asr:
            parts.append(f"ASR k={self.asr_k:g}")
        if self.detector:
            parts.append("artifact detector")
        if self.reject_uv > 0:
            parts.append(f"reject >{self.reject_uv:g} uV")
        return " + ".join(parts)


@dataclass
class CleaningStats:
    windows: int = 0
    rejected: int = 0
    channel_windows: int = 0
    masked_detector: int = 0
    masked_amplitude: int = 0
    asr_blocks: int = 0
    asr_repaired_blocks: int = 0
    asr_skipped_subjects: int = 0

    def summary(self):
        cw = max(1, self.channel_windows)
        return {"windows": self.windows, "rejected": self.rejected,
                "masked_by_detector": self.masked_detector / cw, "masked_by_amplitude": self.masked_amplitude / cw,
                "asr_repaired_block_fraction": self.asr_repaired_blocks / max(1, self.asr_blocks),
                "asr_skipped_subjects": self.asr_skipped_subjects}


def extract(trials, hop=1.0, reject_uv=0.0, bad_uv=F.BAD_UV, cleaning=None, return_stats=False):
    """Filter, clean, window and compute per-subject z-scored DE.
    Channels that are unusable in a window (lost contact, flat, or flagged by the artifact detector) are
    stored as NaN and masked by the region embedding. With cleaning=None, only filtering and the
    amplitude rules (reject_uv, bad_uv) apply, as in earlier versions.
    Returns (list[SubjectData], n_rejected_windows[, CleaningStats])."""
    if cleaning is None:
        cleaning = Cleaning(asr=False, detector=False, reject_uv=reject_uv, bad_uv=bad_uv)
    det_model = artifact.load() if cleaning.detector else None
    stats = CleaningStats()
    gid = {}
    subjects = {}
    lab_keys = list(TARGETS) + sorted({k for t in trials for k in t.labels} - set(TARGETS))
    for t in trials:
        if t.fs < 100:
            raise ValueError(f"{t.trial_id}: fs={t.fs:.1f} Hz; need >= 100 Hz for the 45 Hz gamma band")
        subjects.setdefault(t.subject, []).append(t)
    out = []
    for sname, ts in subjects.items():
        chans = ts[0].channels
        for t in ts:
            if [c.upper() for c in chans] != [c.upper() for c in t.channels]:
                raise ValueError(f"subject {sname} has recordings with different channel lists")
            if abs(t.fs - ts[0].fs) > 1e-6:
                raise ValueError(f"subject {sname} has recordings at different sampling rates")
        fs = ts[0].fs
        xs = [t.x if t.filtered else F.filter_signal(t.x, t.fs) for t in ts]
        xf = xs                                           # filtered, before artifact repair: face indices
        if cleaning.asr:
            cal = np.concatenate(xs, axis=1)[:, : int(cleaning.max_cal_sec * fs)]
            try:
                state = asr_mod.calibrate(cal, fs, cleaning.asr_k)
            except ValueError:
                state = None
                stats.asr_skipped_subjects += 1
            if state is not None:
                cleaned = []
                for x in xs:
                    y, removed = asr_mod.clean(x, state, fs)
                    stats.asr_blocks += len(removed)
                    stats.asr_repaired_blocks += int((removed > 0).sum())
                    cleaned.append(y)
                xs = cleaned
        N = F.window_len(fs)
        rows = []          # (de, feats or None, good, labels, group, key, face)
        for t, x, x_f in zip(ts, xs, xf):
            g = gid.setdefault((t.subject, t.group or t.trial_id), len(gid))
            slow = FACE.slow_signal(x_f, fs)
            for s0 in F.sliding_windows(x, fs, hop):
                win = x[:, s0:s0 + N]
                stats.windows += 1
                good, reject = F.channel_quality(win, cleaning.reject_uv, cleaning.bad_uv)
                if reject:
                    stats.rejected += 1
                    continue
                bp = F.band_powers(win, fs)
                feats = artifact.channel_features(win, fs, bp) if det_model else None
                fc = FACE.window_indices(x_f[:, s0:s0 + N], slow[:, s0:s0 + N], fs)
                rows.append((F.de_from_power(bp), feats, good, t.labels, g, (t.trial_id, s0), fc))
        if not rows:
            continue
        if det_model:
            base = artifact.baseline([r[1] for r in rows])
        de_all, groups, keys, faces, labs = [], [], [], [], {k: [] for k in lab_keys}
        for de, feats, good, lab, g, key, fc in rows:
            stats.channel_windows += len(good)
            stats.masked_amplitude += int((~good).sum())
            mask = good.copy()
            if det_model:
                bad, _ = artifact.bad_channels(det_model, feats, base)
                newly = bad & mask
                stats.masked_detector += int(newly.sum())
                mask &= ~bad
                if not mask.any():
                    stats.rejected += 1
                    continue
            de = de.copy()
            de[~mask] = np.nan
            de_all.append(de)
            faces.append(fc)
            groups.append(g)
            keys.append(key)
            for k in lab_keys:
                labs[k].append(lab.get(k, np.nan))
        if not de_all:
            continue
        de = np.array(de_all)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu, sd = np.nanmean(de, axis=0), np.nanstd(de, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where(np.isnan(sd) | (sd < 1e-6), 1.0, sd)
        fa = np.array(faces)
        out.append(SubjectData(sname, chans, (de - mu) / sd, {k: np.array(v, dtype=float) for k, v in labs.items()},
                               np.array(groups), ts[0].dataset, keys, fa - np.median(fa, axis=0)))
    if not out:
        raise ValueError("no usable windows")
    return (out, stats.rejected, stats) if return_stats else (out, stats.rejected)


def montage_variants(channels, rng, n_random=4):
    """Channel-index subsets used for augmentation: every headset layout this montage can imitate
    (>= 2 electrodes) plus random 20-80 % electrode dropouts."""
    C = len(channels)
    out = {}
    for dev in M.DEVICES:
        idx = M.device_subset(channels, dev)
        if len(idx) >= 2 and len(idx) < C:
            out[dev] = idx
    for k in range(n_random):
        keep = max(2, int(round(C * rng.uniform(0.2, 0.8))))
        if keep >= C:
            continue
        out[f"random{k}"] = sorted(rng.choice(C, size=keep, replace=False).tolist())
    return out


def featurize(sd, idx=None):
    """Region features for the channels in idx (default all). Windows are grouped by which of those
    channels were usable, so every window is pooled only over electrodes that actually had signal."""
    idx = list(range(len(sd.channels))) if idx is None else list(idx)
    z = sd.zde[:, idx, :]
    avail = ~np.isnan(z[:, :, 0])
    out = np.zeros((z.shape[0], M.N_FEATURES))
    if z.shape[0] == 0:
        return out
    patterns, inv = np.unique(avail, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    for p_i, pat in enumerate(patterns):
        rows = np.flatnonzero(inv == p_i)
        cols = np.flatnonzero(pat)
        if len(cols) == 0:
            continue
        W, mask = M.region_weights([sd.channels[idx[c]] for c in cols])
        out[rows] = M.region_features(z[rows][:, cols, :], W, mask)
    return out


def featurize_face(sd, idx=None):
    """Face indices pooled into regions exactly like featurize pools the EEG: per window, over the channels that were
    usable for the EEG features. (W, 51)."""
    idx = list(range(len(sd.channels))) if idx is None else list(idx)
    avail = ~np.isnan(sd.zde[:, idx, 0])
    out = np.zeros((sd.zde.shape[0], FACE.N_REGION_FACE))
    if sd.face is None or sd.zde.shape[0] == 0:
        return out
    fc = sd.face[:, idx, :]
    patterns, inv = np.unique(avail, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    for p_i, pat in enumerate(patterns):
        rows = np.flatnonzero(inv == p_i)
        cols = np.flatnonzero(pat)
        if len(cols) == 0:
            continue
        W, _ = M.region_weights([sd.channels[idx[c]] for c in cols])
        out[rows] = FACE.pool(fc[rows][:, cols, :], W)
    return out


def build_training_set(subjects, target, augment=2, seed=0, rows=None, with_face=False):
    """Full-montage rows plus `augment` montage-variant copies per window. rows: optional dict
    subject -> boolean mask of windows to include."""
    rng = np.random.default_rng(seed)
    X, y, Q = [], [], []
    for sd in subjects:
        m = ~np.isnan(sd.labels[target])
        if rows is not None:
            m &= rows.get(sd.subject, np.zeros_like(m))
        if not m.any():
            continue
        X.append(featurize(sd)[m])
        y.append(sd.labels[target][m])
        if with_face:
            Q.append(featurize_face(sd)[m])
        if augment > 0:
            variants = list(montage_variants(sd.channels, rng).values())
            if variants:
                wi = np.flatnonzero(m)
                for _ in range(augment):
                    choice = rng.integers(0, len(variants), size=len(wi))
                    for v, idx in enumerate(variants):
                        sel = wi[choice == v]
                        if len(sel):
                            sub = SubjectData(sd.subject, sd.channels, sd.zde[sel], {}, sd.groups[sel], sd.dataset,
                                              face=None if sd.face is None else sd.face[sel])
                            X.append(featurize(sub, idx))
                            y.append(sd.labels[target][sel])
                            if with_face:
                                Q.append(featurize_face(sub, idx))
    if not X:
        empty = (np.zeros((0, M.N_FEATURES)), np.zeros(0, dtype=int))
        return (*empty, np.zeros((0, FACE.N_REGION_FACE))) if with_face else empty
    out = (np.vstack(X), np.concatenate(y).astype(int))
    return (*out, np.vstack(Q)) if with_face else out


GRAPH_KINDS = {"graph": "logreg", "graph-mlp": "mlp"}
BRAIN_KINDS = {"brain": "logreg", "brain-mlp": "mlp"}
KINDS = ("logreg", "mlp", "graph", "graph-mlp", "brain", "brain-mlp")


class GraphModel:
    """Region-graph diffusion (montage.graph_features) in front of a logistic or MLP head."""
    def __init__(self, inner, hops=M.GRAPH_HOPS):
        self.inner, self.hops = inner, hops

    def transform(self, X):
        return M.graph_features(X, hops=self.hops)

    def fit(self, X, y):
        self.inner.fit(self.transform(X), y)
        self.classes_ = self.inner.classes_
        return self

    def predict(self, X):
        return self.inner.predict(self.transform(X))

    def predict_proba(self, X):
        return self.inner.predict_proba(self.transform(X))


class BrainModel:
    """Brain-only decoder. It removes from the EEG features everything the facial-activity indices predict (pooled
    indices and their squares, ridge fitted on the training people), then decodes what is left. The removal is
    affine, so it folds into the first layer: the exported head reads [EEG features, standardized face input] and
    needs nothing else at run time.
        x_brain = x - [q_s, 1] B         q_s = (face.design(f) - mu) / sd
    hold_z: the 99.5th percentile, over training windows, of the largest |standardized pooled face index|. Beyond it
    the face is outside what the correction was learned on, so the app holds the last output instead of decoding.
    """
    def __init__(self, inner, alpha=1.0):
        self.inner, self.alpha = inner, alpha

    def _z(self, F_reg):
        q = (FACE.design(F_reg) - self.q_mu) / self.q_sd
        return np.hstack([q, np.ones((len(q), 1))])

    def fit(self, X, y, F_reg):
        q = FACE.design(F_reg)
        self.q_mu = q.mean(axis=0)
        sd = q.std(axis=0)
        self.q_sd = np.where(sd < 1e-9, 1.0, sd)
        Z = self._z(F_reg)
        self.hold_z = float(np.percentile(np.abs(Z[:, :FACE.N_REGION_FACE]).max(axis=1), 99.5))
        reg = self.alpha * np.eye(Z.shape[1])
        reg[-1, -1] = 0.0
        self.B = np.linalg.solve(Z.T @ Z + reg, Z.T @ X)
        self.inner.fit(X - Z @ self.B, y)
        self.classes_ = self.inner.classes_
        return self

    def face_out_of_range(self, F_reg):
        return np.abs(self._z(F_reg)[:, :FACE.N_REGION_FACE]).max(axis=1) > self.hold_z

    def residual(self, X, F_reg):
        return X - self._z(F_reg) @ self.B

    def export_input(self, X, F_reg):
        return np.hstack([X, self._z(F_reg)[:, :-1]])

    def predict(self, X, F_reg):
        return self.inner.predict(self.residual(X, F_reg))

    def predict_proba(self, X, F_reg):
        return self.inner.predict_proba(self.residual(X, F_reg))


def fit_model(kind, X, y, F_reg=None, seed=0):
    clf = make_model(kind, seed)
    return clf.fit(X, y, F_reg) if isinstance(clf, BrainModel) else clf.fit(X, y)


def predict_model(clf, X, F_reg=None):
    return clf.predict(X, F_reg) if isinstance(clf, BrainModel) else clf.predict(X)


def head_input(clf, X, F_reg=None):
    """What the exported head reads for these windows (python mirror of the app's modelInput)."""
    if isinstance(clf, BrainModel):
        return clf.export_input(X, F_reg)
    if isinstance(clf, GraphModel):
        return clf.transform(X)
    return X


def make_model(kind, seed=0):
    if kind in BRAIN_KINDS:
        return BrainModel(make_model(BRAIN_KINDS[kind], seed))
    if kind in GRAPH_KINDS:
        return GraphModel(make_model(GRAPH_KINDS[kind], seed))
    if kind == "logreg":
        return LogisticRegression(C=0.3, max_iter=4000)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, early_stopping=True,
                             n_iter_no_change=15, random_state=seed)
    raise ValueError(kind)


def _score(y_true, y_pred):
    return {"acc": float(np.mean(y_true == y_pred)),
            "bacc": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro")),
            "n": int(len(y_true))}


def _test_sets(sd, mask):
    """(features, pooled face indices) for the full montage and every headset layout this subject can imitate."""
    out = {"full": (featurize(sd)[mask], featurize_face(sd)[mask])}
    for dev in M.DEVICES:
        idx = M.device_subset(sd.channels, dev)
        if len(idx) >= 2 and len(idx) < len(sd.channels):
            out[dev] = (featurize(sd, idx)[mask], featurize_face(sd, idx)[mask])
    return out


def _accumulate(acc, name, y_true, y_pred):
    a = acc.setdefault(name, ([], []))
    a[0].append(y_true)
    a[1].append(y_pred)


def _finish(acc):
    return {k: _score(np.concatenate(t), np.concatenate(p)) for k, (t, p) in acc.items()}


def evaluate(subjects, target, kind="logreg", augment=2, folds=5, modes=("within", "loso"), seed=0):
    """Returns {"within": {montage: scores}, "loso": {montage: scores}, "chance": ...}."""
    res = {}
    labels_all = np.concatenate([sd.labels[target][~np.isnan(sd.labels[target])] for sd in subjects]).astype(int)
    if len(np.unique(labels_all)) < 2:
        raise ValueError(f"{target}: fewer than two classes")
    classes, counts = np.unique(labels_all, return_counts=True)
    res["chance"] = float(counts.max() / counts.sum())          # majority-class accuracy
    res["chance_bacc"] = float(1.0 / len(classes))              # balanced-accuracy chance

    if "within" in modes:
        acc = {}
        for sd in subjects:
            m = ~np.isnan(sd.labels[target])
            g = sd.groups[m]
            y = sd.labels[target][m].astype(int)
            if len(np.unique(g)) < 2 or len(np.unique(y)) < 2:
                continue
            idx_valid = np.flatnonzero(m)
            for tr, te in GroupKFold(n_splits=min(folds, len(np.unique(g)))).split(idx_valid, y, g):
                if len(np.unique(y[tr])) < 2:
                    continue
                trm = np.zeros(len(m), bool)
                trm[idx_valid[tr]] = True
                tem = np.zeros(len(m), bool)
                tem[idx_valid[te]] = True
                X, yy, Qf = build_training_set([sd], target, augment, seed, rows={sd.subject: trm}, with_face=True)
                clf = fit_model(kind, X, yy, Qf, seed)
                for name, (Xt, Qt) in _test_sets(sd, tem).items():
                    _accumulate(acc, name, sd.labels[target][tem].astype(int), predict_model(clf, Xt, Qt))
        if acc:
            res["within"] = _finish(acc)

    if "loso" in modes and len(subjects) >= 2:
        acc, per = {}, {}
        for i, sd in enumerate(subjects):
            train = [s for j, s in enumerate(subjects) if j != i]
            X, yy, Qf = build_training_set(train, target, augment, seed, with_face=True)
            m = ~np.isnan(sd.labels[target])
            if len(np.unique(yy)) < 2 or not m.any():
                continue
            clf = fit_model(kind, X, yy, Qf, seed)
            for name, (Xt, Qt) in _test_sets(sd, m).items():
                yt, yp = sd.labels[target][m].astype(int), predict_model(clf, Xt, Qt)
                _accumulate(acc, name, yt, yp)
                per.setdefault(name, {})[sd.subject] = _score(yt, yp)
        if acc:
            res["loso"] = _finish(acc)
            res["loso_by_subject"] = per
    return res


def export_layers(clf):
    """Layers as {W: (in, out), b: (out,), act}; the last layer is softmax over clf.classes_. For a BrainModel the
    face removal is folded into the first layer, whose input becomes [features, standardized face input]."""
    if isinstance(clf, BrainModel):
        layers = export_layers(clf.inner)
        W1, b1 = np.array(layers[0]["W"]), np.array(layers[0]["b"])
        Bq, B0 = clf.B[:-1], clf.B[-1]
        layers[0] = {"W": np.vstack([W1, -Bq @ W1]).tolist(), "b": (b1 - B0 @ W1).tolist(), "act": layers[0]["act"]}
        return layers
    if isinstance(clf, GraphModel):
        clf = clf.inner
    def two_class(W, b):
        return np.hstack([np.zeros_like(W), W]), np.array([0.0, b[0]])

    if isinstance(clf, LogisticRegression):
        W, b = clf.coef_.T, clf.intercept_
        if W.shape[1] == 1:
            W, b = two_class(W, b)
        return [{"W": W.tolist(), "b": b.tolist(), "act": "softmax"}]
    layers, n = [], len(clf.coefs_)
    if clf.activation != "relu":
        raise ValueError("only relu MLPs are exportable")
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        if i < n - 1:
            layers.append({"W": W.tolist(), "b": b.tolist(), "act": "relu"})
        else:
            if W.shape[1] == 1:
                W, b = two_class(W, b)
            layers.append({"W": W.tolist(), "b": b.tolist(), "act": "softmax"})
    return layers


def forward(layers, X, temperature=1.0):
    """Class probabilities. temperature divides the final logits (temperature scaling, Guo et al. 2017)."""
    z = np.atleast_2d(X)
    for L in layers:
        z = z @ np.array(L["W"]) + np.array(L["b"])
        if L["act"] == "relu":
            z = np.maximum(z, 0)
        else:
            z = z / temperature
            z = np.exp(z - z.max(axis=1, keepdims=True))
            z /= z.sum(axis=1, keepdims=True)
    return z


def ece(prob, y_idx, bins=10):
    """Expected calibration error of the top-class probability (equal-width bins)."""
    conf, pred = prob.max(axis=1), prob.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(np.mean(pred[m] == y_idx[m]) - conf[m].mean())
    return float(e)


def calibrate(subjects, target, kind="logreg", augment=2, seed=0, folds=5):
    """Out-of-fold probabilities on the full montage (leave-one-subject-out with >= 2 subjects, else
    recording-wise folds), then the temperature that best calibrates them. Returns a dict with the
    temperature and ECE / NLL before and after; the temperature is fitted and scored on the same
    out-of-fold set, so the 'after' numbers are optimistic by construction."""
    P_layers, X_all, y_all = [], [], []
    if len(subjects) >= 2:   # test people: only those with this target (a model's heads can come from different datasets)
        splits = [([s for s in subjects if s is not sd], sd, None) for sd in subjects if (~np.isnan(sd.labels[target])).any()]
    else:
        sd = subjects[0]
        m = ~np.isnan(sd.labels[target])
        g = sd.groups[m]
        idx = np.flatnonzero(m)
        splits = []
        for tr, te in GroupKFold(n_splits=min(folds, len(np.unique(g)))).split(idx, groups=g):
            trm = np.zeros(len(m), bool)
            trm[idx[tr]] = True
            tem = np.zeros(len(m), bool)
            tem[idx[te]] = True
            splits.append(([sd], sd, (trm, tem)))
    classes = None
    for train, test_sd, masks in splits:
        rows = None if masks is None else {test_sd.subject: masks[0]}
        X, y, Qf = build_training_set(train, target, augment, seed, rows=rows, with_face=True)
        if len(np.unique(y)) < 2:
            continue
        clf = fit_model(kind, X, y, Qf, seed)
        if classes is None:
            classes = [int(c) for c in clf.classes_]
        if [int(c) for c in clf.classes_] != classes:
            continue
        m = ~np.isnan(test_sd.labels[target]) if masks is None else masks[1]
        Xt = head_input(clf, featurize(test_sd)[m], featurize_face(test_sd)[m])
        layers = export_layers(clf)
        P_layers.append(layers)
        X_all.append(Xt)
        y_all.append(np.array([classes.index(int(v)) for v in test_sd.labels[target][m]], dtype=int))
    if not P_layers:
        return None
    # stack logits from each fold's own model
    def probs(T):
        return np.vstack([forward(L, X, T) for L, X in zip(P_layers, X_all)])
    y = np.concatenate(y_all)
    grid = np.exp(np.linspace(np.log(0.25), np.log(20.0), 241))

    def best_T(folds_idx):
        Pg = [np.vstack([forward(P_layers[k], X_all[k], T) for k in folds_idx]) for T in grid]
        yy = np.concatenate([y_all[k] for k in folds_idx])
        nll = [-np.mean(np.log(P[np.arange(len(yy)), yy] + 1e-12)) for P in Pg]
        return float(grid[int(np.argmin(nll))]), float(min(nll))

    T, nll_after = best_T(range(len(P_layers)))
    p1, pT = probs(1.0), probs(T)
    # honest estimate: each fold calibrated with a temperature fitted on the other folds only
    held = np.vstack([forward(P_layers[k], X_all[k], best_T([j for j in range(len(P_layers)) if j != k])[0])
                      for k in range(len(P_layers))]) if len(P_layers) > 1 else pT
    return {"temperature": T, "ece_before": ece(p1, y), "ece_after": ece(pT, y), "ece_after_heldout": ece(held, y),
            "nll_before": float(-np.mean(np.log(p1[np.arange(len(y)), y] + 1e-12))),
            "nll_after": nll_after, "n": int(len(y))}


def train_and_export(subjects, kind="logreg", augment=2, targets=TARGETS, cv_modes=("within", "loso"),
                     folds=5, seed=0, name="", notes=""):
    heads, report = {}, {}
    for target in targets:
        if all(np.isnan(sd.labels[target]).all() for sd in subjects):
            continue
        X, y, Qf = build_training_set(subjects, target, augment, seed, with_face=True)
        if len(np.unique(y)) < 2:
            continue
        if cv_modes:
            report[target] = evaluate(subjects, target, kind, augment, folds, cv_modes, seed)
        clf = fit_model(kind, X, y, Qf, seed)
        heads[target] = {"classes": [int(c) for c in clf.classes_], "layers": export_layers(clf)}
        if isinstance(clf, BrainModel):
            heads[target]["face"] = {"mu": clf.q_mu.tolist(), "sd": clf.q_sd.tolist(), "hold_z": clf.hold_z}
        cal = calibrate(subjects, target, kind, augment, seed, folds)
        if cal is not None:
            heads[target]["temperature"] = cal["temperature"]
            report.setdefault(target, {})["calibration"] = cal
    if not heads:
        raise ValueError("no target has two or more classes")
    transform = None
    if kind in GRAPH_KINDS:
        transform = {"kind": "graph-diffusion", "hops": M.GRAPH_HOPS, "sigma_deg": M.GRAPH_SIGMA_DEG,
                     "adjacency": M.region_adjacency().round(8).tolist()}
    face_input = None
    if kind in BRAIN_KINDS:
        face_input = {"kind": "face-indices-v1", "eye_lowpass_hz": FACE.EYE_LP_HZ,
                      "muscle_bands": [list(b) for b in FACE.MUSCLE_BANDS], "n": FACE.N_FACE_INPUT,
                      "centering": "per-person median (training: session; app: resting baseline)"}
    return {
        "format": FORMAT,
        "transform": transform,
        "face_input": face_input,
        "name": name or "+".join(sorted({sd.dataset for sd in subjects})),
        "notes": notes,
        "created": datetime.date.today().isoformat(),
        "feature_space": {
            "kind": "region-v1",
            "regions": M.REGIONS,
            "region_pairs": [list(p) for p in M.REGION_PAIRS],
            "region_anchors": {r: M.POSITIONS[r.upper()].round(5).tolist() for r in M.REGIONS},
            "sigma_deg": M.SIGMA_DEG,
            "max_deg": M.MAX_DEG,
            "n_features": M.N_FEATURES,
            "normalization": "per-session z-score of channel DE before region pooling",
        },
        "bands": [list(b) for b in F.BANDS],
        "model": kind,
        "augment": augment,
        "datasets": sorted({sd.dataset for sd in subjects}),
        "n_subjects": len(subjects),
        "n_windows": int(sum(len(sd.groups) for sd in subjects)),
        "heads": heads,
        "cv": report,
    }
