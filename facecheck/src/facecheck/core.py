"""Audit an EEG decoder for two leaks that inflate reported accuracy.

1. Split leakage. When each recording has one label and windows are split at random, the same recording is in
   training and test, so a model can score well by recognising the recording. Test: give every recording a random
   label and rerun the same split. If accuracy stays high, the protocol measures recording identity.
2. Face leakage. Eye blinks, eye movements and jaw or forehead muscle activity are large in scalp EEG and change
   with emotion and effort. Test: decode from face indices alone, and measure how much of the EEG decoder's output
   the face indices explain (face share).
"""
import numpy as np
from scipy.signal import butter, sosfilt, welch

EPS = 1e-12


def face_indices(x, fs, win_sec=2.0, hop_sec=None):
    """Face indices per window and channel from a continuous recording.

    x: (channels, samples) in microvolts. Returns (windows, channels * 3) with, per channel,
      eye  = log peak-to-peak of the 0.5-3 Hz component (blinks, eye movements; causal filters over the whole
             recording, so there are no window-edge transients)
      jaw1 = log power 20-30 Hz, jaw2 = log power 30-45 Hz (scalp power above ~20 Hz is mostly muscle)
    each relative to that channel's median over the recording, and the window start samples.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be (channels, samples)")
    n = int(round(win_sec * fs))
    hop = int(round((hop_sec or win_sec) * fs))
    if x.shape[1] < n:
        raise ValueError("recording shorter than one window")
    hp = butter(2, 0.5, "highpass", fs=fs, output="sos")
    lp = butter(4, 3.0, "lowpass", fs=fs, output="sos")
    xf = sosfilt(hp, x - x[:, :1], axis=1)
    slow = sosfilt(lp, xf, axis=1)
    top = min(45.0, 0.45 * fs)
    bands = [(20.0, min(30.0, top)), (min(30.0, top), top)]
    starts = np.arange(0, x.shape[1] - n + 1, hop)
    out = np.zeros((len(starts), x.shape[0], 3))
    for k, s in enumerate(starts):
        out[k, :, 0] = np.log(np.ptp(slow[:, s:s + n], axis=1) + 1e-9)
        f, p = welch(xf[:, s:s + n], fs=fs, nperseg=min(n, int(fs)), axis=1)
        for j, (lo, hi) in enumerate(bands):
            sel = (f >= lo) & (f < hi) if hi > lo else (f >= lo)
            out[k, :, 1 + j] = np.log(p[:, sel].mean(axis=1) + EPS) if sel.any() else 0.0
    out -= np.median(out, axis=0, keepdims=True)
    return out.reshape(len(starts), -1), starts


def _bacc(y, p):
    cls = np.unique(y)
    return float(np.mean([np.mean(p[y == c] == c) for c in cls]))


def _model(kind, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "forest":
        return RandomForestClassifier(200, n_jobs=-1, random_state=seed, class_weight="balanced")
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    if hasattr(kind, "fit"):
        from sklearn.base import clone
        return clone(kind)
    raise ValueError("model must be 'logreg', 'forest' or a scikit-learn estimator")


def _cv(kind, X, y, splits, seed, scores=False):
    pred = np.full(len(y), -1)
    sc = None
    for tr, te in splits:
        m = _model(kind, seed).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        if scores:
            s = m.predict_proba(X[te])
            if sc is None:
                sc = np.zeros((len(y), s.shape[1]))
            sc[te] = s
    return (pred, sc) if scores else pred


def _random_labels(y, recording, rng):
    recs = np.unique(recording)
    lab = np.resize(np.unique(y), len(recs))[rng.permutation(len(recs))]
    m = dict(zip(recs, lab))
    return np.array([m[r] for r in recording])


def _temporal(recording, frac=0.7, gap=2):
    tr, te = [], []
    for r in np.unique(recording):
        idx = np.flatnonzero(recording == r)
        k = int(round(len(idx) * frac))
        tr += idx[:k].tolist()
        te += idx[k + gap:].tolist()
    return np.array(tr, int), np.array(te, int)


def _cv_r2(out, face, seed):
    """Cross-validated R^2 of the decoder's output explained by the face indices and their squares (ridge)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    Z = np.hstack([face, face ** 2])
    r2 = []
    for col in range(out.shape[1] if out.shape[1] > 2 else 1):
        t = out[:, col] if out.shape[1] > 2 else out[:, -1]
        pred = np.zeros(len(t))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(Z):
            sc = StandardScaler().fit(Z[tr])
            pred[te] = Ridge(1.0).fit(sc.transform(Z[tr]), t[tr]).predict(sc.transform(Z[te]))
        ss = np.sum((t - t.mean()) ** 2)
        r2.append(1 - np.sum((t - pred) ** 2) / ss if ss > 0 else 0.0)
    return float(np.mean(r2))


def audit(X, y, recording, subject, face=None, model="logreg", n_scrambles=3, n_folds=5, seed=0):
    """Run every check. Windows must be in time order within each recording.

    X: (windows, features) decoder input; y: label per window; recording, subject: ids per window;
    face: optional (windows, k) face indices (e.g. from face_indices). Returns a dict of balanced accuracies and
    summary numbers; see report().
    """
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    X = np.asarray(X, dtype=np.float64)
    y, recording, subject = np.asarray(y), np.asarray(recording), np.asarray(subject)
    if not (len(X) == len(y) == len(recording) == len(subject)):
        raise ValueError("X, y, recording and subject must have one entry per window")
    labels_per_rec = {r: np.unique(y[recording == r]) for r in np.unique(recording)}
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    n_sub = len(np.unique(subject))
    res = {"n_windows": int(len(y)), "n_recordings": len(labels_per_rec), "n_subjects": int(n_sub),
           "chance": 1.0 / len(classes),
           "one_label_per_recording": all(len(v) == 1 for v in labels_per_rec.values())}
    rnd = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y))
    subj = list(GroupKFold(min(n_folds, n_sub)).split(X, y, subject))
    res["random_split"] = _bacc(y, _cv(model, X, y, rnd, seed))
    sc = []
    for k in range(n_scrambles):
        yr = _random_labels(y, recording, rng)
        s = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed + k).split(X, yr))
        sc.append(_bacc(yr, _cv(model, X, yr, s, seed)))
    res["scrambled_labels"] = float(np.mean(sc))
    tr, te = _temporal(recording)
    res["temporal_split"] = _bacc(y[te], _model(model, seed).fit(X[tr], y[tr]).predict(X[te])) if len(te) else None
    pred, scores = _cv(model, X, y, subj, seed, scores=True)
    res["subject_split"] = _bacc(y, pred)
    if face is not None:
        face = np.asarray(face, dtype=np.float64)
        res["face_only_random"] = _bacc(y, _cv(model, face, y, rnd, seed))
        res["face_only_subject"] = _bacc(y, _cv(model, face, y, subj, seed))
        res["face_share"] = _cv_r2(scores, face, seed)
        gain = res["subject_split"] - res["chance"]
        res["leakage_ratio"] = (res["face_only_subject"] - res["chance"]) / gain if gain > 0.02 else None
    res["inflation"] = res["random_split"] - res["subject_split"]
    res["verdicts"] = verdicts(res)
    return res


def verdicts(r):
    v = []
    ch = r["chance"]
    if r["scrambled_labels"] - ch > 0.5 * max(r["random_split"] - ch, 1e-9) and r["random_split"] - ch > 0.05:
        v.append("SPLIT LEAKAGE: random-split accuracy survives scrambled labels "
                 f"({r['scrambled_labels']:.2f} vs {r['random_split']:.2f}); the random split measures recording "
                 "identity. Report subject-wise results.")
    if r["inflation"] > 0.1:
        v.append(f"INFLATED: random split exceeds new-subject accuracy by {r['inflation']:.2f}.")
    lr = r.get("leakage_ratio")
    if lr is not None and lr >= 0.5:
        v.append(f"FACE LEAKAGE: face indices alone reach {lr:.0%} of the EEG decoder's gain over chance on new subjects.")
    if r.get("face_share") is not None and r["face_share"] >= 0.2:
        v.append(f"FACE SHARE: {r['face_share']:.0%} of the decoder's output is explained by face indices.")
    if not v:
        v.append("No leak detected by these checks.")
    return v


def report(r):
    rows = [("Random window split", r["random_split"]), ("Same split, scrambled labels", r["scrambled_labels"]),
            ("Train early / test late in each recording", r["temporal_split"]),
            ("New subjects (subject-wise split)", r["subject_split"])]
    if "face_only_subject" in r:
        rows += [("Face only, random split", r["face_only_random"]), ("Face only, new subjects", r["face_only_subject"])]
    lines = [f"facecheck: {r['n_windows']} windows, {r['n_recordings']} recordings, {r['n_subjects']} subjects, "
             f"chance {r['chance']:.2f}", ""]
    lines += [f"  {name:<44} {v:.3f}" for name, v in rows if v is not None]
    if r.get("face_share") is not None:
        lines.append(f"  {'Face share of decoder output (CV R^2)':<44} {r['face_share']:.3f}")
    if r.get("leakage_ratio") is not None:
        lines.append(f"  {'Leakage ratio (face gain / EEG gain)':<44} {r['leakage_ratio']:.2f}")
    lines += [""] + [f"- {v}" for v in r["verdicts"]]
    return "\n".join(lines)
