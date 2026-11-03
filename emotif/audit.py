"""Artifact-leakage audit: is an EEG affect decoder reading the brain or the face?

  python -m emotif.audit --data muse_mental:data/external/bird_muse --target arousal --out results/audit

Any study can run this on its own data. On the same windows and the same leave-one-subject-out (LOSO) split it
trains five decoders:

  eeg_filtered     region EEG features, filtering only (what most consumer-EEG studies report)
  eeg_cleaned      region EEG features after ASR + learned artifact detector (Emotif)
  face_index       face evidence only: per electrode, a blink/eye-movement index (peak-to-peak of the
                   0.5-3 Hz signal, filtered over whole recordings) and two muscle indices (20-30 and 30-45 Hz
                   power), each relative to that subject's median. Scalp power above ~20 Hz is dominated by muscle
                   (Whitham et al. 2007), and a change in theta or alpha moves none of them, so this is a
                   *specific* face measure (a genuine beta/gamma change would count as face: conservative).
  face_detector    the learned detector's p(eye) and p(muscle) per electrode: more sensitive, less specific
                   (its inputs include band-power shape, so it can also respond to genuine state changes)
  eeg_partialled   cleaned EEG features minus everything predictable from the face indices and their squares
                   (ridge fit on the training subjects only): EEG information the face cannot explain
  combined         face indices + cleaned EEG

Reading it: face_index near eeg_filtered means the filtered decoder's result can be produced by the face alone;
eeg_partialled above chance means the cleaned EEG carries label information beyond the face. Brain changes
that happen together with facial activity are removed by partialling too, so eeg_partialled is a conservative
lower bound on neural decoding and face_index an upper bound on the face's share.

Statistics:
  * permutation test: labels are shuffled between recordings within each subject (a window keeps its
    recording's label), the whole LOSO is rerun, p = (1 + #null >= observed) / (1 + n_perm). Shuffling whole
    recordings keeps the slow drift inside a recording, which window-level shuffling would wrongly break.
  * 95 % CI: recordings are resampled with replacement within each subject (cluster bootstrap).
  * leakage ratio = (face_index - chance) / (eeg_filtered - chance). Near 1: the filtered decoder's
    advantage over chance can be fully explained by eye and muscle activity.
"""
import argparse
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import artifact as A
from . import features as F
from . import models

CONDITIONS = ["eeg_filtered", "eeg_cleaned", "face_index", "face_detector", "eeg_partialled", "combined"]
LABELS = {
    "eeg_filtered": "EEG, filter only",
    "eeg_cleaned": "EEG, cleaned (ASR + detector)",
    "face_index": "Face only: blink + muscle indices",
    "face_detector": "Face only: detector p(eye), p(muscle)",
    "eeg_partialled": "Cleaned EEG minus what the face explains",
    "combined": "Face indices + cleaned EEG",
}
EOG_BAND = (0.5, 3.0)


def balanced_accuracy(y, p, classes):
    rec = [np.mean(p[y == c] == c) for c in classes if np.any(y == c)]
    return float(np.mean(rec))


def slow_component(x, fs):
    """0.5-3 Hz band of a whole recording (zero-phase). Filtering whole recordings, not 2 s windows, keeps
    edge transients of strong rhythms (e.g. alpha) out of the eye index."""
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(4, EOG_BAND, "bandpass", fs=fs, output="sos"), x, axis=1)


MUSCLE_BANDS = ((20.0, 30.0), (30.0, 45.0))


def _band_log_power(win, fs, bands):
    from scipy.signal import welch
    f, p = welch(win, fs, nperseg=int(round(F.SEG_SEC * fs)), axis=1)
    return np.column_stack([np.log(p[:, (f >= lo) & (f < hi)].sum(axis=1) + 1e-12) for lo, hi in bands])


def face_indices(slow_win, win, fs):
    """(C, 3) per electrode: log peak-to-peak of the 0.5-3 Hz signal (blinks, eye movements) and log power
    at 20-30 and 30-45 Hz (muscle; scalp power above 20 Hz is mostly EMG, Whitham et al. 2007)."""
    return np.column_stack([np.log(np.ptp(slow_win, axis=1) + 1e-9), _band_log_power(win, fs, MUSCLE_BANDS)])


def artifact_evidence(trials, hop):
    """{(trial_id, start): {"index": (2C,), "detector": (2C,)}} from the filtered, uncleaned signal, each
    relative to the subject's median over all windows (the detector's baseline is defined the same way)."""
    det = A.load()
    by_subject = {}
    for t in trials:
        x = t.x if t.filtered else F.filter_signal(t.x, t.fs)
        slow = slow_component(x, t.fs)
        N = F.window_len(t.fs)
        for s in F.sliding_windows(x, t.fs, hop):
            win = x[:, s:s + N]
            bp = F.band_powers(win, t.fs)
            by_subject.setdefault(t.subject, []).append(
                ((t.trial_id, s), A.channel_features(win, t.fs, bp), face_indices(slow[:, s:s + N], win, t.fs)))
    out = {}
    for rows in by_subject.values():
        feats = np.array([r[1] for r in rows])
        idx = np.array([r[2] for r in rows])
        base, ibase = A.baseline(feats), np.median(idx, axis=0)
        for (key, f, ix) in rows:
            out[key] = {"detector": A.predict_proba(det, f - base)[:, 1:].reshape(-1), "index": (ix - ibase).reshape(-1)}
    return out


def eeg_features(trials, hop, cleaning, target):
    """{key: (subject, group, label, (137,) region features)} for windows the pipeline keeps."""
    subs, _ = models.extract(trials, hop, cleaning=cleaning)
    out = {}
    for sd in subs:
        X = models.featurize(sd)
        for i, key in enumerate(sd.keys):
            y = sd.labels[target][i]
            if not np.isnan(y):
                out[key] = (sd.subject, f"{sd.subject}/{sd.groups[i]}", int(y), X[i])
    return out


def assemble(trials, target, hop=1.0, asr_k=None):
    """Aligned arrays over windows that every condition can use."""
    raw = eeg_features(trials, hop, models.Cleaning(asr=False, detector=False), target)
    cl_kw = {} if asr_k is None else {"asr_k": asr_k}
    clean = eeg_features(trials, hop, models.Cleaning(asr=True, detector=True, **cl_kw), target)
    art = artifact_evidence(trials, hop)
    keys = sorted(set(raw) & set(clean) & set(art))
    if not keys:
        raise ValueError("no windows shared by all pipelines")
    return {
        "subject": np.array([raw[k][0] for k in keys]),
        "group": np.array([raw[k][1] for k in keys]),
        "y": np.array([raw[k][2] for k in keys]),
        "eeg_filtered": np.array([raw[k][3] for k in keys]),
        "eeg_cleaned": np.array([clean[k][3] for k in keys]),
        "face": np.array([art[k]["index"] for k in keys]),
        "face_detector": np.array([art[k]["detector"] for k in keys]),
        "n_windows_total": len(raw),
    }


def _standardize(tr, te):
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (tr - mu) / sd, (te - mu) / sd


def _residualize(E_tr, E_te, A_tr, A_te, alpha=1.0):
    """Remove from E the part predictable from A and A**2 (ridge, fit on training rows only). The squared terms
    catch the curvature between an artifact's size and its effect on log band power."""
    A_tr, A_te = np.hstack([A_tr, A_tr ** 2]), np.hstack([A_te, A_te ** 2])
    A_tr_s, A_te_s = _standardize(A_tr, A_te)
    Z_tr = np.hstack([A_tr_s, np.ones((len(A_tr_s), 1))])
    Z_te = np.hstack([A_te_s, np.ones((len(A_te_s), 1))])
    reg = alpha * np.eye(Z_tr.shape[1])
    reg[-1, -1] = 0.0
    B = np.linalg.solve(Z_tr.T @ Z_tr + reg, Z_tr.T @ E_tr)
    return E_tr - Z_tr @ B, E_te - Z_te @ B


def _residualize_lin(E_tr, E_te, A_tr, A_te, alpha=1.0):
    """As _residualize with linear terms only."""
    A_tr_s, A_te_s = _standardize(A_tr, A_te)
    Z_tr = np.hstack([A_tr_s, np.ones((len(A_tr_s), 1))])
    Z_te = np.hstack([A_te_s, np.ones((len(A_te_s), 1))])
    reg = alpha * np.eye(Z_tr.shape[1])
    reg[-1, -1] = 0.0
    B = np.linalg.solve(Z_tr.T @ Z_tr + reg, Z_tr.T @ E_tr)
    return E_tr - Z_tr @ B, E_te - Z_te @ B


def _design(cond, d, tr, te):
    if cond == "eeg_filtered":
        return d["eeg_filtered"][tr], d["eeg_filtered"][te]
    if cond == "eeg_cleaned":
        return d["eeg_cleaned"][tr], d["eeg_cleaned"][te]
    if cond == "face_index":
        return d["face"][tr], d["face"][te]
    if cond == "face_detector":
        return d["face_detector"][tr], d["face_detector"][te]
    if cond == "eeg_partialled":
        return _residualize(d["eeg_cleaned"][tr], d["eeg_cleaned"][te], d["face"][tr], d["face"][te])
    if cond == "combined":
        return (np.hstack([d["face"][tr], d["eeg_cleaned"][tr]]),
                np.hstack([d["face"][te], d["eeg_cleaned"][te]]))
    raise ValueError(cond)


def loso_predict(d, cond, y):
    """Leave-one-subject-out predictions for labels y (which may be permuted)."""
    pred = np.empty_like(y)
    for s in np.unique(d["subject"]):
        te = d["subject"] == s
        tr = ~te
        Xtr, Xte = _design(cond, d, tr, te)
        Xtr, Xte = _standardize(Xtr, Xte)
        clf = LogisticRegression(C=0.3, max_iter=2000).fit(Xtr, y[tr])
        pred[te] = clf.predict(Xte)
    return pred


def permute_recordings(d, rng):
    """Shuffle labels between recordings within each subject."""
    y = d["y"].copy()
    for s in np.unique(d["subject"]):
        ms = d["subject"] == s
        groups = np.unique(d["group"][ms])
        labs = np.array([d["y"][ms & (d["group"] == g)][0] for g in groups])
        perm = rng.permutation(labs)
        for g, l in zip(groups, perm):
            y[ms & (d["group"] == g)] = l
    return y


def cluster_bootstrap(d, pred, classes, n_boot, rng):
    """95 % CI of balanced accuracy, resampling recordings within subjects."""
    idx_by = {}
    for s in np.unique(d["subject"]):
        ms = d["subject"] == s
        idx_by[s] = [np.flatnonzero(ms & (d["group"] == g)) for g in np.unique(d["group"][ms])]
    vals = []
    for _ in range(n_boot):
        rows = []
        for groups in idx_by.values():
            pick = rng.integers(0, len(groups), len(groups))
            rows += [groups[i] for i in pick]
        rows = np.concatenate(rows)
        vals.append(balanced_accuracy(d["y"][rows], pred[rows], classes))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _null_bacc(d, cond, yp, classes):
    return balanced_accuracy(yp, loso_predict(d, cond, yp), classes)


def run(d, n_perm=200, n_boot=2000, seed=0, conditions=CONDITIONS, log=print, n_jobs=1):
    classes = np.unique(d["y"])
    chance = 1.0 / len(classes)
    rng = np.random.default_rng(seed)
    perms = [permute_recordings(d, rng) for _ in range(n_perm)]
    res = {"chance": chance, "n_windows": int(len(d["y"])), "n_subjects": int(len(np.unique(d["subject"]))),
           "n_recordings": int(len(np.unique(d["group"]))), "n_perm": n_perm, "conditions": {}}
    for cond in conditions:
        pred = loso_predict(d, cond, d["y"])
        obs = balanced_accuracy(d["y"], pred, classes)
        if n_jobs == 1:
            null = np.array([_null_bacc(d, cond, yp, classes) for yp in perms])
        else:
            from joblib import Parallel, delayed
            null = np.array(Parallel(n_jobs=n_jobs)(delayed(_null_bacc)(d, cond, yp, classes) for yp in perms))
        per = {s: balanced_accuracy(d["y"][d["subject"] == s], pred[d["subject"] == s], classes)
               for s in np.unique(d["subject"])}
        res["conditions"][cond] = {
            "label": LABELS[cond], "bacc": obs, "ci95": cluster_bootstrap(d, pred, classes, n_boot, rng),
            "p_perm": float((1 + np.sum(null >= obs)) / (1 + n_perm)),
            "null_mean": float(null.mean()) if n_perm else None,
            "null_95": float(np.percentile(null, 95)) if n_perm else None,
            "per_subject": per,
        }
        r = res["conditions"][cond]
        log(f"{LABELS[cond]:44s} {obs:.3f}  95% CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  p = {r['p_perm']:.4f}")
    c = res["conditions"]
    if "face_index" in c and "eeg_filtered" in c:
        denom = c["eeg_filtered"]["bacc"] - chance
        res["leakage_ratio"] = float((c["face_index"]["bacc"] - chance) / denom) if denom > 0 else None
    return res


def markdown(res):
    lines = ["| Decoder (LOSO) | Balanced accuracy | 95 % CI | Permutation p |", "|---|---|---|---|"]
    for cond, r in res["conditions"].items():
        lines.append(f"| {r['label']} | {r['bacc']:.3f} | {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f} | {r['p_perm']:.4f} |")
    lines.append(f"\nChance {res['chance']:.3f}; {res['n_windows']} windows, {res['n_recordings']} recordings, "
                 f"{res['n_subjects']} subjects; {res['n_perm']} recording-level permutations.")
    if res.get("leakage_ratio") is not None:
        lines.append(f"Leakage ratio (face-index gain / filtered-EEG gain over chance): {res['leakage_ratio']:.2f}.")
    return "\n".join(lines) + "\n"


def figure(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(res["conditions"])
    colors = {"eeg_filtered": "#9aa5b1", "eeg_cleaned": "#2f6f9f", "face_index": "#c2410c",
              "face_detector": "#e08a5c", "eeg_partialled": "#1d4e3f", "combined": "#6b7280"}
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=170)
    for i, c in enumerate(conds):
        r = res["conditions"][c]
        yv = len(conds) - 1 - i
        ax.barh(yv, r["bacc"], color=colors[c], height=0.62)
        ax.errorbar(r["bacc"], yv, xerr=[[r["bacc"] - r["ci95"][0]], [r["ci95"][1] - r["bacc"]]],
                    fmt="none", ecolor="#111827", elinewidth=0.9, capsize=2.5)
        if r["null_95"] is not None:
            ax.plot([r["null_95"]] * 2, [yv - 0.36, yv + 0.36], color="#111827", lw=1.2, ls=":")
        star = "*" if r["p_perm"] < 0.05 else ""
        ax.text(r["ci95"][1] + 0.012, yv, f"{r['bacc']:.2f}{star}", va="center", fontsize=8)
    ax.axvline(res["chance"], color="#334155", ls="--", lw=0.9)
    ax.set_yticks(range(len(conds))[::-1], [res["conditions"][c]["label"] for c in conds], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy on a new person (LOSO); bars 95 % CI, dotted 95th pct of null", fontsize=7.5)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    warnings.filterwarnings("ignore")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train import load_all
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", required=True, metavar="NAME:PATH")
    p.add_argument("--target", default="arousal")
    p.add_argument("--hop", type=float, default=1.0)
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1, help="parallel workers for the permutations")
    p.add_argument("--out", default="results/audit")
    a = p.parse_args(argv)
    trials = [t for t in load_all(a.data) if a.target in t.labels]
    d = assemble(trials, a.target, a.hop)
    print(f"{len(d['y'])} windows shared by all pipelines (of {d['n_windows_total']})")
    res = run(d, a.perms, a.boot, a.seed, n_jobs=a.jobs)
    res.update({"target": a.target, "datasets": a.data, "hop": a.hop})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out, "table.md"), "w") as fh:
        fh.write(markdown(res))
    figure(res, os.path.join(a.out, "fig_audit.png"), f"Artifact-leakage audit: {', '.join(x.split(':')[0] for x in a.data)}, {a.target}")
    print(markdown(res))
    return res


if __name__ == "__main__":
    main()
