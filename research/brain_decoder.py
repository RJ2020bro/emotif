#!/usr/bin/env python3
"""Brain-only decoding experiments: erase every direction of the EEG feature space that co-varies with facial
activity, then decode, and measure both accuracy on new people and how much of the decoder's output the face
can still explain.

  python research/brain_decoder.py --data gameemo_raw:data/external/gameemo --hop 2 --out results/brain_gameemo
  python research/brain_decoder.py --data muse_mental:data/external/bird_muse --targets arousal --out results/brain_muse

Decoders (all LOSO logistic regression on the same windows as the audit):
  eeg_cleaned       cleaned region features (the current Emotif model)
  brain_live        the deployed Emotif decoder (models.BrainModel): cleaned features minus everything predicted by the
                    face indices the app computes live (causal filter, pooled into regions like the EEG)
  eeg_partialled    the audit's reference: cleaned features minus their ridge prediction from face indices; needs the
                    face indices of every test window, so it can't run live
  erased_*          face-subspace erasure (LEACE; Belrose et al. 2023) fitted on the training people only: an affine
                    map after which no linear function of the features co-varies with the face concept. It needs no
                    face measurement at test time and folds into the classifier's weights.
                      erased_global  concept = mean face indices over the scalp (3) and their squares
                      erased_region  concept = face indices pooled into the 17 regions (51) and their squares
Face leakage is always measured with the audit's own face indices (zero-phase, per electrode), not with the indices
the decoder used, so a decoder can't pass by matching its own definition.
Face leakage of a decoder = out-of-person R^2 of its continuous output (expected class value, as the app shows)
predicted from the face indices.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotif import audit, models  # noqa: E402
from emotif import montage as M  # noqa: E402
from emotif.erasure import FaceEraser, face_concept  # noqa: E402


def assemble_all(trials, hop, targets):
    """Like audit.assemble, for several targets at once, plus face indices pooled into regions."""
    def feats(cleaning):
        subs, _ = models.extract(trials, hop, cleaning=cleaning)
        out = {}
        for sd in subs:
            X = models.featurize(sd)
            FL = models.featurize_face(sd)
            for i, key in enumerate(sd.keys):
                out[key] = (sd.subject, f"{sd.subject}/{sd.groups[i]}", {t: sd.labels[t][i] for t in targets}, X[i], sd.channels, FL[i])
        return out
    raw = feats(models.Cleaning(asr=False, detector=False))
    clean = feats(models.Cleaning(asr=True, detector=True))
    art = audit.artifact_evidence(trials, hop)
    keys = sorted(set(raw) & set(clean) & set(art))
    W_cache = {}
    face_reg = []
    for k in keys:
        ch = tuple(clean[k][4])
        if ch not in W_cache:
            W_cache[ch] = M.region_weights(list(ch))[0]
        f = art[k]["index"].reshape(len(ch), 3)
        face_reg.append((W_cache[ch] @ f).reshape(-1))
    d = {
        "subject": np.array([raw[k][0] for k in keys]), "group": np.array([raw[k][1] for k in keys]),
        "eeg_filtered": np.array([raw[k][3] for k in keys]), "eeg_cleaned": np.array([clean[k][3] for k in keys]),
        "face": np.array([art[k]["index"] for k in keys]), "face_region": np.array(face_reg),
        "face_live": np.array([clean[k][5] for k in keys]),
    }
    for t in targets:
        d["y_" + t] = np.array([clean[k][2][t] for k in keys])
    return d


def std(tr, te):
    return audit._standardize(tr, te)


def design(cond, d, tr, te):
    """Training and test design matrices for one LOSO fold."""
    E = d["eeg_cleaned"]
    if cond == "eeg_cleaned":
        return E[tr], E[te]
    if cond == "eeg_filtered":
        return d["eeg_filtered"][tr], d["eeg_filtered"][te]
    if cond == "eeg_partialled":
        return audit._residualize(E[tr], E[te], d["face"][tr], d["face"][te])
    if cond == "brain_live":       # the deployed decoder (models.BrainModel): causal face indices pooled like the EEG
        return audit._residualize(E[tr], E[te], d["face_live"][tr], d["face_live"][te])
    if cond == "partialled_lin":
        return audit._residualize_lin(E[tr], E[te], d["face"][tr], d["face"][te])
    if cond.startswith("erased_"):
        parts = cond.split("_")
        kind, k = parts[1], (int(parts[2][1:]) if len(parts) > 2 else None)
        er = FaceEraser(rank=k).fit(E[tr], face_concept(d["face_region"][tr], kind))
        return er.transform(E[tr]), er.transform(E[te])
    raise ValueError(cond)


def loso(d, cond, y, C=0.3):
    from sklearn.linear_model import LogisticRegression
    pred = np.empty_like(y)
    dec = np.zeros(len(y))
    for s in np.unique(d["subject"]):
        te = d["subject"] == s
        tr = ~te & ~np.isnan(y)
        te &= ~np.isnan(y)
        if not te.any():
            continue
        Xtr, Xte = design(cond, d, tr, te)
        Xtr, Xte = std(Xtr, Xte)
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y[tr])
        pred[te] = clf.predict(Xte)
        if SMOOTH > 1:      # moving average of class probabilities over the last SMOOTH windows of a recording
            P = clf.predict_proba(Xte)
            g = d["group"][te]
            Ps = np.empty_like(P)
            for gg in np.unique(g):
                ix = np.flatnonzero(g == gg)
                c = np.cumsum(P[ix], axis=0)
                lo = np.maximum(np.arange(len(ix)) - SMOOTH, -1)
                Ps[ix] = (c - np.where(lo[:, None] >= 0, c[np.maximum(lo, 0)], 0)) / (np.arange(len(ix)) - lo)[:, None]
            pred[te] = clf.classes_[Ps.argmax(axis=1)]
        dec[te] = clf.predict_proba(Xte) @ clf.classes_.astype(float)   # the app's continuous output
    return pred, dec


def hold_mask(d, ok):
    """Windows the live guard would hold: the largest |standardized pooled face index| above the 99.5th percentile
    of the training people's (models.BrainModel.hold_z), computed per LOSO fold."""
    from emotif import face as FACE
    held = np.zeros(len(ok), bool)
    Q = FACE.design(d["face_live"])[:, :FACE.N_REGION_FACE]
    for s in np.unique(d["subject"]):
        te = (d["subject"] == s) & ok
        tr = (d["subject"] != s) & ok
        q = FACE.design(d["face_live"][tr])
        mu, sd = q.mean(0)[:FACE.N_REGION_FACE], q.std(0)[:FACE.N_REGION_FACE]
        sd = np.where(sd < 1e-9, 1.0, sd)
        hz = np.percentile(np.abs((Q[tr] - mu) / sd).max(axis=1), 99.5)
        held[te] = np.abs((Q[te] - mu) / sd).max(axis=1) > hz
    return held


def face_leakage(d, dec, ok):
    """Adjusted R^2 of the decoder's out-of-person output explained linearly by the face indices and their squares,
    pooled over all held-out windows (how much of what the decoder says the face can account for)."""
    F = d["face"][ok]
    Z = np.hstack([F, F ** 2, np.ones((len(F), 1))])
    v = dec[ok]
    B, *_ = np.linalg.lstsq(Z, v, rcond=None)
    r2 = 1 - np.sum((v - Z @ B) ** 2) / np.sum((v - v.mean()) ** 2)
    n, k = Z.shape
    return float(1 - (1 - r2) * (n - 1) / (n - k))


SMOOTH = 1
CONDS = ["eeg_filtered", "eeg_cleaned", "eeg_partialled", "brain_live", "erased_region"]


def main(argv=None):
    warnings.filterwarnings("ignore")
    from train import load_all
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--targets", default="valence,arousal")
    p.add_argument("--hop", type=float, default=1.0)
    p.add_argument("--perms", type=int, default=0)
    p.add_argument("--conds", default=",".join(CONDS))
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--smooth", type=int, default=1, help="average probabilities over this many consecutive windows")
    a = p.parse_args(argv)
    global SMOOTH
    SMOOTH = a.smooth
    targets = [t for t in a.targets.split(",") if t]
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "cache.npz")
    if os.path.exists(cache):
        d = dict(np.load(cache, allow_pickle=False))
    else:
        trials = load_all(a.data)
        d = assemble_all(trials, a.hop, targets)
        np.savez_compressed(cache, **d)
    print(f"{len(d['subject'])} windows, {len(np.unique(d['subject']))} people")
    res = {}
    rng = np.random.default_rng(0)
    for t in targets:
        y = d["y_" + t]
        ok = ~np.isnan(y)
        classes = np.unique(y[ok])
        res[t] = {}
        perms = []
        if a.perms:
            dd = dict(d, y=np.where(ok, y, 0))
            perms = [audit.permute_recordings(dd, rng) for _ in range(a.perms)]
        for cond in a.conds.split(","):
            pred, dec = loso(d, cond, y)
            bacc = audit.balanced_accuracy(y[ok], pred[ok], classes)
            r = {"bacc": bacc, "face_r2": face_leakage(d, dec, ok)}
            if cond == "brain_live" and "face_live" in d:
                h = hold_mask(d, ok)
                r["held_fraction"] = float(h[ok].mean())
                r["bacc_not_held"] = audit.balanced_accuracy(y[ok & ~h], pred[ok & ~h], classes)
                print(f"         guard holds {r['held_fraction']:.1%} of held-out windows; accuracy on the rest {r['bacc_not_held']:.3f}")
            if perms:
                def null(yp):
                    pp, _ = loso(d, cond, np.where(ok, yp, np.nan))
                    return audit.balanced_accuracy(yp[ok], pp[ok], classes)
                if a.jobs > 1:
                    from joblib import Parallel, delayed
                    nv = np.array(Parallel(n_jobs=a.jobs)(delayed(null)(yp) for yp in perms))
                else:
                    nv = np.array([null(yp) for yp in perms])
                r["p_perm"] = float((1 + np.sum(nv >= bacc)) / (1 + len(nv)))
                r["null_95"] = float(np.percentile(nv, 95))
            res[t][cond] = r
            print(f"{t:8s} {cond:16s} bacc {bacc:.3f}  face R^2 of decoder output {r['face_r2']:+.3f}"
                  + (f"  p = {r['p_perm']:.4f}" if "p_perm" in r else ""), flush=True)
    with open(os.path.join(a.out, f"results_s{SMOOTH}.json"), "w") as fh:
        json.dump({"smooth_windows": SMOOTH, "hop": a.hop, **res}, fh, indent=1)
    figure(a.out)
    return res


NAMES = {"eeg_filtered": "Standard, filter only", "eeg_cleaned": "Standard, cleaned", "brain_live": "Emotif brain-only",
         "eeg_partialled": "Audit reference (offline)", "erased_region": "Full subspace erasure"}


def figure(out):
    """Accuracy (per window and smoothed) and face-explained share of the output, when both result files exist."""
    import glob
    files = sorted(glob.glob(os.path.join(out, "results_s*.json")), key=lambda f: int(f.split("_s")[-1][:-5]))
    if len(files) < 2:
        return
    R = [json.load(open(f)) for f in (files[0], files[-1])]
    targets = [t for t in ("valence", "arousal") if t in R[0]]
    conds = [c for c in NAMES if all(c in R[0][t] for t in targets)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"eeg_filtered": "#9aa5b1", "eeg_cleaned": "#64748b", "brain_live": "#1d4e89", "eeg_partialled": "#7c9cbf", "erased_region": "#c7d2de"}
    fig, axes = plt.subplots(1, len(targets) + 1, figsize=(4.0 * (len(targets) + 1), 3.2), dpi=170,
                             gridspec_kw={"width_ratios": [1.25] * len(targets) + [1.0]})
    sec = R[1]["smooth_windows"] * R[1]["hop"]
    for k, t in enumerate(targets):
        ax = axes[k]
        for i, c in enumerate(conds):
            y = len(conds) - 1 - i
            w, s_ = R[0][t][c]["bacc"], R[1][t][c]["bacc"]
            ax.barh(y, s_, color=col[c], alpha=0.35, height=0.7)
            ax.barh(y, w, color=col[c], height=0.7)
            ax.text(s_ + 0.004, y, f"{w:.3f} / {s_:.3f}", va="center", fontsize=7)
        ax.axvline(0.5, color="#334155", ls="--", lw=0.8)
        ax.set_xlim(0.45, 0.75)
        ax.set_yticks(range(len(conds))[::-1], [NAMES[c] for c in conds] if k == 0 else [""] * len(conds), fontsize=8)
        ax.set_title(f"{t}: accuracy on a new person", fontsize=9)
        ax.set_xlabel(f"balanced accuracy (dark: one 2 s window, light: {sec:.0f} s smoothed)", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[-1]
    for i, c in enumerate(conds):
        y = len(conds) - 1 - i
        vals = [max(0.0, R[0][t][c]["face_r2"]) for t in targets]
        ax.barh(y, np.mean(vals), color=col[c], height=0.7)
        note = "  (scored on the indices it removed)" if c in ("eeg_partialled", "erased_region") else ""
        ax.text(np.mean(vals) + 0.01, y, " / ".join(f"{v:.2f}" for v in vals) + note, va="center", fontsize=7)
    ax.set_xlim(0, 0.8)
    ax.set_yticks([])
    ax.set_title("share of output the face explains", fontsize=9)
    ax.set_xlabel("R² from independent face indices (" + " / ".join(targets) + ")", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_brain_decoder.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
