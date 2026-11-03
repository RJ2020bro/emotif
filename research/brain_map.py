#!/usr/bin/env python3
"""What the brain-only decoder reads: weights of the face-removed decoder per scalp region and band, and how
consistent each weight's sign is across the 28 leave-one-person-out folds.

  python research/brain_map.py --cache results/brain_gameemo_v2/cache.npz --out results/brain_gameemo_v2

Weights are logistic-regression coefficients on standardized, face-removed features (models.BrainModel's residual),
toward the positive class (pleasant / high arousal). Stability: the people are resampled with replacement 200 times
and the decoder refitted; a region-band is outlined when >= 95 % of refits agree in sign. Only scalp regions the
headset covers are drawn.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotif import audit  # noqa: E402
from emotif import montage as M  # noqa: E402

BANDS = M.BAND_NAMES
R = len(M.REGIONS)


def weights(E, FL, y, rows):
    rows = np.asarray(rows)
    from sklearn.linear_model import LogisticRegression
    Xr, _ = audit._residualize(E[rows], E[rows][:1], FL[rows], FL[rows][:1])
    mu, sd = Xr.mean(0), Xr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    clf = LogisticRegression(C=0.3, max_iter=3000).fit((Xr - mu) / sd, y[rows])
    return clf.coef_[0][:R * len(BANDS)].reshape(R, len(BANDS))


def xy(region):
    """Azimuthal equidistant projection from the vertex: x to the right ear, y to the nose."""
    p = M.POSITIONS[region.upper()]
    th = np.arccos(np.clip(p[2], -1, 1))
    ph = np.arctan2(p[1], p[0])
    return th * np.cos(ph), th * np.sin(ph)


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = dict(np.load(a.cache))
    E, FL, subj = d["eeg_cleaned"], d["face_live"], d["subject"]
    res = {}
    for t in ("valence", "arousal"):
        y = d["y_" + t]
        full = weights(E, FL, y, np.arange(len(y)))
        people = np.unique(subj)
        by = {s: np.flatnonzero(subj == s) for s in people}
        rng = np.random.default_rng(0)
        boots = np.array([weights(E, FL, y, np.concatenate([by[s] for s in rng.choice(people, len(people))]))
                          for _ in range(200)])
        agree = np.maximum((boots > 0).mean(0), (boots < 0).mean(0))
        res[t] = {"weights": full.tolist(), "sign_agreement": agree.tolist()}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "brain_map.json"), "w") as fh:
        json.dump({"regions": M.REGIONS, "bands": BANDS, **res}, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(2, len(BANDS), figsize=(11, 4.9), dpi=170)
    pts = np.array([xy(r) for r in M.REGIONS])
    seen = np.abs(d["eeg_cleaned"][:, -R:]).max(axis=0) > 0          # regions the headset covers
    for i, t in enumerate(("valence", "arousal")):
        W = np.array(res[t]["weights"])
        A = np.array(res[t]["sign_agreement"])
        lim = np.abs(W).max()
        for j, b in enumerate(BANDS):
            ax = axes[i, j]
            ax.add_patch(Circle((0, 0), 1.75, fill=False, lw=0.8, color="#475569"))
            ax.plot([-0.15, 0, 0.15], [1.72, 1.92, 1.72], color="#475569", lw=0.8)
            sc = ax.scatter(pts[seen, 0], pts[seen, 1], c=W[seen, j], cmap="RdBu_r", vmin=-lim, vmax=lim, s=170,
                            edgecolors=["#111827" if A[r, j] >= 0.95 else "#cbd5e1" for r in np.flatnonzero(seen)], linewidths=1.3)
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.05)
            ax.set_aspect("equal")
            ax.axis("off")
            if i == 0:
                ax.set_title(b, fontsize=10)
            if j == 0:
                ax.text(-2.15, 0, "pleasant" if t == "valence" else "high arousal", rotation=90, va="center", ha="center", fontsize=9)
    cb = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("weight after face removal (red: higher power -> class)", fontsize=8)
    fig.suptitle("What the brain-only decoder reads (GAMEEMO, 28 people, 14-ch Emotiv). Dark outline: same sign in >= 95 % of 200 bootstrap refits over people", fontsize=9.5)
    fig.savefig(os.path.join(a.out, "fig_brain_map.png"), bbox_inches="tight")
    plt.close(fig)
    for t in ("valence", "arousal"):
        W, A = np.array(res[t]["weights"]), np.array(res[t]["sign_agreement"])
        top = np.dstack(np.unravel_index(np.argsort(-np.abs(W).ravel()), W.shape))[0][:6]
        print(t, "; ".join(f"{BANDS[b]} {M.REGIONS[r]} {W[r, b]:+.2f} ({A[r, b]:.0%})" for r, b in top))


if __name__ == "__main__":
    main()
