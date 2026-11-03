#!/usr/bin/env python3
"""Is a consumer-EEG decoder reading the brain, or the face? Real-data control analysis.

  python research/confound_analysis.py --data muse_mental:data/external/bird_muse --target arousal --out results/confound

1. Pipelines: the same decoder (region features, logistic regression, leave-one-subject-out) trained after
   each cleaning pipeline: filter only; amplitude rejection; ASR; ASR + artifact detector.
2. Artifact-only control: a classifier that sees *only* artifact evidence (the detector's baseline-relative
   features per channel, no EEG band power). If it predicts the label well, the label is confounded with
   muscle / eye activity and accuracy from uncleaned data overstates neural decoding.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotif import artifact as A, features as F, models  # noqa: E402
from train import load_all  # noqa: E402

PIPELINES = [
    ("Filter only", models.Cleaning(asr=False, detector=False, reject_uv=0)),
    ("Filter + amplitude rejection", models.Cleaning(asr=False, detector=False, reject_uv=F.REJECT_UV)),
    ("Filter + ASR", models.Cleaning(asr=True, detector=False, reject_uv=0)),
    ("Filter + ASR + artifact detector", models.Cleaning(asr=True, detector=True, reject_uv=0)),
]


def artifact_only(trials, target, hop, strict=True):
    """LOSO balanced accuracy of a classifier on artifact evidence only. strict=True: the detector's
    p(eye) and p(muscle) per channel (2 numbers per electrode, no spectra). strict=False: all 11
    baseline-relative detector features (these include band-power shape, so they are not purely artifact)."""
    det = A.load()
    by = {}
    for t in trials:
        if target not in t.labels:
            continue
        x = F.filter_signal(t.x, t.fs)
        N = F.window_len(t.fs)
        for s in F.sliding_windows(x, t.fs, hop):
            by.setdefault(t.subject, []).append((A.channel_features(x[:, s:s + N], t.fs), t.labels[target]))
    X, y, g = [], [], []
    for sub, rows in by.items():
        f = np.array([r[0] for r in rows])
        base = A.baseline(f)
        rel = f - base
        if strict:
            pr = np.stack([A.predict_proba(det, w)[:, 1:] for w in rel])     # (n, C, 2)
            X.append(pr.reshape(len(f), -1))
        else:
            X.append(rel.reshape(len(f), -1))
        y.append([r[1] for r in rows])
        g += [sub] * len(rows)
    X, y, g = np.vstack(X), np.concatenate(y), np.array(g)
    yt, yp, per = [], [], {}
    for sub in np.unique(g):
        tr, te = g != sub, g == sub
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(C=0.3, max_iter=4000).fit((X[tr] - mu) / sd, y[tr])
        p = clf.predict((X[te] - mu) / sd)
        yt.append(y[te])
        yp.append(p)
        per[sub] = float(balanced_accuracy_score(y[te], p))
    return float(balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp))), per, len(y)


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--target", default="arousal")
    p.add_argument("--out", default="results/confound")
    p.add_argument("--hop", type=float, default=1.0)
    a = p.parse_args(argv)
    trials = load_all(a.data)
    rows = []
    for name, cl in PIPELINES:
        subs, _, st = models.extract(trials, a.hop, cleaning=cl, return_stats=True)
        r = models.evaluate(subs, a.target, "logreg", 2, modes=("loso",))
        sm = st.summary()
        per = [v["bacc"] for v in r["loso_by_subject"]["full"].values()]
        rows.append({"pipeline": name, "bacc": r["loso"]["full"]["bacc"], "per_subject": per, "chance": r["chance_bacc"],
                     "windows_kept": 1 - sm["rejected"] / max(1, sm["windows"]),
                     "channels_masked": sm["masked_by_detector"] + sm["masked_by_amplitude"],
                     "asr_repaired": sm["asr_repaired_block_fraction"]})
        print(f"{name:36s} LOSO bal. acc {rows[-1]['bacc']:.3f}  kept {rows[-1]['windows_kept']:.0%}  "
              f"masked {rows[-1]['channels_masked']:.0%}  ASR-repaired {rows[-1]['asr_repaired']:.0%}")
    ao, ao_per, n = artifact_only(trials, a.target, a.hop, strict=True)
    print(f"{'Detector eye/muscle probabilities only':36s} LOSO bal. acc {ao:.3f}  ({n} windows)")
    ao_full, _, _ = artifact_only(trials, a.target, a.hop, strict=False)
    print(f"{'All detector features':36s} LOSO bal. acc {ao_full:.3f}")
    os.makedirs(a.out, exist_ok=True)
    res = {"target": a.target, "datasets": a.data, "pipelines": rows,
           "artifact_only": {"bacc": ao, "per_subject": ao_per, "all_detector_features_bacc": ao_full},
           "chance": rows[0]["chance"]}
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    lines = ["| Pipeline | LOSO balanced accuracy | Windows kept | Channel-windows masked |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['pipeline']} | {r['bacc']:.3f} | {r['windows_kept']:.0%} | {r['channels_masked']:.0%} |")
    lines.append(f"| Eye/muscle detector probabilities only (no EEG spectra) | {ao:.3f} | 100% | |")
    lines.append(f"\nChance = {rows[0]['chance']:.3f}. Target: {a.target}.")
    with open(os.path.join(a.out, "table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=170)
    labels = [r["pipeline"].replace("Filter + ", "+ ") for r in rows] + ["Eye/muscle evidence only\n(no EEG spectra)"]
    vals = [r["bacc"] for r in rows] + [ao]
    cols = ["#9aa5b1", "#9aa5b1", "#2f6f9f", "#2f6f9f", "#c2410c"]
    ax.barh(range(len(vals))[::-1], vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, len(vals) - 1 - i, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(rows[0]["chance"], color="#334155", ls="--", lw=1)
    ax.text(rows[0]["chance"] + 0.005, len(vals) - 0.45, "chance", fontsize=7, color="#334155")
    ax.set_yticks(range(len(vals))[::-1], labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy, new subject (LOSO)")
    ax.set_title(f"Real Muse data, {a.target}: what drives the decoder?", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_confound.png"))
    plt.close(fig)
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main()
