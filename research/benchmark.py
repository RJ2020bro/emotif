#!/usr/bin/env python3
"""Cross-montage benchmark: does montage augmentation keep accuracy when a model meets fewer electrodes?

  python research/benchmark.py --data muse_mental:data/external/bird_muse --out results/muse
  python research/benchmark.py --data gameemo:./GAMEEMO --data dreamer:./DREAMER.mat --out results/emotiv

For every target and model it runs leave-one-subject-out evaluation with montage augmentation off and on,
tests the held-out subject on the full montage and on every consumer headset layout the dataset can
imitate, and writes:
  results.json      all scores, including per-subject balanced accuracy
  table.md          paper-ready table (mean ± 95 % CI across held-out subjects, Wilcoxon p aug vs no-aug)
  fig_montage_<target>.png   balanced accuracy by montage, aug off vs on
  fig_regions_<target>.png   scalp map of the logistic model's alpha-band weights
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotif import features as F, models, montage as M  # noqa: E402
from train import load_all  # noqa: E402


def ci95(x):
    from scipy import stats
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def wilcoxon_p(a, b):
    from scipy import stats
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.allclose(a, b):
        return float("nan")
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def project(v):
    """Azimuthal equidistant projection of a unit vector (z up) to the 2-D head disk."""
    x, y, z = v
    theta = np.arccos(np.clip(z, -1, 1))
    r = theta / (np.pi / 2) * 0.5
    phi = np.arctan2(y, x)
    return r * np.cos(phi), r * np.sin(phi)


def plot_montage(res_off, res_on, target, path, chance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in res_on["loso_by_subject"] if n in res_off.get("loso_by_subject", {})]
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2, 3.6), dpi=160)
    w = 0.38
    for k, (res, lab, col) in enumerate(((res_off, "No augmentation", "#9aa5b1"), (res_on, "Montage augmentation", "#2f6f9f"))):
        means, errs = [], []
        for n in names:
            m, e = ci95([s["bacc"] for s in res["loso_by_subject"][n].values()])
            means.append(m)
            errs.append(e)
        xs = np.arange(len(names)) + (k - 0.5) * w
        means, errs = np.array(means), np.array(errs)
        yerr = np.vstack([np.minimum(errs, means), np.minimum(errs, 1 - means)])   # keep bars inside [0, 1]
        ax.bar(xs, means, w, yerr=yerr, capsize=3, label=lab, color=col, edgecolor="none")
    ax.axhline(chance, color="#c2410c", lw=1, ls="--", label=f"Chance ({chance:.2f})")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy (LOSO)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{target.capitalize()}: held-out subjects, by electrode layout")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_regions(model, target, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    head = model["heads"][target]
    if len(head["layers"]) != 1:
        return False
    W = np.array(head["layers"][0]["W"])          # (features, classes)
    contrast = W[:, -1] - W[:, 0]                   # highest class minus lowest class
    alpha = [contrast[i * 5 + 2] for i in range(len(M.REGIONS))]
    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=160)
    ax.add_patch(plt.Circle((0, 0), 0.5, fill=False, lw=1.2, color="#334155"))
    ax.plot([-0.05, 0, 0.05], [0.49, 0.56, 0.49], color="#334155", lw=1.2)
    lim = max(1e-9, float(np.max(np.abs(alpha))))
    for r, a in zip(M.REGIONS, alpha):
        x, y = project(M.POSITIONS[r.upper()])
        ax.scatter([x], [y], s=520, c=[a], cmap="RdBu_r", vmin=-lim, vmax=lim, edgecolors="#334155", linewidths=0.6)
        ax.text(x, y, r, ha="center", va="center", fontsize=6.5)
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.62, 0.62)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{target.capitalize()}: alpha-band weight\n(red = pushes toward high {target})", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", default="logreg")
    p.add_argument("--augment", type=int, default=2)
    p.add_argument("--hop", type=float, default=1.0)
    p.add_argument("--reject-uv", type=float, default=0.0)
    p.add_argument("--clean", default="full", choices=["full", "asr", "detector", "none"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    trials = load_all(a.data)
    cleaning = models.Cleaning(asr=a.clean in ("full", "asr"), detector=a.clean in ("full", "detector"),
                               reject_uv=a.reject_uv or 0.0)
    subjects, rejected = models.extract(trials, a.hop, cleaning=cleaning)
    if len(subjects) < 2:
        raise SystemExit("cross-montage benchmark needs at least two subjects (LOSO)")
    print(f"{len(subjects)} subjects, {sum(len(s.groups) for s in subjects)} windows, {rejected} rejected")

    out = {"datasets": a.data, "cleaning": cleaning.describe(), "n_subjects": len(subjects), "rejected": rejected, "runs": {}}
    lines = ["| Target | Model | Layout | No aug. bal. acc | Aug. bal. acc | Wilcoxon p |", "|---|---|---|---|---|---|"]
    for target in models.TARGETS:
        if all(np.isnan(sd.labels[target]).all() for sd in subjects):
            continue
        for kind in [k.strip() for k in a.models.split(",")]:
            off = models.evaluate(subjects, target, kind, 0, modes=("loso",), seed=a.seed)
            on = models.evaluate(subjects, target, kind, a.augment, modes=("loso",), seed=a.seed)
            out["runs"][f"{target}/{kind}"] = {"no_aug": off, "aug": on}
            for lay in on["loso_by_subject"]:
                so = off["loso_by_subject"].get(lay, {})
                sa = on["loso_by_subject"][lay]
                subs = sorted(set(so) & set(sa))
                bo = [so[s]["bacc"] for s in subs]
                ba = [sa[s]["bacc"] for s in subs]
                mo, eo = ci95(bo)
                ma, ea = ci95(ba)
                pv = wilcoxon_p(bo, ba)
                lines.append(f"| {target} | {kind} | {lay} | {mo:.3f} ± {eo:.3f} | {ma:.3f} ± {ea:.3f} | "
                             f"{'n/a' if np.isnan(pv) else f'{pv:.3f}'} |")
            plot_montage(off, on, target, os.path.join(a.out, f"fig_montage_{target}_{kind}.png"), on["chance_bacc"])
            print(f"{target}/{kind}: done")
        full = models.train_and_export(subjects, "logreg", a.augment, (target,), (), seed=a.seed)
        plot_regions(full, target, os.path.join(a.out, f"fig_regions_{target}.png"))
    lines.append("")
    lines.append(f"Mean ± 95 % CI across {len(subjects)} held-out subjects (leave-one-subject-out). "
                 f"Chance balanced accuracy = 1 / number of classes.")
    with open(os.path.join(a.out, "table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main()
