#!/usr/bin/env python3
"""Train a montage-agnostic Emotif affect model and export JSON for the app.

  python train.py --data gameemo:./GAMEEMO --out models/gameemo.json
  python train.py --data deap:./deap/data_preprocessed_python --data dreamer:./DREAMER.mat --model mlp
  python train.py --data csv:./recordings --cv within --out models/me.json
  python train.py --data muse_mental:./eeg-feature-generation/dataset/original_data

Datasets: seed, deap, dreamer, gameemo, muse_mental, csv, session. Several --data flags are combined into one model:
region features let recordings from different headsets share one feature space. Signals are cleaned exactly as
in the app: ASR artifact repair (calibrated per subject) and the learned artifact detector.
"""
import argparse
import json
import os
import sys
import warnings

from emotif import datasets, models


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", required=True, metavar="NAME:PATH",
                   help=f"dataset to include; NAME is one of {', '.join(datasets.LOADERS)}")
    p.add_argument("--out", default="model.json")
    p.add_argument("--name", default="")
    p.add_argument("--model", default="logreg", choices=list(models.KINDS),
                   help="graph / graph-mlp add region-graph diffusion (montage.graph_features) before the head")
    p.add_argument("--augment", type=int, default=2, help="montage-variant copies per training window (0 = off)")
    p.add_argument("--hop", type=float, default=1.0, help="window hop in seconds")
    p.add_argument("--clean", default="full", choices=["full", "asr", "detector", "none"],
                   help="artifact handling: full = ASR repair + learned detector masking (default, matches the app); "
                        "none = filter only")
    p.add_argument("--asr-k", type=float, default=20.0, help="ASR threshold in robust standard deviations")
    p.add_argument("--reject-uv", type=float, default=0.0,
                   help="also discard windows above this p-p in µV after cleaning (0 = off)")
    p.add_argument("--targets", default="valence,arousal")
    p.add_argument("--cv", default="within,loso", help="comma list of within, loso; or 'none'")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def load_all(specs):
    trials = []
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"--data expects NAME:PATH, got {spec!r}")
        name, path = spec.split(":", 1)
        if name not in datasets.LOADERS:
            raise SystemExit(f"unknown dataset {name!r}; choose from {list(datasets.LOADERS)}")
        if not os.path.exists(path):
            raise SystemExit(f"{path} does not exist")
        got = datasets.LOADERS[name](path)
        print(f"  {name}: {len(got)} recordings, {len({t.subject for t in got})} subjects")
        trials.extend(got)
    return trials


def fmt(res):
    lines = []
    for mode in ("within", "loso"):
        if mode in res:
            row = "  ".join(f"{k}={v['bacc']:.3f}" for k, v in res[mode].items())
            lines.append(f"    {mode:6s} balanced acc: {row}")
    if "chance_bacc" in res:
        lines.append(f"    chance: balanced acc {res['chance_bacc']:.3f}, majority-class acc {res['chance']:.3f}")
    else:
        lines.append("    (no cross-validation run)")
    return "\n".join(lines)


def main(argv=None):
    warnings.filterwarnings("ignore", category=UserWarning)
    a = parse_args(argv)
    print("loading")
    trials = load_all(a.data)
    cleaning = models.Cleaning(asr=a.clean in ("full", "asr"), asr_k=a.asr_k, detector=a.clean in ("full", "detector"),
                               reject_uv=a.reject_uv)
    print(f"cleaning: {cleaning.describe()}")
    subjects, rejected, stats = models.extract(trials, a.hop, cleaning=cleaning, return_stats=True)
    n_win = sum(len(s.groups) for s in subjects)
    sm = stats.summary()
    print(f"{len(subjects)} subjects, {n_win} windows kept, {rejected} discarded; ASR repaired "
          f"{sm['asr_repaired_block_fraction']:.0%} of 0.125 s blocks; detector masked {sm['masked_by_detector']:.0%} of channel-windows")
    cv = () if a.cv == "none" else tuple(m.strip() for m in a.cv.split(",") if m.strip())
    targets = tuple(t.strip() for t in a.targets.split(",") if t.strip())
    model = models.train_and_export(subjects, a.model, a.augment, targets, cv, a.folds, a.seed, a.name)
    model["cleaning"] = {"pipeline": cleaning.describe(), "asr_k": a.asr_k if cleaning.asr else None,
                         "detector": cleaning.detector, "reject_uv": a.reject_uv, **sm}
    for t, r in model["cv"].items():
        print(f"  {t}:\n{fmt(r)}")
        c = r.get("calibration")
        if c:
            print(f"    calibration: temperature {c['temperature']:.2f}, expected calibration error "
                  f"{c['ece_before']:.3f} -> {c['ece_after_heldout']:.3f} (held-out)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(model, fh)
    print(f"wrote {a.out} (heads: {', '.join(model['heads'])})")
    return model


if __name__ == "__main__":
    main()
    sys.exit(0)
