#!/usr/bin/env python3
"""Group analysis of personal state calibration sessions (protocol "states" in the app, recorded by the server).

  python research/personal_states.py --sessions data/sessions --out results/personal_states [--perms 199]

For every participant: windows from the calibration blocks (cleaned exactly like train.py), the personal brain-only
decoder from emotif/states.py, each state's AUC on held-out blocks with its block-permutation p, and the face-only
reference. Across participants (the pre-registered tests):
  H1  per state, AUC > 0.5 across participants (one-sided Wilcoxon signed-rank on AUC - 0.5);
  H2  per state, more participants pass the personal check (p <= 0.05) than the 5 % expected by chance (binomial);
  H3  brain-only held-out accuracy does not depend on how well the face alone separates the states (Spearman).
Manipulation check: the participant's own 0-10 rating of the target state, target block vs other blocks.
Nothing here is a diagnosis; anxiety, anger, low mood and drowsiness are brief, induced states.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotif import datasets, models  # noqa: E402
from emotif import states as ST  # noqa: E402

STATES = datasets.PERSONAL_STATES


def participant_windows(trials, hop=1.0):
    """Per participant: (X, F, y, blocks) from models.extract (filter, ASR, detector; per-session normalization)."""
    subs, _ = models.extract(trials, hop, cleaning=models.Cleaning())
    out = {}
    for sd in subs:
        X, F = models.featurize(sd), models.featurize_face(sd)
        y = sd.labels["state"]
        keep = ~np.isnan(y)
        # drop each block's first 5 s, as the app does
        start = {}
        for i, (tid, s0) in enumerate(sd.keys):
            start.setdefault(sd.groups[i], s0)
        fs = next(t.fs for t in trials if t.subject == sd.subject)
        settle = np.array([(sd.keys[i][1] - start[sd.groups[i]]) / fs >= 5 for i in range(len(y))])
        m = keep & settle
        out[sd.subject] = (X[m], F[m], y[m].astype(int), sd.groups[m])
    return out


def ratings(root):
    """{participant: {state: (mean rating in target blocks, mean rating of that word in other blocks)}} from events."""
    import csv
    res = {}
    for d in sorted(os.listdir(root)):
        meta_p, ev_p = os.path.join(root, d, "meta.json"), os.path.join(root, d, "events.csv")
        if not (os.path.isfile(meta_p) and os.path.isfile(ev_p)):
            continue
        meta = json.load(open(meta_p))
        state_of, vals = {}, {}
        with open(ev_p, newline="") as fh:
            for r in list(csv.reader(fh))[1:]:
                if len(r) < 3:
                    continue
                try:
                    v = json.loads(r[2]) if r[2] else {}
                except ValueError:
                    continue
                if r[1] == "block_start" and isinstance(v, dict) and v.get("state"):
                    state_of[v["block"]] = v["state"]
                if r[1] == "intensity" and isinstance(v, dict) and v.get("block") in state_of:
                    vals.setdefault(state_of[v["block"]], []).append(float(v["value"]))
        if vals:
            res.setdefault(meta.get("participant", d), {}).update({s: float(np.mean(x)) for s, x in vals.items()})
    return res


def main(argv=None):
    warnings.filterwarnings("ignore")
    from scipy.stats import binomtest, spearmanr, wilcoxon
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--perms", type=int, default=199)
    a = p.parse_args(argv)
    trials = datasets.load_sessions(a.sessions, labels="state")
    per = participant_windows(trials)
    res = {"participants": {}}
    for subj, (X, F, y, b) in per.items():
        try:
            r = ST.calibrate(X, F, y, b, STATES, perms=a.perms, seed=1)
        except ValueError as e:
            print(f"{subj}: skipped ({e})")
            continue
        res["participants"][subj] = {"checks": r["checks"], "bacc": r["bacc"], "bacc_p": r["bacc_p"], "face_only": r["face_only"],
                                     "blocks": r["blocks"], "windows": r["windows"]}
        print(f"{subj}: held-out accuracy {r['bacc']:.2f} (p {r['bacc_p']:.3f}); face alone {r['face_only']['bacc']:.2f}; passing: "
              + (", ".join(s for s in STATES if r["checks"][s]["on"]) or "none"))
    P = res["participants"]
    n = len(P)
    res["n"] = n
    res["by_state"] = {}
    for s in STATES:
        aucs = np.array([P[k]["checks"][s]["auc"] for k in P])
        passed = int(sum(P[k]["checks"][s]["on"] for k in P))
        row = {"mean_auc": float(np.nanmean(aucs)) if n else float("nan"), "passed": passed}
        if n >= 5:
            row["wilcoxon_p"] = float(wilcoxon(aucs - 0.5, alternative="greater").pvalue)
            row["binomial_p"] = float(binomtest(passed, n, 0.05, alternative="greater").pvalue)
        res["by_state"][s] = row
        print(f"{s:9s} mean AUC {row['mean_auc']:.2f}; {passed}/{n} pass"
              + (f"; Wilcoxon p {row['wilcoxon_p']:.3f}; binomial p {row['binomial_p']:.3f}" if "wilcoxon_p" in row else ""))
    if n >= 5:
        rho = spearmanr([P[k]["face_only"]["bacc"] for k in P], [P[k]["bacc"] for k in P])
        res["face_vs_brain"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
    res["ratings"] = ratings(a.sessions)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "personal_states.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    main()
