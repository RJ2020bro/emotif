#!/usr/bin/env python3
"""Train on SYNTHETIC Emotiv Insight sessions (scripts/synth_insight.py) and print held-out accuracy.

  python scripts/synth_insight.py --participants 8 --protocol face,states,induction
  python scripts/train_synthetic.py                       # reads data/synthetic, writes models/synthetic

  1. Personal states, per simulated participant (protocol "states"): the app's brain-only decoder (emotif/states.py),
     each state checked on calibration blocks it never saw, with a block-permutation test. Exported in the app's
     format (emotif-states-v1) as models/synthetic/personal_states_<code>.json.
  2. Valence/arousal model across participants (protocols with SAM ratings: induction, closedloop) through train.py,
     leave-one-participant-out. Exported as models/synthetic/affect_synthetic.json.

Every input session must be marked "synthetic": true in meta.json (real recordings are refused), and models are
written only to models/synthetic/ (git-ignored) or a folder outside the repository, never over the real models in
models/. Nothing printed here is a research result.
"""
import argparse
import datetime
import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
from emotif import datasets  # noqa: E402
from emotif import states as ST  # noqa: E402

SYN_MODELS = os.path.join(ROOT, "models", "synthetic")


def session_dirs(root):
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "meta.json")))


def check_out(out):
    """Models go to models/synthetic/ or outside the repository, never next to the real models or into results/."""
    o = os.path.abspath(out)
    inside = lambda base: o == base or o.startswith(base + os.sep)  # noqa: E731
    if inside(ROOT) and not inside(SYN_MODELS):
        raise SystemExit(f"refusing to write synthetic models to {out}: use models/synthetic/ (or a folder outside the repository)")
    return o


def export_states(r, code):
    K = len(r["states"])
    return {
        "format": "emotif-states-v1", "states": list(r["states"]),
        "head": {"classes": list(range(K)), "layers": [{"W": np.asarray(r["layer"]["W"]).tolist(), "b": np.asarray(r["layer"]["b"]).tolist(), "act": "softmax"}],
                 "face": {"mu": np.asarray(r["face"]["mu"]).tolist(), "sd": np.asarray(r["face"]["sd"]).tolist(), "hold_z": float(r["face"]["hold_z"])}},
        "checks": r["checks"], "faceOnly": r["face_only"], "bacc": r["bacc"], "baccP": r["bacc_p"],
        "windows": int(r["windows"]), "blocks": int(r["blocks"]), "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "synthetic": True, "name": f"SYNTHETIC personal states · {code}",
    }


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", default=os.path.join(ROOT, "data", "synthetic"))
    p.add_argument("--out", default=SYN_MODELS)
    p.add_argument("--perms", type=int, default=99, help="block permutations per participant for the state checks")
    p.add_argument("--skip-affect", action="store_true", help="only the personal states")
    a = p.parse_args(argv)
    out = check_out(a.out)
    dirs = session_dirs(a.sessions)
    if not dirs:
        raise SystemExit(f"No sessions in {a.sessions}. Make some first: python scripts/synth_insight.py --participants 8 --protocol face,states,induction")
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        if meta.get("synthetic") is not True:
            raise SystemExit(f"{d} is not marked synthetic; this script only trains on scripts/synth_insight.py output.")
    os.makedirs(out, exist_ok=True)
    import personal_states as PS
    report = {"synthetic": True, "sessions": a.sessions, "personal_states": {}, "affect": None}

    # 1. personal states, per participant
    try:
        trials = datasets.load_sessions(a.sessions, labels="state", synthetic=True)
    except FileNotFoundError:
        trials = []
    if trials:
        print("Personal states (held-out calibration blocks, chance = 1/6 = 0.17):")
        for subj, (X, F, y, b) in PS.participant_windows(trials).items():
            code = subj.split("@")[0]
            try:
                r = ST.calibrate(X, F, y, b, datasets.PERSONAL_STATES, perms=a.perms, seed=1)
            except ValueError as e:
                print(f"  {code}: skipped ({e})")
                continue
            path = os.path.join(out, f"personal_states_{code}.json")
            with open(path, "w") as fh:
                json.dump(export_states(r, code), fh)
            on = [s for s in datasets.PERSONAL_STATES if r["checks"][s]["on"]]
            print(f"  {code}: accuracy {r['bacc']:.2f} (permutation p {r['bacc_p']:.3f}), face alone {r['face_only']['bacc']:.2f}; "
                  f"AUC " + " ".join(f"{s} {r['checks'][s]['auc']:.2f}" for s in datasets.PERSONAL_STATES)
                  + f"; passing {len(on)}/6 -> {os.path.relpath(path, ROOT) if path.startswith(ROOT) else path}")
            report["personal_states"][code] = {"bacc": r["bacc"], "bacc_p": r["bacc_p"], "face_only": r["face_only"]["bacc"],
                                               "auc": {s: r["checks"][s]["auc"] for s in datasets.PERSONAL_STATES}, "passing": on}
        if report["personal_states"]:
            acc = [v["bacc"] for v in report["personal_states"].values()]
            print(f"  mean held-out accuracy {np.mean(acc):.2f} over {len(acc)} simulated participants (chance 0.17)")
    else:
        print("No state calibration sessions (protocol states) found; skipping personal states.")

    # 2. valence / arousal across participants
    if not a.skip_affect:
        import train
        try:
            rated = datasets.load_sessions(a.sessions, labels="sam", synthetic=True)
        except FileNotFoundError:
            rated = []
        if len({t.subject.split("@")[0] for t in rated}) >= 2:
            path = os.path.join(out, "affect_synthetic.json")
            print("Valence/arousal model (train.py, leave one simulated participant out):")
            m = train.main(["--data", f"synthetic:{a.sessions}", "--cv", "loso", "--out", path, "--name", "SYNTHETIC Insight (not research data)"])
            m["synthetic"] = True
            m["notes"] = "Trained on simulated Emotiv Insight sessions from scripts/synth_insight.py. Not research data."
            with open(path, "w") as fh:
                json.dump(m, fh)
            report["affect"] = {t: {"loso_bacc": m["cv"][t]["loso"]["full"]["bacc"], "chance_bacc": m["cv"][t]["chance_bacc"]} for t in m["cv"]}
        else:
            print("Fewer than two participants with SAM-rated sessions (protocol induction or closedloop); skipping the valence/arousal model.")
    with open(os.path.join(out, "report_synthetic.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report


if __name__ == "__main__":
    main()
