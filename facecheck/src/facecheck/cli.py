"""facecheck audit DATA.npz [--model logreg|forest] [--json OUT]

DATA.npz holds arrays X (windows x features), y, recording, subject (one per window, windows in time order within
each recording) and optionally face (windows x k face indices).
"""
import argparse
import json
import sys

import numpy as np

from .core import audit, report


def main(argv=None):
    p = argparse.ArgumentParser(prog="facecheck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("data")
    a.add_argument("--model", default="logreg", choices=["logreg", "forest"])
    a.add_argument("--scrambles", type=int, default=3)
    a.add_argument("--json")
    args = p.parse_args(argv)
    d = np.load(args.data, allow_pickle=False)
    missing = [k for k in ("X", "y", "recording", "subject") if k not in d.files]
    if missing:
        sys.exit(f"{args.data}: missing arrays {missing}")
    r = audit(d["X"], d["y"], d["recording"], d["subject"], d["face"] if "face" in d.files else None,
              model=args.model, n_scrambles=args.scrambles)
    print(report(r))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(r, fh, indent=1)
    return 0


if __name__ == "__main__":
    main()
