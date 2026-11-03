#!/usr/bin/env python3
"""Quality check of recorded sessions: is the headset reading a brain, and are the face measures working?

  python research/session_qa.py data/sessions/<session folder> [more folders...] [--out results/qa]

For every session: per-electrode share of unusable 2 s windows (lost contact, flat, saturated), mains hum, and the
share of windows the cleaning pipeline repaired or masked. If the session ran the "Headset check" protocol:
  alpha reactivity   alpha (8-13 Hz) power eyes closed / eyes open, per electrode; pass if the best posterior or
                     temporal electrode (Pz, T7, T8, O1, O2, P7, P8) is >= 1.5 (a healthy brain roughly doubles it)
  blink index        0-3 Hz peak-to-peak during "blinks" / eyes open at frontal electrodes; pass if >= 2
  jaw index          20-45 Hz power during "clench" / eyes open at temporal electrodes; pass if >= 3
The "Blink and jaw check" protocol has no eyes-closed minute, so alpha is reported as not run. Ready = contact OK
(every electrode <= 20 % unusable windows), blink and jaw pass, and alpha passes when it was run. Writes <out>/<session>.md and .json.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotif import features as F  # noqa: E402
from emotif import montage as M  # noqa: E402

POSTERIOR = {"PZ", "P7", "P8", "O1", "O2", "T7", "T8", "TP9", "TP10", "P3", "P4", "OZ", "T5", "T6"}
FRONTAL = {"AF3", "AF4", "AF7", "AF8", "FP1", "FP2", "F7", "F8"}
TEMPORAL = {"T7", "T8", "TP9", "TP10", "T3", "T4", "FT7", "FT8"}


def read_session(folder):
    meta = json.load(open(os.path.join(folder, "meta.json")))
    with open(os.path.join(folder, "eeg.csv"), newline="") as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    chans = [h for h in head[1:] if h not in ("valence_label", "arousal_label")]
    idx = [head.index(c) for c in chans]
    body = [r for r in rows[1:] if len(r) == len(head)]
    x = np.array([[float(r[i]) for i in idx] for r in body]).T
    fs = float(meta["fs"])
    blocks = {}
    with open(os.path.join(folder, "events.csv"), newline="") as fh:
        for r in list(csv.reader(fh))[1:]:
            if len(r) < 3:
                continue
            try:
                v = json.loads(r[2]) if r[2] else {}
            except ValueError:
                continue
            if isinstance(v, dict) and "block" in v:
                if r[1] == "block_start":
                    blocks.setdefault(v["block"], [None, None])[0] = float(r[0])
                elif r[1] == "block_end":
                    blocks.setdefault(v["block"], [None, None])[1] = float(r[0])
    return meta, chans, x, fs, {k: v for k, v in blocks.items() if v[0] is not None and v[1] is not None}


def band(xw, fs, lo, hi):
    return F.band_powers(xw, fs, [(lo, hi)])[:, 0]


def segment_power(x, fs, t0, t1, lo, hi, skip=3.0):
    """Median over 2 s windows inside [t0 + skip, t1] of band power per channel."""
    a, b = int((t0 + skip) * fs), int(t1 * fs)
    N = F.window_len(fs)
    seg = x[:, a:b]
    vals = [band(seg[:, s:s + N], fs, lo, hi) for s in range(0, seg.shape[1] - N + 1, N // 2)]
    return np.median(np.array(vals), axis=0) if vals else None


def segment_ptp(x, fs, t0, t1, skip=2.0):
    """Median over 2 s windows of the 0-3 Hz peak-to-peak per channel (the blink index)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, 3.0, "low", fs=fs, output="sos")
    a, b = int((t0 + skip) * fs), int(t1 * fs)
    seg = sosfiltfilt(sos, x[:, a:b], axis=1)
    N = F.window_len(fs)
    vals = [np.ptp(seg[:, s:s + N], axis=1) for s in range(0, seg.shape[1] - N + 1, N // 2)]
    return np.median(np.array(vals), axis=0) if vals else None


def qa(folder):
    meta, chans, x, fs, blocks = read_session(folder)
    xf = F.filter_signal(x, fs)
    names = [M.canonical(c) or c.upper() for c in chans]
    N = F.window_len(fs)
    bad = np.zeros(len(chans))
    n = 0
    for s in F.sliding_windows(xf, fs, 1.0):
        good, _ = F.channel_quality(xf[:, s:s + N], 0.0, F.BAD_UV)
        bad += ~good
        n += 1
    hum = []
    from scipy.signal import welch
    f, P = welch(xf, fs, nperseg=int(2 * fs))
    for c in range(len(chans)):
        line = P[c, (f >= 58) & (f <= 62)].mean() if fs > 130 else np.nan
        ref = P[c, (f >= 40) & (f <= 45)].mean()
        hum.append(float(line / ref) if ref > 0 else np.nan)
    res = {"session": os.path.basename(os.path.normpath(folder)), "participant": meta.get("participant"), "device": meta.get("device"),
           "fs": fs, "minutes": x.shape[1] / fs / 60, "channels": chans,
           "unusable_share": dict(zip(chans, (bad / max(n, 1)).round(3).tolist())),
           "mains_ratio_60hz": dict(zip(chans, [None if np.isnan(h) else round(h, 2) for h in hum])), "checks": {}}
    if {"eyes_open", "eyes_closed"} <= set(blocks):
        eo = segment_power(xf, fs, *blocks["eyes_open"], 8, 13)
        ec = segment_power(xf, fs, *blocks["eyes_closed"], 8, 13)
        ratio = ec / np.maximum(eo, 1e-12)
        sel = [i for i, nm in enumerate(names) if nm in POSTERIOR] or list(range(len(chans)))
        best = float(ratio[sel].max())
        res["checks"]["alpha_reactivity"] = {"per_channel": dict(zip(chans, ratio.round(2).tolist())), "best_posterior": round(best, 2),
                                             "pass": best >= 1.5}
    if {"eyes_open", "blinks"} <= set(blocks):
        b0 = segment_ptp(xf, fs, *blocks["eyes_open"])
        b1 = segment_ptp(xf, fs, *blocks["blinks"])
        r = b1 / np.maximum(b0, 1e-9)
        sel = [i for i, nm in enumerate(names) if nm in FRONTAL] or list(range(len(chans)))
        res["checks"]["blink_index"] = {"per_channel": dict(zip(chans, r.round(2).tolist())), "best_frontal": round(float(r[sel].max()), 2),
                                        "pass": float(r[sel].max()) >= 2}
    if {"eyes_open", "clench"} <= set(blocks):
        j0 = segment_power(xf, fs, *blocks["eyes_open"], 20, 45, skip=2.0)
        j1 = segment_power(xf, fs, *blocks["clench"], 20, 45, skip=2.0)
        r = j1 / np.maximum(j0, 1e-12)
        sel = [i for i, nm in enumerate(names) if nm in TEMPORAL] or list(range(len(chans)))
        res["checks"]["jaw_index"] = {"per_channel": dict(zip(chans, r.round(2).tolist())), "best_temporal": round(float(r[sel].max()), 2),
                                      "pass": float(r[sel].max()) >= 3}
    worst = max(res["unusable_share"].values()) if chans else 1
    res["contact_ok"] = worst <= 0.2
    # the face checks are required; alpha reactivity counts when the eyes-closed minute was run (Headset check)
    res["ready"] = (res["contact_ok"] and {"blink_index", "jaw_index"} <= set(res["checks"])
                    and all(c["pass"] for c in res["checks"].values()))
    return res


def markdown(r):
    L = [f"# Session check: {r['session']}", "",
         f"Participant {r['participant']} · {r['device']} · {r['fs']:.0f} Hz · {r['minutes']:.1f} min", "",
         "| Electrode | Unusable windows | 60 Hz hum ratio |", "|---|---|---|"]
    for c in r["channels"]:
        L.append(f"| {c} | {r['unusable_share'][c]:.0%} | {r['mains_ratio_60hz'][c] if r['mains_ratio_60hz'][c] is not None else '—'} |")
    L.append("")
    names = {"alpha_reactivity": ("Alpha eyes closed / open", "best_posterior", "≥ 1.5"),
             "blink_index": ("Blink index blinks / open", "best_frontal", "≥ 2"), "jaw_index": ("Jaw power clench / open", "best_temporal", "≥ 3")}
    if r["checks"]:
        L += ["| Check | Result | Needed | |", "|---|---|---|---|"]
        for k, (lab, key, need) in names.items():
            if k in r["checks"]:
                c = r["checks"][k]
                L.append(f"| {lab} | {c[key]} | {need} | {'pass' if c['pass'] else 'FAIL'} |")
            else:
                L.append(f"| {lab} | not run | {need} | |")
    else:
        L.append("No check blocks in this session (run Study → Blink and jaw check, or Headset check).")
    L += ["", f"**{'Ready for calibration.' if r['ready'] else 'Not ready: fix contact or repeat the check.'}**"]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+")
    p.add_argument("--out", default="results/qa")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out = []
    for folder in a.sessions:
        r = qa(folder)
        out.append(r)
        with open(os.path.join(a.out, r["session"] + ".json"), "w") as fh:
            json.dump(r, fh, indent=1)
        with open(os.path.join(a.out, r["session"] + ".md"), "w") as fh:
            fh.write(markdown(r))
        print(markdown(r))
    return out


if __name__ == "__main__":
    main()
