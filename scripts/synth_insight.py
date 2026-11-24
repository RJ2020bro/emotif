#!/usr/bin/env python3
"""SYNTHETIC Emotiv Insight sessions, in the Emotif recorder's folder format (meta.json, eeg.csv, events.csv).

  python scripts/synth_insight.py --participants 8 --protocol face,states --seed 1 --out data/synthetic

Every session is simulated: no person wore a headset. meta.json carries "synthetic": true, the folder names start
with "synthetic-", and the default output folder (data/synthetic/) is git-ignored. Use them to exercise and test the
pipeline (session_qa, personal state calibration, train.py), never as research data: keep them out of results/ and
docs/RESULTS.md.

The signal model is the app's simulator (app/src/sources.ts Simulator), with the same parameters
(emotif/data/synthetic_states.json): a 1/f background, six band oscillators whose amplitudes follow a brain
signature per state (calm, stress, anxiety, anger, low mood, drowsiness) or per valence/arousal, spontaneous
blinks, blink and jaw-clench tasks (jaw muscle activity is high-passed above 25 Hz and only on T7/T8), and slow
rolling eye movements when drowsy. Each simulated participant differs (alpha peak, amplitude, channel gains, how
strongly each state shows). --effect scales every state difference (0 = states identical, a null check).

Protocols (as in app/src/protocol.ts): states (personal state calibration: 3 rounds x 6 one-minute blocks with a
0-10 intensity rating), face (blink and jaw check), check (headset check: adds eyes open / eyes closed minutes),
closedloop (stress then recovery x 3 with SAM ratings; recovery is identical in every condition, so there is no
planted condition effect), induction (four affect quadrants with SAM ratings). Comma-separate several to record one
session of each per participant. Needs numpy only.
"""
import argparse
import csv
import datetime
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = json.load(open(os.path.join(ROOT, "emotif", "data", "synthetic_states.json")))
_POS = {k.upper(): v for k, v in json.load(open(os.path.join(ROOT, "emotif", "data", "electrodes_1005.json")))["positions"].items()}
CHANNELS = ["AF3", "T7", "Pz", "T8", "AF4"]        # the order the Insight streams them in (Cortex)
FS = 128.0
DC_UV = 4200.0                                     # the Insight's raw EEG sits on a ~4200 µV offset
STATES = ["calm", "stress", "anxiety", "anger", "low_mood", "drowsy"]     # app/src/states.ts STATES
SIG_KEYS = ["delta", "theta", "thetaF", "alpha", "alphaP", "alphaF", "faa", "beta", "betaF", "betaR", "hibeta", "gamma",
            "alphaHz", "blinkRate", "blinkSize", "emg", "roll"]
MULT = {"blinkRate", "blinkSize"}                  # multipliers (neutral 1); every other key is neutral at 0
NEUTRAL = np.array([1.0 if k in MULT else 0.0 for k in SIG_KEYS])
KI = {k: i for i, k in enumerate(SIG_KEYS)}
PROTOCOLS = ("states", "face", "check", "closedloop", "induction")
# app/src/protocol.ts STATE_TASKS: state -> (task, design)
STATE_TASKS = {"calm": ("breathing", (1, -1)), "stress": ("arithmetic", (-1, 1)), "anxiety": ("anticipation", (-1, 1)),
               "anger": ("recall_anger", (-1, 1)), "low_mood": ("recall_negative", (-1, -1)), "drowsy": ("drowsy", (0, -1))}


def state_sig(name):
    s = SPEC["states"][name]
    return np.array([float(s.get(k, NEUTRAL[KI[k]])) for k in SIG_KEYS])


def va_sig(v, a):
    k = SPEC["va"]
    out = NEUTRAL.copy()
    for key, val in (("alpha", k["alpha"] * a), ("beta", k["beta"] * a), ("hibeta", k["hibeta"] * a), ("gamma", k["gamma"] * a),
                     ("theta", k["theta"] * a), ("faa", k["faa"] * v)):
        out[KI[key]] = val
    return out


def regions(name):
    x, y, z = _POS[name.upper()]
    front = float(np.clip((y - 0.2) / 0.5, 0, 1))
    return {"front": front, "post": float(np.clip((-y - 0.2) / 0.5, 0, 1)), "side": float(np.sign(x)) if abs(x) > 0.1 else 0.0,
            "temporal": 1.0 if abs(x) > 0.7 and z < 0.45 else 0.0, "blinkW": max(0.0, (front - 0.3) / 0.7) ** 2,
            "rollW": float(x * np.clip((y + 0.3) / 0.8, 0, 1))}


# ---------------------------------------------------------------- noise processes (FFT shaping: exact spectra)
def ar1_noise(rng, shape, tau, sd, fs):
    """Stationary AR(1) / Ornstein-Uhlenbeck noise with time constant tau (s) and standard deviation sd."""
    n = shape[-1]
    a = np.exp(-1.0 / (fs * tau))
    w = np.fft.rfftfreq(n) * 2 * np.pi
    H = np.sqrt(1 - a * a) / (1 - a * np.exp(-1j * w))
    return sd * np.fft.irfft(np.fft.rfft(rng.normal(0, 1, shape), axis=-1) * H, n=n, axis=-1)


def highpass_noise(rng, shape, fc, fs):
    """Unit-variance white noise through a 4th-order Butterworth high-pass magnitude (the app's biquad pair)."""
    n = shape[-1]
    f = np.fft.rfftfreq(n, 1 / fs)
    H = np.sqrt((f / fc) ** 8 / (1 + (f / fc) ** 8))
    return np.fft.irfft(np.fft.rfft(rng.normal(0, 1, shape), axis=-1) * H, n=n, axis=-1)


def lagged(values, lengths, tau, fs, start):
    """First-order lag toward a piecewise-constant target (values: segments x K), continuous across segments."""
    out, cur = [], np.asarray(start, dtype=float)
    for v, n in zip(values, lengths):
        if n == 0:
            continue
        decay = np.exp(-np.arange(1, n + 1) / (fs * tau))[:, None]
        seg = v[None, :] + (cur - v)[None, :] * decay
        out.append(seg)
        cur = seg[-1]
    return np.vstack(out)


# ---------------------------------------------------------------- protocols
def drive(state=None, va=(0.0, 0.0), ec=False, blinks=False, clench=False):
    return {"state": state, "va": va, "ec": ec, "blinks": blinks, "clench": clench}


def seg(sec, d, block=None, after=()):
    """One stretch of recording: `sec` seconds of drive `d`; `block` (a block_start value) brackets it with
    block_start / block_end; `after` events are written at its end (e.g. a rating saved after a pause)."""
    return {"sec": float(sec), "drive": d, "block": block, "after": list(after)}


def blk(bid, task, design, condition="mirror", state=None):
    v = {"block": bid, "task": task, "condition": condition, "design": {"v": design[0], "a": design[1]}}
    if state:
        v["state"] = state
    return v


def sam_event(rng, bid, va):
    r = lambda x: int(np.clip(np.rint(5 + 4 * x + rng.normal(0, 0.6)), 1, 9))  # noqa: E731
    return ("sam", {"block": bid, "valence": r(va[0]), "arousal": r(va[1])})


def build(protocol, rng, block_sec=None):
    """Segments and the protocol's own event list, mirroring app/src/protocol.ts."""
    rest = drive()
    pre = [seg(5, rest), seg(10, rest)]          # recording starts; instruction card (protocol_start between them)
    if protocol in ("face", "check"):
        order = ["eyes_open", "eyes_closed", "blinks", "clench"] if protocol == "check" else ["eyes_open", "blinks", "clench"]
        sec_open = 60 if protocol == "check" else 20
        segs = [seg(30, rest, after=())]
        segs.append(seg(sec_open, rest, blk("eyes_open", "rest", (0, 0))))
        if protocol == "check":
            segs.append(seg(60, drive(va=(0, -0.2), ec=True), blk("eyes_closed", "eyes_closed", (0, 0))))
        segs += [seg(20, drive(blinks=True), blk("blinks", "blinks", (0, 0))), seg(20, drive(clench=True), blk("clench", "clench", (0, 0)))]
        base = 30
    elif protocol == "states":
        L = block_sec or 60
        segs, order = [seg(90, rest)], []
        for rnd in range(1, 4):
            sts = [STATES[i] for i in rng.permutation(len(STATES))]
            for i, st in enumerate(sts):
                bid = f"{st}_{rnd}"
                order.append(bid)
                task, design = STATE_TASKS[st]
                segs.append(seg(L, drive(state=st, va=tuple(SPEC["states"][st]["va"])), blk(bid, task, design, state=st)))
                val = int(np.clip(np.rint(rng.normal(6.8, 1.2)), 0, 10))
                segs.append(seg(rng.uniform(4, 8), rest, after=[("intensity", {"block": bid, "value": val})]))
                if i < len(sts) - 1 or rnd < 3:
                    segs.append(seg(15, rest, blk(f"rest_{rnd}_{i + 1}", "rest", (0, 0))))
        segs.append(seg(60, drive(state="calm", va=tuple(SPEC["states"]["calm"]["va"])), blk("cooldown", "breathing", (1, -1))))
        base = 90
    elif protocol == "closedloop":
        L = block_sec or 60
        conds = [["mirror", "guide", "sham"][i] for i in rng.permutation(3)]
        segs, order = [seg(90, rest)], conds
        calm_va = tuple(SPEC["states"]["calm"]["va"])
        for i, c in enumerate(conds):
            segs.append(seg(L, drive(state="stress", va=tuple(SPEC["states"]["stress"]["va"])), blk(f"stress{i + 1}", "arithmetic", (-1, 1))))
            segs.append(seg(2.5 * L, drive(state="calm", va=calm_va), blk(f"recover{i + 1}", "watch", (1, -1), condition=c)))
            segs.append(seg(rng.uniform(8, 12), rest, after=[sam_event(rng, f"recover{i + 1}", calm_va)]))
        base = 90
    elif protocol == "induction":
        L = block_sec or 120
        quads = [("happy", "recall_positive", (1, 1), None, (0.7, 0.6)), ("calm", "breathing", (1, -1), "calm", None),
                 ("stress", "arithmetic", (-1, 1), "stress", None), ("sad", "recall_negative", (-1, -1), "low_mood", None)]
        quads = [quads[i] for i in rng.permutation(4)]
        segs, order = [seg(90, rest)], [q[0] for q in quads]
        for i, (bid, task, design, st, va) in enumerate(quads):
            va = va or tuple(SPEC["states"][st]["va"])
            segs.append(seg(L, drive(state=st, va=va), blk(bid, task, design)))
            segs.append(seg(rng.uniform(8, 12), rest, after=[sam_event(rng, bid, va)]))
            if i < len(quads) - 1:
                segs.append(seg(20, rest, blk(f"rest{i + 1}", "rest", (0, 0))))
        base = 90
    else:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {', '.join(PROTOCOLS)}")
    head = [("protocol_start", {"protocol": protocol, "order": order})]
    return pre, segs, head, base


# ---------------------------------------------------------------- signal
class Person:
    """One simulated participant: individual alpha peak, amplitude, channel gains and state expression."""

    def __init__(self, rng, effect=1.0):
        self.alpha_hz = rng.uniform(9.3, 10.7) - SPEC["bands"]["hz"][2]       # offset from the 10 Hz default
        self.scale = rng.uniform(0.8, 1.25)
        self.chan_gain = rng.normal(0, 0.1, (len(CHANNELS), len(SPEC["bands"]["hz"])))
        self.expr = {s: effect * rng.uniform(0.7, 1.3) for s in STATES + ["va"]}

    def target(self, d):
        if d["state"]:
            sig, e = state_sig(d["state"]), self.expr[d["state"]]
        else:
            sig, e = va_sig(*d["va"]), self.expr["va"]
        return NEUTRAL + e * (sig - NEUTRAL)


def synthesize(segs, person, rng, fs=FS):
    """(C, N) µV for the segments, plus each segment's first sample index."""
    S = SPEC
    lengths = [int(round(s["sec"] * fs)) for s in segs]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    N = int(sum(lengths))
    t = np.arange(1, N + 1) / fs
    sig = lagged(np.array([person.target(s["drive"]) for s in segs]), lengths, S["transition_tau"], fs, NEUTRAL)
    ec = lagged(np.array([[1.0 if s["drive"]["ec"] else 0.0] for s in segs]), lengths, 0.5, fs, [0.0])[:, 0]
    task_blinks = np.repeat([s["drive"]["blinks"] for s in segs], lengths)
    clench_on = np.zeros(N, bool)
    for s, a, n in zip(segs, starts, lengths):
        if s["drive"]["clench"]:
            clench_on[a:a + n] = (np.arange(1, n + 1) / fs) % 4 < 2
    g = {k: sig[:, KI[k]] for k in SIG_KEYS}
    C, nb = len(CHANNELS), len(S["bands"]["hz"])
    R = [regions(c) for c in CHANNELS]

    # eyes: blink times (spontaneous outside the blink task and with the eyes open; every ~1 s in the task)
    B = S["blink"]
    impulses = np.zeros(N)
    tb = rng.uniform(4, 10)
    while tb < t[-1]:
        i = min(N - 1, int(tb * fs))
        if not task_blinks[i] and ec[i] < 0.5:
            impulses[i] += g["blinkSize"][i]
        tb += rng.uniform(*B["interval_s"]) / max(0.05, g["blinkRate"][i])
    for s, a, n in zip(segs, starts, lengths):
        if s["drive"]["blinks"]:
            tb = a / fs + 0.5
            while tb < (a + n) / fs:
                impulses[min(N - 1, int(tb * fs))] += 1.0
                tb += rng.uniform(*B["task_interval_s"])
    d = np.arange(-int(0.4 * fs), int(0.4 * fs) + 1) / fs
    kern = B["uv"] * np.exp(-(d ** 2) / (2 * B["width_s"] ** 2))
    shift = int(round(0.15 * fs))
    blink = np.convolve(impulses, kern, mode="full")[len(d) // 2: len(d) // 2 + N]
    blink = np.concatenate([np.zeros(shift), blink[:N - shift]])
    roll = (S["roll"]["uv"] * np.maximum(0, g["roll"]) * (0.5 + 0.5 * np.sin(2 * np.pi * S["roll"]["envelope_hz"] * t))
            * np.sin(2 * np.pi * S["roll"]["hz"] * t + rng.uniform(0, 2 * np.pi)))

    # jaw: clench task and tension bursts (anger)
    E = S["emg"]
    emg_uv = np.where(clench_on, E["clench_uv"], 0.0)
    tension = g["emg"]
    tb = 0.0
    while tb < t[-1]:
        i = min(N - 1, int(tb * fs))
        if tension[i] > 0.05:
            dur = rng.uniform(*E["burst_s"])
            j = min(N, int((tb + dur) * fs))
            emg_uv[i:j] = np.maximum(emg_uv[i:j], E["tension_uv"] * np.clip(tension[i:j], 0, 1))
            tb += dur + rng.uniform(*E["gap_s"]) / tension[i]
        else:
            tb += 0.25
    emg = highpass_noise(rng, (C, N), E["hp_hz"], fs)

    bg = S["background"]
    x = (ar1_noise(rng, (C, N), bg["slow_tau"], bg["slow_sd"], fs) + ar1_noise(rng, (C, N), bg["fast_tau"], bg["fast_sd"], fs)
         + bg["white_sd"] * rng.normal(0, 1, (C, N)))
    fj = ar1_noise(rng, (C, N), S["jitter"]["tau"], S["jitter"]["sd"], fs)
    EC = S["eyes_closed"]
    for c, r in enumerate(R):
        env = np.clip(1 + ar1_noise(rng, (nb, N), S["envelope"]["tau"], S["envelope"]["sd"], fs), S["envelope"]["lo"], S["envelope"]["hi"])
        fr, po, sf = r["front"], r["post"], r["side"] * r["front"]
        g_beta = g["beta"] + g["betaF"] * fr + g["betaR"] * sf
        gains = [g["delta"], g["theta"] + g["thetaF"] * fr,
                 g["alpha"] + g["alphaP"] * po + g["alphaF"] * fr + g["faa"] * sf + S["posterior_alpha_boost"] * po
                 + ec * (EC["alpha"] + EC["alphaP"] * po),
                 g_beta, g_beta + g["hibeta"], g["gamma"]]
        for b in range(nb):
            hz = S["bands"]["hz"][b] + (g["alphaHz"] + person.alpha_hz if b == 2 else 0.0)
            ph = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(hz * (1 + fj[c])) / fs
            x[c] += S["bands"]["uv"][b] * np.exp(gains[b] + person.chan_gain[c, b]) * env[b] * np.sin(ph)
        x[c] *= person.scale
        x[c] += 0.3 * np.sin(2 * np.pi * 60 * t) + blink * r["blinkW"] + roll * r["rollW"]
        if r["temporal"]:
            x[c] += emg_uv * emg[c]
    return x, starts, lengths


# ---------------------------------------------------------------- writing
def write_session(root, code, protocol, person, rng, effect, block_sec, seed):
    pre, segs, head, base = build(protocol, rng, block_sec)
    allsegs = pre + segs
    x, starts, lengths = synthesize(allsegs, person, rng)
    x = x + DC_UV + rng.uniform(-150, 150, (len(CHANNELS), 1))
    sid = f"synthetic-{protocol}-{code}"
    d = os.path.join(root, sid)
    os.makedirs(d, exist_ok=True)
    events = [(0.0, "synthetic", {"generator": "scripts/synth_insight.py", "seed": seed, "effect": effect})]
    events.append((starts[1] / FS, *head[0]))           # protocol_start once the recording is running
    events.append((starts[len(pre)] / FS, "baseline_start", {"seconds": base}))
    for s, a, n in zip(segs, starts[len(pre):], lengths[len(pre):]):
        if s["block"]:
            events.append((a / FS, "block_start", s["block"]))
            events.append(((a + n) / FS, "block_end", {"block": s["block"]["block"]}))
        for typ, val in s["after"]:
            events.append(((a + n) / FS, typ, val))
    N = x.shape[1]
    events.append((N / FS, "protocol_end", {}))
    with open(os.path.join(d, "events.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "type", "value"])
        for tt, typ, val in events:
            w.writerow([f"{tt:.6f}", typ, json.dumps(val)])
    with open(os.path.join(d, "eeg.csv"), "w", newline="") as fh:
        fh.write(",".join(["time_s", *CHANNELS, "valence_label", "arousal_label"]) + "\n")
        tcol = np.arange(N) / FS
        fh.writelines(f"{tt:.6f}," + ",".join(f"{v:.4f}" for v in row) + ",,\n" for tt, row in zip(tcol, x.T))
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    meta = {"id": sid, "participant": code, "device": "Synthetic Emotiv Insight (scripts/synth_insight.py)", "fs": FS,
            "channels": CHANNELS, "protocol": protocol,
            "notes": "SYNTHETIC: simulated signal, no person recorded. Not research data.",
            "started": now, "app_version": "2.0.0", "origin": "synthetic", "synthetic": True,
            "generator": {"script": "scripts/synth_insight.py", "seed": seed, "effect": effect, "block_sec": block_sec,
                          "params": "emotif/data/synthetic_states.json"},
            "ended": now, "n_samples": int(N), "duration_s": round(N / FS, 3)}
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participants", type=int, default=8)
    p.add_argument("--protocol", default="face,states", help=f"comma list of {', '.join(PROTOCOLS)}")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=os.path.join(ROOT, "data", "synthetic"))
    p.add_argument("--effect", type=float, default=1.0, help="scale of every state difference (0 = none, a null check)")
    p.add_argument("--block-sec", type=float, default=None, help="shorten the emotion/state task blocks (default: the protocol's own)")
    a = p.parse_args(argv)
    protocols = [s.strip() for s in a.protocol.split(",") if s.strip()]
    bad = [s for s in protocols if s not in PROTOCOLS]
    if bad or not protocols:
        p.error(f"unknown protocol {', '.join(bad) or '(none)'}; choose from {', '.join(PROTOCOLS)}")
    if a.participants < 1:
        p.error("--participants must be at least 1")
    if a.block_sec is not None and a.block_sec < 10:
        p.error("--block-sec must be at least 10 (the analysis drops each block's first 5 s)")
    out_abs = os.path.abspath(a.out)
    for guard in ("results", os.path.join("data", "sessions")):
        g = os.path.join(ROOT, guard)
        if out_abs == g or out_abs.startswith(g + os.sep):
            p.error(f"refusing to write synthetic sessions under {guard}/ (real data and results live there)")
    os.makedirs(a.out, exist_ok=True)
    made = []
    for i in range(a.participants):
        code = f"SYN{i + 1:02d}"
        person = Person(np.random.default_rng([a.seed, i]), a.effect)        # the same simulated person in every session
        for k, prot in enumerate(protocols):
            rng = np.random.default_rng([a.seed, i, k + 1])
            made.append(write_session(a.out, code, prot, person, rng, a.effect, a.block_sec, a.seed))
            print(f"wrote {made[-1]}")
    return made


if __name__ == "__main__":
    main()
