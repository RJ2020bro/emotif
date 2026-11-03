#!/usr/bin/env python3
"""Ground-truth denoising benchmark for multichannel consumer and lab layouts.

  python research/denoise_benchmark.py --data data/external/eegdenoisenet --out results/denoise

Clean multichannel EEG is built from EEGdenoiseNet *test-split* clean epochs (never used to train the
artifact detector), mixed across electrodes with a volume-conduction kernel so neighbouring channels are
correlated. Real EOG and EMG epochs are added with physiologically placed spatial gains (eyes in front,
temporalis muscles at the sides), plus electrode pops, at SNR -7..2 dB on the worst channel. Each 60 s trial
has a 30 s calibration part with two mild blinks (like a real resting baseline) and a 30 s test part with
seven artifacts, during which alpha power genuinely doubles (a brain-state change that cleaning must keep).

Methods: filter only; amplitude rejection (> 200 uV p-p, windows discarded); ASR at k = 5, 10, 20, 30.
Metrics on 2 s windows: relative RMS error and correlation vs. ground truth, |alpha| and |beta| band-power
error (dB), separately for contaminated and clean windows (the second measures distortion of good data).
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotif import asr, features as F, montage as M  # noqa: E402
from research.train_artifact import load_edn, split_idx  # noqa: E402

FS = 256
LAYOUTS = {
    "muse (4 ch)": M.DEVICES["muse"],
    "openbci (8 ch)": M.DEVICES["openbci_cyton"],
    "emotiv (14 ch)": M.DEVICES["emotiv_epoc"],
    "lab cap (32 ch)": ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1",
                        "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2",
                        "P4", "P8", "PO4", "O2"],
}
KS = (3, 5, 10, 20)
METHODS = ["filter only", "reject >200 uV"] + [f"ASR k={k}" for k in KS]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


EYES = unit([0.0, 0.92, -0.40])
TEMPORALIS = [unit([-0.92, 0.10, -0.38]), unit([0.92, 0.10, -0.38])]


def ang(a, b):
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


def continuous(epochs, n_epochs, rng):
    """Concatenate random 2 s epochs with 0.25 s raised-cosine cross-fades into one smooth signal."""
    ov = int(0.25 * FS)
    w = (1 - np.cos(np.linspace(0, np.pi, ov))) / 2
    out = epochs[rng.integers(len(epochs))].copy()
    for _ in range(n_epochs - 1):
        e = epochs[rng.integers(len(epochs))]
        out[-ov:] = out[-ov:] * (1 - w) + e[:ov] * w
        out = np.concatenate([out, e[ov:]])
    return out


def make_trial(chans, edn, rng, seconds=60):
    pos = np.array([M.position(c) for c in chans])
    C, N = len(chans), seconds * FS
    n_ep = int(np.ceil(N / (2 * FS - 0.25 * FS))) + 1
    S = np.array([continuous(edn["EEG"], n_ep, rng)[:N] for _ in range(C)])
    S = S / S.std(axis=1, keepdims=True)
    # genuine brain-state change after the baseline: alpha amplitude ramps up to 2x over 5 s
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [8 / (FS / 2), 13 / (FS / 2)], btype="band")
    alpha = filtfilt(b, a, S, axis=1)
    ramp = np.clip((np.arange(N) / FS - 30.0) / 5.0, 0, 1)
    S = S + ramp * alpha
    K = np.exp(-np.array([[ang(p, q) for q in pos] for p in pos]) ** 2 / (2 * 30.0 ** 2))
    X = (K / K.sum(axis=1, keepdims=True)) @ S
    X = 10.0 * X / X.std()                                   # ~10 uV RMS like scalp EEG
    art = np.zeros_like(X)
    events = []
    g_eye = np.exp(-(np.array([ang(p, EYES) for p in pos]) / 35.0) ** 2)
    g_tmp = [np.exp(-(np.array([ang(p, t) for p in pos]) / 30.0) ** 2) for t in TEMPORALIS]

    def add(kind, t0, snr):
        seg = slice(t0, t0 + 2 * FS)
        x_ref = X[:, seg]
        if kind == "ocular":
            n = edn["EOG"][rng.integers(len(edn["EOG"]))]
            gains = g_eye
            pattern = np.outer(gains, n)
        elif kind == "muscle":
            g = g_tmp[rng.integers(2)] if rng.random() < 0.7 else np.maximum(*g_tmp)
            common = edn["EMG"][rng.integers(len(edn["EMG"]))]
            local = np.array([edn["EMG"][rng.integers(len(edn["EMG"]))] for _ in range(C)])
            pattern = g[:, None] * (0.6 * common + 0.4 * local * common.std() / local.std(axis=1, keepdims=True))
            gains = g
        else:                                                 # electrode pop / cable motion
            c = rng.integers(C)
            t = np.arange(2 * FS) / FS
            step = np.where(t > 0.3, np.exp(-(t - 0.3) / 0.25), 0.0)
            pattern = np.zeros((C, 2 * FS))
            pattern[c] = step
            gains = np.zeros(C)
            gains[c] = 1.0
        worst = int(np.argmax(gains))
        lam = np.sqrt(np.mean(x_ref[worst] ** 2)) / (np.sqrt(np.mean(pattern[worst] ** 2)) * 10 ** (snr / 10))
        art[:, seg] += lam * pattern
        events.append((t0, t0 + 2 * FS, kind, gains > 0.1))

    for t0 in (int(8.0 * FS), int(21.0 * FS)):                 # mild blinks in the calibration part
        add("ocular", t0, rng.uniform(0, 4))
    starts = np.sort(rng.choice(np.arange(31, 57, 0.5), 7, replace=False))
    for s in starts:
        add(rng.choice(["ocular", "muscle", "pop"], p=[0.4, 0.4, 0.2]), int(s * FS), rng.uniform(-7, 2))
    return X, X + art, [e for e in events if e[0] >= 30 * FS]


def window_metrics(est, ref, fs):
    err = est - ref
    rr = np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(ref ** 2))
    cc = np.mean([np.corrcoef(est[c], ref[c])[0, 1] for c in range(ref.shape[0])])
    pe, pr = F.band_powers(est, fs), F.band_powers(ref, fs)
    db = np.abs(10 * np.log10((pe + 1e-12) / (pr + 1e-12)))
    return rr, cc, float(db[:, 2].mean()), float(db[:, 3].mean())


def run(layout, chans, edn, n_trials, rng):
    res = {m: {"contam": [], "clean": [], "kept": []} for m in METHODS}
    for _ in range(n_trials):
        X, Y = None, None
        X, Y, events = make_trial(chans, edn, rng)
        Xf, Yf = F.filter_signal(X, FS), F.filter_signal(Y, FS)
        outs = {"filter only": Yf}
        for k in KS:
            st = asr.calibrate(Yf[:, : 30 * FS], FS, k)
            outs[f"ASR k={k}"] = asr.clean(Yf, st, FS)[0]
        W = 2 * FS
        for s0 in range(30 * FS, 60 * FS - W + 1, FS):
            contam = any(s0 < e1 and s0 + W > e0 for e0, e1, _, _ in events)
            key = "contam" if contam else "clean"
            ref = Xf[:, s0:s0 + W]
            for m in METHODS:
                if m == "reject >200 uV":
                    win = Yf[:, s0:s0 + W]
                    d = F.detrend(win)
                    keep = bool((d.max(axis=1) - d.min(axis=1)).max() <= 200)
                    res[m]["kept"].append(keep)
                    if keep:
                        res[m][key].append(window_metrics(win, ref, FS))
                else:
                    res[m][key].append(window_metrics(outs[m][:, s0:s0 + W], ref, FS))
                    res[m]["kept"].append(True)
    summary = {}
    for m, r in res.items():
        def agg(rows):
            if not rows:
                return None
            a = np.array(rows)
            return {"rrmse": float(np.mean(a[:, 0])), "rrmse_median": float(np.median(a[:, 0])),
                    "cc": float(np.mean(a[:, 1])), "alpha_db": float(np.mean(a[:, 2])), "beta_db": float(np.mean(a[:, 3])),
                    "n": len(rows)}
        summary[m] = {"contaminated": agg(r["contam"]), "clean": agg(r["clean"]), "kept": float(np.mean(r["kept"]))}
    return summary


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lays = list(results)
    meths = ["filter only", "ASR k=5", "ASR k=10", "ASR k=20"]
    cols = ["#9aa5b1", "#b7d3e8", "#6fa3cb", "#173f5f"]
    names = {"ASR k=20": "ASR k=20 (chosen)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=170)
    w = 0.2
    for k, (m, c) in enumerate(zip(meths, cols)):
        xs = np.arange(len(lays)) + (k - 1.5) * w
        axes[0].bar(xs, [results[l][m]["contaminated"]["rrmse"] for l in lays], w, color=c, label=names.get(m, m))
        axes[1].bar(xs, [results[l][m]["clean"]["rrmse"] for l in lays], w, color=c, label=names.get(m, m))
    axes[0].set_title("Windows with artifacts: error vs. ground truth", fontsize=9)
    axes[1].set_title("Clean windows (incl. a real alpha change): distortion", fontsize=9)
    for ax in axes:
        ax.set_xticks(range(len(lays)), lays, fontsize=8)
        ax.set_ylabel("Relative RMS error (mean)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/external/eegdenoisenet")
    p.add_argument("--out", default="results/denoise")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    full = load_edn(a.data)
    split_rng = np.random.default_rng(0)                    # same split as research/train_artifact.py
    splits = {k: split_idx(len(v), split_rng) for k, v in full.items()}
    edn = {k: full[k][splits[k]["test"]] for k in full}
    results = {}
    for lay, chans in LAYOUTS.items():
        results[lay] = run(lay, chans, edn, a.trials, rng)
        r = results[lay]
        print(f"{lay}: " + "  ".join(f"{m}: {r[m]['contaminated']['rrmse']:.2f}/{r[m]['clean']['rrmse']:.3f}"
                                    if r[m]["contaminated"] else f"{m}: n/a" for m in METHODS))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    lines = ["| Layout | Method | Artifact windows: RRMSE | corr. | alpha error (dB) | beta error (dB) | Clean windows: RRMSE | Windows kept |",
             "|---|---|---|---|---|---|---|---|"]
    for lay, r in results.items():
        for m in METHODS:
            c, k = r[m]["contaminated"], r[m]["clean"]
            if c is None:
                lines.append(f"| {lay} | {m} | (all rejected) | | | | {k['rrmse']:.3f} | {r[m]['kept']:.0%} |")
                continue
            lines.append(f"| {lay} | {m} | {c['rrmse']:.3f} | {c['cc']:.3f} | {c['alpha_db']:.2f} | {c['beta_db']:.2f} | "
                         f"{k['rrmse']:.3f} | {r[m]['kept']:.0%} |")
    lines.append("")
    lines.append(f"Means over 2 s windows from {a.trials} simulated 60 s recordings per layout (seed {a.seed}). RRMSE = RMS(error) / RMS(clean signal).")
    with open(os.path.join(a.out, "table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    plot(results, os.path.join(a.out, "fig_denoise.png"))
    print("\n".join(lines))
    return results


if __name__ == "__main__":
    main()
