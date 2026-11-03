#!/usr/bin/env python3
"""Train the artifact detector (clean / ocular / muscle) on EEGdenoiseNet and export it for the app.

  python research/train_artifact.py --data data/external/eegdenoisenet --out models/artifact_detector.json

Semi-synthetic windows follow the EEGdenoiseNet protocol: y = x + lambda * n with
SNR = 10 log10(RMS(x) / RMS(lambda n)) drawn from [-7, 2] dB. EEG, EOG and EMG epochs are split
70 / 15 / 15 into train / validation / test *before* mixing, so no source epoch appears in two splits.
Each simulated session gets its own headset (noise floor 3-30 dB below EEG, 30-110 Hz analogue low-pass,
50/60 Hz hum, 256 or 128 Hz) and a 16-window baseline; features are taken relative to that baseline, which
is what makes the detector transfer from lab EEG to consumer headsets.
Half of all windows (clean and contaminated) also carry a genuine brain-state change relative to the baseline:
one rhythm (theta, alpha or beta) scaled 0.4-3x and overall amplitude 0.7-1.5x. These stay labelled by their
artifact class, so the detector learns that eyes-closed alpha or drowsy theta is not an artifact.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotif import artifact as A, features as F, models  # noqa: E402

FS = 256
SNR_RANGE = (-7.0, 2.0)


def load_edn(root):
    out = {}
    for k in ("EEG", "EOG", "EMG"):
        p = os.path.join(root, f"{k}_all_epochs.npy")
        if not os.path.isfile(p):
            raise SystemExit(f"{p} not found; run scripts/get_eegdenoisenet.sh")
        out[k] = np.load(p).astype(np.float64)
    return out


def split_idx(n, rng):
    idx = rng.permutation(n)
    a, b = int(0.7 * n), int(0.85 * n)
    return {"train": idx[:a], "val": idx[a:b], "test": idx[b:]}


def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=-1))


def device(Y, fs, sess, sigma_ref, rng):
    """Simulated headset: optional 2:1 resampling, amplifier noise floor, analogue low-pass, mains hum,
    then the Emotif filter. Y: (n, 512) at 256 Hz."""
    from scipy.signal import butter, lfilter
    W = Y if fs == FS else resample_poly(Y, 1, 2, axis=1)
    W = W + rng.normal(0, sigma_ref * 10 ** (-sess["noise_db"] / 20), W.shape)
    if sess["fc"] < 0.45 * fs:
        b, a = butter(2, sess["fc"] / (fs / 2))
        W = lfilter(b, a, W, axis=1)
    t = np.arange(W.shape[1]) / fs
    W = W + sess["mains"] * sigma_ref * np.sin(2 * np.pi * sess["mains_hz"] * t + rng.uniform(0, 2 * np.pi, (len(W), 1)))
    return F.filter_signal(W, fs)


STATE_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0))


def state_change(x, rng, fs=FS):
    """Genuine brain-state change: one rhythm scaled by 0.4-3x (log-uniform, raised-cosine band edges)
    and the whole epoch by 0.7-1.5x. x: (n, N)."""
    n, N = x.shape
    f = np.fft.rfftfreq(N, 1 / fs)
    X = np.fft.rfft(x, axis=1)
    for i in range(n):
        lo, hi = STATE_BANDS[rng.integers(0, len(STATE_BANDS))]
        g = np.exp(rng.uniform(np.log(0.4), np.log(3.0)))
        edge = np.clip(np.minimum(f - (lo - 1), (hi + 1) - f) / 2.0, 0, 1)        # 0 outside, 1 inside
        w = 0.5 - 0.5 * np.cos(np.pi * edge)
        X[i] *= 1 + (g - 1) * w
        X[i] *= np.exp(rng.uniform(np.log(0.7), np.log(1.5)))
    return np.fft.irfft(X, n=N, axis=1)


def make_split(edn, splits, name, n_sessions, per_class, rng, state_frac=0.5):
    """Each simulated session has its own device and a 16-window resting baseline (2 of which contain a
    blink, as real baselines do). A fraction state_frac of windows carries a genuine state change.
    Returns relative features X, labels y, SNR, fs, state-change flag."""
    eeg = edn["EEG"][splits["EEG"][name]]
    eog = edn["EOG"][splits["EOG"][name]]
    emg = edn["EMG"][splits["EMG"][name]]
    sigma_ref = float(np.median(rms(eeg)))
    Xs, ys, snrs, fss, sts = [], [], [], [], []
    for _ in range(n_sessions):
        fs = FS if rng.random() < 0.5 else 128
        sess = {"noise_db": rng.uniform(3, 30), "fc": rng.uniform(30, 110), "mains": rng.uniform(0, 0.5),
                "mains_hz": 50.0 if rng.random() < 0.5 else 60.0}
        base = eeg[rng.integers(0, len(eeg), 16)].copy()
        for i in range(2):
            nz = eog[rng.integers(0, len(eog))]
            base[i] += rms(base[i]) / (rms(nz) * 10 ** (rng.uniform(-3, 2) / 10)) * nz
        bfeat = A.baseline(A.channel_features(device(base, fs, sess, sigma_ref, rng), fs)[:, None, :])[0]
        Y, lab, snr, st = [], [], [], []
        for cls in range(3):
            x = eeg[rng.integers(0, len(eeg), per_class)]
            changed = rng.random(per_class) < state_frac
            if changed.any():
                x = x.copy()
                x[changed] = state_change(x[changed], rng)
            st.append(changed)
            if cls == 0:
                Y.append(x)
                snr.append(np.full(per_class, np.inf))
            else:
                src = eog if cls == 1 else emg
                nz = src[rng.integers(0, len(src), per_class)]
                sn = rng.uniform(*SNR_RANGE, per_class)
                Y.append(x + (rms(x) / (rms(nz) * 10 ** (sn / 10)))[:, None] * nz)
                snr.append(sn)
            lab.append(np.full(per_class, cls))
        feats = A.channel_features(device(np.vstack(Y), fs, sess, sigma_ref, rng), fs)
        Xs.append(feats - bfeat)
        ys.append(np.concatenate(lab))
        snrs.append(np.concatenate(snr))
        fss.append(np.full(3 * per_class, fs))
        sts.append(np.concatenate(st))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(snrs), np.concatenate(fss), np.concatenate(sts)


def fold_standardization(layers, mu, sd):
    """x_std = (x - mu) / sd folded into the first layer, so the app feeds raw features."""
    W = np.array(layers[0]["W"])
    b = np.array(layers[0]["b"])
    W2 = W / sd[:, None]
    b2 = b - (mu / sd) @ W
    layers[0] = {"W": W2.tolist(), "b": b2.tolist(), "act": layers[0]["act"]}
    return layers


def plots(out_dir, cm, snr_bins, det_rates, fs_acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=170)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.6 else "#1f2a36")
    ax.set_xticks(range(3), ["clean", "eye", "muscle"])
    ax.set_yticks(range(3), ["clean", "eye", "muscle"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Artifact detector, held-out test set", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_confusion.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.0), dpi=170)
    centers = (snr_bins[:-1] + snr_bins[1:]) / 2
    for k, (lab, col) in enumerate((("Eye (EOG)", "#2f6f9f"), ("Muscle (EMG)", "#c2410c"))):
        ax.plot(centers, det_rates[k], marker="o", color=col, label=lab)
    ax.set_xlabel("Signal-to-noise ratio (dB); lower = stronger artifact")
    ax.set_ylabel("Detected as contaminated")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Detection rate by artifact strength (256 Hz {fs_acc[256]:.3f}, 128 Hz {fs_acc[128]:.3f} acc.)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_detection_vs_snr.png"))
    plt.close(fig)


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/external/eegdenoisenet")
    p.add_argument("--out", default="models/artifact_detector.json")
    p.add_argument("--results", default="results/artifact_detector")
    p.add_argument("--sessions", type=int, default=600, help="simulated training sessions (30 windows each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state-frac", type=float, default=0.5, help="fraction of windows with a genuine state change")
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    edn = load_edn(a.data)
    splits = {k: split_idx(len(v), rng) for k, v in edn.items()}
    tr = make_split(edn, splits, "train", a.sessions, 10, rng, a.state_frac)
    va = make_split(edn, splits, "val", max(60, a.sessions // 4), 10, rng, 0.5)
    te = make_split(edn, splits, "test", max(120, a.sessions // 2), 10, rng, 0.5)
    mu, sd = tr[0].mean(axis=0), tr[0].std(axis=0) + 1e-9
    Z = lambda X: (X - mu) / sd  # noqa: E731

    cands = {"logreg": LogisticRegression(C=1.0, max_iter=4000),
             "mlp32": MLPClassifier((32,), alpha=1e-3, max_iter=600, early_stopping=True, random_state=a.seed),
             "mlp64x32": MLPClassifier((64, 32), alpha=1e-3, max_iter=600, early_stopping=True, random_state=a.seed)}
    val_acc = {}
    for name, clf in cands.items():
        clf.fit(Z(tr[0]), tr[1])
        val_acc[name] = float(np.mean(clf.predict(Z(va[0])) == va[1]))
        print(f"  {name}: validation accuracy {val_acc[name]:.4f}")
    best = max(val_acc, key=val_acc.get)
    clf = cands[best]

    # contamination threshold: largest sensitivity with <= 2 % false alarms on clean validation windows
    pv = clf.predict_proba(Z(va[0]))
    contam_v = 1 - pv[:, 0]
    thr = float(np.quantile(contam_v[va[1] == 0], 0.98))
    thr = min(max(thr, 0.5), 0.95)

    pt = clf.predict_proba(Z(te[0]))
    pred = pt.argmax(axis=1)
    acc = float(np.mean(pred == te[1]))
    cm = confusion_matrix(te[1], pred, labels=[0, 1, 2])
    contam = (1 - pt[:, 0]) >= thr
    far = float(contam[te[1] == 0].mean())
    far_state = float(contam[(te[1] == 0) & te[4]].mean())
    far_rest = float(contam[(te[1] == 0) & ~te[4]].mean())
    sens = float(contam[te[1] != 0].mean())
    bins = np.linspace(*SNR_RANGE, 7)
    det = []
    for cls in (1, 2):
        m = te[1] == cls
        det.append([float(contam[m & (te[2] >= lo) & (te[2] < hi)].mean()) for lo, hi in zip(bins[:-1], bins[1:])])
    fs_acc = {int(f): float(np.mean(pred[te[3] == f] == te[1][te[3] == f])) for f in (256, 128)}
    print(f"best={best} test accuracy {acc:.4f}  (256 Hz {fs_acc[256]:.4f}, 128 Hz {fs_acc[128]:.4f})")
    print(f"contaminated if p(artifact) >= {thr:.3f}: sensitivity {sens:.3f}, false alarms on clean {far:.3f} "
          f"(resting {far_rest:.3f}, genuine state change {far_state:.3f})")
    print("confusion (rows true clean/eye/muscle):\n", cm)

    layers = fold_standardization(models.export_layers(clf), mu, sd)
    assert np.allclose(models.forward(layers, te[0][:50]), pt[:50], atol=1e-8)
    model = {
        "format": "emotif-artifact-v2", "classes": A.CLASSES, "feature_names": A.FEATURE_NAMES,
        "layers": layers, "mask_threshold": thr, "model": best,
        "trained_on": "EEGdenoiseNet (Zhang et al. 2021), semi-synthetic, SNR -7..2 dB, simulated headsets at 256 and 128 Hz, "
                      f"{a.state_frac:.0%} of windows with a genuine brain-state change",
        "input": "channel_features minus the channel's baseline median",
        "metrics": {"test_accuracy": acc, "test_accuracy_by_fs": fs_acc, "sensitivity": sens,
                    "false_alarm_rate": far, "false_alarm_rate_resting": far_rest,
                    "false_alarm_rate_state_change": far_state, "confusion": cm.tolist(), "validation_accuracy": val_acc,
                    "n_test": int(len(te[1]))},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(model, fh)
    os.makedirs(a.results, exist_ok=True)
    with open(os.path.join(a.results, "metrics.json"), "w") as fh:
        json.dump({**model["metrics"], "snr_bins": bins.tolist(), "detection_by_snr": {"ocular": det[0], "muscle": det[1]},
                   "mask_threshold": thr, "model": best}, fh, indent=1)
    plots(a.results, cm, bins, det, fs_acc)
    print(f"wrote {a.out} and {a.results}/")
    return model


if __name__ == "__main__":
    main()
