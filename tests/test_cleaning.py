"""Behavioural tests for the cleaning pipeline: ASR repairs bursts, leaves clean data and genuine brain-state
changes alone, never amplifies (including rank-deficient data), and the artifact detector separates clean
from contaminated channels. Uses EEGdenoiseNet when present (data/external/eegdenoisenet)."""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from emotif import artifact as A, asr, features as F, models, datasets  # noqa: E402

FS = 256


def eeg_like(C, seconds, rng, alpha_gain=None):
    """Correlated 1/f-ish background with alpha; alpha_gain(t) scales the alpha rhythm over time."""
    N = int(seconds * FS)
    t = np.arange(N) / FS
    mix = rng.normal(0, 1, (C, C)) * 0.4 + np.eye(C)
    bg = F.filter_signal(np.cumsum(rng.normal(0, 1, (C, N)), axis=1), FS) * 0.6 + rng.normal(0, 3, (C, N))
    g = np.ones(N) if alpha_gain is None else alpha_gain(t)
    alpha = 8 * g * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6, (C, 1)))
    return F.filter_signal(mix @ (bg + alpha), FS)


def rel(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def test_asr_repairs_and_preserves(rng):
    C = 8
    X = eeg_like(C, 60, rng, alpha_gain=lambda t: 1 + np.clip((t - 30) / 5, 0, 1))   # alpha doubles after 30 s
    Y = X.copy()
    gains = np.linspace(1, 0.1, C)[:, None]
    t = np.arange(FS) / FS
    blink = 250 * np.exp(-((t - 0.5) ** 2) / (2 * 0.08 ** 2))
    for t0 in (36, 42, 49, 55):
        Y[:, t0 * FS:(t0 + 1) * FS] += gains * blink
    st = asr.calibrate(Y[:, :30 * FS], FS)
    Z, removed = asr.clean(Y, st, FS)
    seg = slice(34 * FS, 57 * FS)
    before, after = rel(Y[:, seg], X[:, seg]), rel(Z[:, seg], X[:, seg])
    assert after < 0.5 * before, (before, after)
    # genuine state change (alpha x2, no artifacts) is untouched: windows between blinks
    calm = np.r_[31 * FS:35 * FS]
    assert rel(Z[:, calm], X[:, calm]) < 0.02, rel(Z[:, calm], X[:, calm])
    # no amplification anywhere
    assert np.sum(Z ** 2) <= np.sum(Y ** 2)
    print(f"ASR: blink-period error {before:.2f} -> {after:.2f}; alpha-doubling segment distortion "
          f"{rel(Z[:, calm], X[:, calm]):.4f}")


def test_asr_rank_deficient(rng):
    C = 8
    X = eeg_like(C, 40, rng)
    X = X - X.mean(axis=0, keepdims=True)                      # average reference: rank C-1
    Y = X.copy()
    Y[:, 25 * FS:26 * FS] += rng.normal(0, 1, (C, 1)) * 200 * np.hanning(FS)
    Y = Y - Y.mean(axis=0, keepdims=True)
    st = asr.calibrate(Y[:, :20 * FS], FS)
    Z, _ = asr.clean(Y, st, FS)
    for s0 in range(0, Z.shape[1] - FS // 2, FS // 2):
        w = slice(s0, s0 + FS // 2)
        assert np.sum(Z[:, w] ** 2) <= np.sum(Y[:, w] ** 2) * 1.0001, s0
    print("ASR: average-referenced (rank-deficient) data never amplified")


def test_detector_heldout():
    root = os.path.join(ROOT, "data", "external", "eegdenoisenet")
    if not os.path.isdir(root):
        print("detector held-out test skipped (no EEGdenoiseNet)")
        return
    sys.path.insert(0, os.path.join(ROOT, "research"))
    from research.train_artifact import load_edn, make_split, split_idx
    m = A.load()
    edn = load_edn(root)
    rng = np.random.default_rng(0)                       # same generator sequence as train_artifact.py --seed 0
    splits = {k: split_idx(len(v), rng) for k, v in edn.items()}
    X, y, snr, fs, state = make_split(edn, splits, "test", 60, 10, np.random.default_rng(123))
    p = A.predict_proba(m, X)
    acc = float(np.mean(p.argmax(1) == y))
    contam = (1 - p[:, 0]) >= m["mask_threshold"]
    far = float(contam[y == 0].mean())
    far_state = float(contam[(y == 0) & state].mean())
    assert acc > 0.88 and far < 0.05 and far_state < 0.05, (acc, far, far_state)
    print(f"detector on fresh held-out sessions: accuracy {acc:.3f}, false alarms on clean {far:.3f} "
          f"(with a genuine brain-state change {far_state:.3f})")


def test_pipeline_on_real_muse():
    root = os.path.join(ROOT, "data", "external", "bird_muse")
    if not os.path.isdir(root):
        print("real-data pipeline test skipped (run scripts/get_bird_muse.sh)")
        return
    tr = datasets.load_muse_mental(root)
    subs, rej, st = models.extract(tr, 1.0, cleaning=models.Cleaning(), return_stats=True)
    sm = st.summary()
    assert 0.02 < sm["asr_repaired_block_fraction"] < 0.5, sm
    assert sm["masked_by_detector"] < 0.7 and sum(len(s.groups) for s in subs) > 500, sm
    print(f"real Muse pipeline: ASR repaired {sm['asr_repaired_block_fraction']:.0%} of blocks, detector held out "
          f"{sm['masked_by_detector']:.0%} of channel-windows, {sum(len(s.groups) for s in subs)} windows kept")


def _narrowband_eeg(C, seconds, rng, alpha_gain=1.0, mix=None):
    """Background like eeg_like, with alpha as 8-12 Hz band-limited noise (waxing and waning, as real alpha).
    mix: the head's channel mixing; pass the same one for recordings of the same person."""
    from scipy.signal import butter, sosfiltfilt
    N = int(seconds * FS)
    mix = rng.normal(0, 1, (C, C)) * 0.4 + np.eye(C) if mix is None else mix
    bg = F.filter_signal(np.cumsum(rng.normal(0, 1, (C, N)), axis=1), FS) * 0.6 + rng.normal(0, 3, (C, N))
    alpha = sosfiltfilt(butter(4, [8, 12], "bandpass", fs=FS, output="sos"), rng.normal(0, 1, (C, N)), axis=1)
    return F.filter_signal(mix @ (bg + alpha_gain * 8 * alpha / alpha.std()), FS)


def _audit_trials(rng, driver):
    """3 subjects x 6 one-minute Muse-layout recordings, binary label. driver='brain': label +1 doubles alpha,
    no artifacts. driver='face': label +1 adds jaw-muscle bursts (25-100 Hz) on the temporal electrodes in 60 %
    of seconds and blinks on the frontal ones, brain unchanged."""
    from scipy.signal import butter, sosfilt
    from emotif.datasets import Trial
    chans = ["TP9", "AF7", "AF8", "TP10"]
    sos = butter(4, [25, 100], "bandpass", fs=FS, output="sos")
    tb = np.arange(FS // 2) / FS
    blink = 120 * np.exp(-((tb - 0.25) ** 2) / (2 * 0.05 ** 2))
    trials = []
    for s in range(3):
        head = rng.normal(0, 1, (4, 4)) * 0.4 + np.eye(4)
        for r, lab in enumerate([1, -1, 1, -1, 1, -1]):
            face = driver == "face" and lab == 1
            x = _narrowband_eeg(4, 60, rng, 2.0 if (driver == "brain" and lab == 1) else 1.0, head)
            if face:
                for sec in np.flatnonzero(rng.random(60) < 0.6):
                    x[[0, 3], sec * FS:(sec + 1) * FS] += sosfilt(sos, rng.normal(0, 25, (2, FS)), axis=1)
                for sec in np.flatnonzero(rng.random(60) < 0.3):
                    x[[1, 2], sec * FS:sec * FS + FS // 2] += blink
            trials.append(Trial(f"s{s}", f"s{s}r{r}", x, FS, chans, {"arousal": lab}, filtered=True,
                                dataset="synthetic", group=f"s{s}r{r}"))
    return trials


def test_audit_known_answer(rng):
    """The audit must point to the brain when the label is in the brain and to the face when it is in the face.
    Judged by the recording-level permutation test, since 18 recordings give chance-level accuracies of 0.6+."""
    from emotif import audit
    out = {}
    for driver in ("brain", "face"):
        d = audit.assemble(_audit_trials(rng, driver), "arousal")
        res = audit.run(d, n_perm=39, n_boot=200, conditions=["eeg_cleaned", "face_index", "eeg_partialled"],
                        log=lambda *_: None)
        out[driver] = {c: (r["bacc"], r["p_perm"]) for c, r in res["conditions"].items()}
    b, f = out["brain"], out["face"]
    assert b["eeg_cleaned"][0] > 0.9 and b["eeg_partialled"][0] > 0.75 and b["eeg_partialled"][1] <= 0.05, b
    assert b["face_index"][1] > 0.05, b
    assert f["face_index"][0] > 0.85 and f["face_index"][1] <= 0.05, f
    assert f["eeg_partialled"][0] < 0.65, f      # linear partialling removes most, not always all, leakage
    fmt = lambda v: f"{v[0]:.2f} (p={v[1]:.3f})"  # noqa: E731
    print(f"audit known-answer: brain-driven -> partialled EEG {fmt(b['eeg_partialled'])}, face {fmt(b['face_index'])}; "
          f"face-driven -> face {fmt(f['face_index'])}, partialled EEG {fmt(f['eeg_partialled'])}")


def main():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(7)
    test_asr_repairs_and_preserves(rng)
    test_asr_rank_deficient(rng)
    test_detector_heldout()
    test_pipeline_on_real_muse()
    test_audit_known_answer(rng)
    print("cleaning tests passed")


if __name__ == "__main__":
    main()
