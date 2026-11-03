"""Canonical signal pipeline. app/src/dsp.ts mirrors it; tests/test_parity.py compares the two.

Filter  : causal biquad cascade on the continuous signal, before windowing:
          high-pass 0.5 Hz (Butterworth, Q = 1/sqrt 2) + notch 50 Hz + notch 60 Hz (Q = 30, each only if
          f < 0.45*fs). Each stage starts in steady state for its first input sample, so DC offsets do not ring.
Window  : 2.0 s, all channels (µV).
Welch   : 1.0 s segments, 50 % overlap, periodic Hann, per-segment mean removal.
Bands   : delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz (lo <= f < hi).
DE      : 0.5 * ln(2*pi*e*P) per channel per band (Duan et al. 2013; used by SEED / Fernandes et al. 2024).
"""
import numpy as np

BANDS = (("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0))
WIN_SEC = 2.0
SEG_SEC = 1.0
EPS = 1e-12
_LOG2PIE = np.log(2.0 * np.pi * np.e)

HP_HZ = 0.5
NOTCH_HZ = (50.0, 60.0)
NOTCH_Q = 30.0


def filter_coeffs(fs):
    """RBJ biquads normalised so a0 = 1. Returns a list of (b, a)."""
    out = []

    def norm(b, a):
        return [v / a[0] for v in b], [1.0, a[1] / a[0], a[2] / a[0]]

    w0 = 2 * np.pi * HP_HZ / fs
    cw, al = np.cos(w0), np.sin(w0) / (2 * (1 / np.sqrt(2)))
    out.append(norm([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2], [1 + al, -2 * cw, 1 - al]))
    for f0 in NOTCH_HZ:
        if f0 < 0.45 * fs:
            w0 = 2 * np.pi * f0 / fs
            cw, al = np.cos(w0), np.sin(w0) / (2 * NOTCH_Q)
            out.append(norm([1.0, -2 * cw, 1.0], [1 + al, -2 * cw, 1 - al]))
    return out


def filter_signal(x, fs):
    """x: (C, T). Causal, not zero-phase; matches the live app filter sample for sample."""
    from scipy.signal import lfilter
    y = np.asarray(x, dtype=np.float64)
    for b, a in filter_coeffs(fs):
        x0 = y[:, :1]
        yss = x0 * (sum(b) / sum(a))
        z2 = b[2] * x0 - a[2] * yss
        z1 = b[1] * x0 - a[1] * yss + z2
        y, _ = lfilter(b, a, y, axis=1, zi=np.concatenate([z1, z2], axis=1))
    return y


def window_len(fs):
    return int(round(WIN_SEC * fs))


def band_powers(win, fs, bands=None):
    """win: (C, N). Returns (C, len(bands)) band power in µV²; bands = (lo, hi) pairs in Hz, default the 5 EEG
    bands."""
    bands = [(lo, hi) for _, lo, hi in BANDS] if bands is None else list(bands)
    win = np.asarray(win, dtype=np.float64)
    C, N = win.shape
    L = int(round(SEG_SEC * fs))
    step = L // 2
    starts = list(range(0, N - L + 1, step))
    if not starts:
        raise ValueError("window shorter than one Welch segment")
    n = np.arange(L)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / L)
    scale = 2.0 / (fs * np.sum(w * w))
    psd = np.zeros((C, L // 2 + 1))
    for s in starts:
        seg = win[:, s:s + L]
        seg = seg - seg.mean(axis=1, keepdims=True)
        X = np.fft.rfft(seg * w, axis=1)
        psd += (X.real ** 2 + X.imag ** 2) * scale
    psd /= len(starts)
    freqs = np.arange(L // 2 + 1) * fs / L
    df = fs / L
    out = np.zeros((C, len(bands)))
    for b, (lo, hi) in enumerate(bands):
        sel = (freqs >= lo) & (freqs < hi)
        out[:, b] = psd[:, sel].sum(axis=1) * df
    return out


def de_from_power(p):
    return 0.5 * (_LOG2PIE + np.log(p + EPS))


def de_matrix(win, fs):
    """(C, 5) differential entropy."""
    return de_from_power(band_powers(win, fs))


def detrend(win):
    """Remove the least-squares line from each channel."""
    win = np.asarray(win, dtype=np.float64)
    N = win.shape[1]
    t = np.arange(N) - (N - 1) / 2.0
    slope = (win @ t)[:, None] / np.sum(t * t)
    return win - win.mean(axis=1, keepdims=True) - slope * t[None, :]


REJECT_UV = 200.0   # window rejected if a usable channel exceeds this p-p (blink, movement)
BAD_UV = 500.0      # channel unusable in this window above this p-p (lost contact) or if flat
FLAT_UV = 0.1


def channel_quality(win, reject_uv=REJECT_UV, bad_uv=BAD_UV):
    """Per window, after linear detrend. Returns (good: bool (C,), reject: bool).
    Bad channels are dropped from this window (region pooling masks them); the window is rejected if a
    remaining channel exceeds reject_uv or nothing usable is left. reject_uv <= 0 disables both checks
    except the flat-line test."""
    d = detrend(win)
    ptp = d.max(axis=1) - d.min(axis=1)
    flat = d.std(axis=1) < FLAT_UV
    if reject_uv <= 0:
        good = ~flat
        return good, not good.any()
    good = ~flat & (ptp <= bad_uv)
    reject = (not good.any()) or bool(np.any(ptp[good] > reject_uv))
    return good, reject


def sliding_windows(x, fs, hop_sec):
    N = window_len(fs)
    hop = max(1, int(round(hop_sec * fs)))
    return range(0, x.shape[1] - N + 1, hop)
