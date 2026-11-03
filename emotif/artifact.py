"""Learned artifact detector: classifies one channel of one 2 s window as clean EEG, ocular (blink / eye
movement) or muscle (EMG) contamination, *relative to that channel's own resting baseline*.
Trained on EEGdenoiseNet (Zhang et al. 2021, J. Neural Eng.) by research/train_artifact.py, with simulated
headset differences (noise floor, low-pass, mains hum, 128/256 Hz) so it transfers to consumer devices.
app/src/artifact.ts mirrors this file.

Per-window features (sampling-rate independent: 2 s windows, bands in Hz):
  0-4   log relative band power, delta theta alpha beta gamma (1-45 Hz)
  5     kurtosis (excess), sign(k) * log(1 + |k|)
  6     skewness
  7     log(peak-to-peak / std)
  8     log Hjorth mobility in Hz  ( std(dx) / std(x) * fs / 2pi )
  9     log Hjorth complexity     ( mobility(dx) / mobility(x) )
  10    log std (amplitude)
Model input = features minus the channel's baseline median (so device gain and noise floor cancel).
"""
import json
import os

import numpy as np

from . import features as F

CLASSES = ["clean", "ocular", "muscle"]
FEATURE_NAMES = ["rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
                 "kurtosis", "skewness", "ptp_over_std", "mobility_hz", "complexity", "log_std"]
N_FEATURES = len(FEATURE_NAMES)
EPS = 1e-12
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(os.path.dirname(_HERE), "models", "artifact_detector.json")


def channel_features(win, fs, bp=None):
    """win: (C, N) filtered window. bp: optional (C, 5) band powers already computed. Returns (C, 11)."""
    win = np.asarray(win, dtype=np.float64)
    if bp is None:
        bp = F.band_powers(win, fs)
    rel = np.log(bp / (bp.sum(axis=1, keepdims=True) + EPS) + EPS)
    d = F.detrend(win)
    m2 = np.mean(d ** 2, axis=1) + EPS
    m3 = np.mean(d ** 3, axis=1)
    m4 = np.mean(d ** 4, axis=1)
    kurt = m4 / m2 ** 2 - 3.0
    skew = m3 / m2 ** 1.5
    std = np.sqrt(m2)
    ptp = d.max(axis=1) - d.min(axis=1)
    dx = np.diff(d, axis=1)
    ddx = np.diff(dx, axis=1)
    sx = np.sqrt(np.mean(dx ** 2, axis=1) + EPS)
    sxx = np.sqrt(np.mean(ddx ** 2, axis=1) + EPS)
    mob = sx / std
    comp = (sxx / sx) / mob
    return np.column_stack([rel, np.sign(kurt) * np.log1p(np.abs(kurt)), skew, np.log(ptp / std + EPS),
                            np.log(mob * fs / (2 * np.pi) + EPS), np.log(comp + EPS), np.log(std + EPS)])


def baseline(feats):
    """feats: (n_windows, C, 11) -> per-channel baseline (C, 11): robust median over windows."""
    return np.median(np.asarray(feats), axis=0)


def load(path=DEFAULT_MODEL):
    with open(path) as fh:
        m = json.load(fh)
    if m.get("format") != "emotif-artifact-v2":
        raise ValueError(f"{path} is not an Emotif v2 artifact model")
    return m


def predict_proba(model, rel_feats):
    from .models import forward
    return forward(model["layers"], rel_feats)


def bad_channels(model, feats, base, threshold=None):
    """feats (C, 11) for one window, base (C, 11). Returns (bad bool (C,), probabilities (C, 3))."""
    thr = model.get("mask_threshold", 0.8) if threshold is None else threshold
    p = predict_proba(model, feats - base)
    return (1.0 - p[:, 0]) >= thr, p
