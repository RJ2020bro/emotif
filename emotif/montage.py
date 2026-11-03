"""Montage-agnostic feature space.

Any headset's channels are placed on the scalp from the 10-05 table, then pooled into 17 fixed
scalp regions with a truncated Gaussian kernel on the sphere. A model trained on region features
runs on any electrode layout; regions with no electrode nearby are masked.

Feature vector (fixed length, see feature_names()):
  region z-DE   17 regions x 5 bands
  asymmetry      7 left/right region pairs x 5 bands  (left - right)
  mask          17 (1 = region observed)
web/src/montage.ts mirrors this file; tests/test_parity.py checks them against each other.
"""
import json
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "data", "electrodes_1005.json")) as _fh:
    POSITIONS = {k.upper(): np.array(v) for k, v in json.load(_fh)["positions"].items()}

REGIONS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "Pz", "P8", "O1", "O2"]
REGION_PAIRS = [("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("T7", "T8"), ("C3", "C4"), ("P7", "P8"), ("O1", "O2")]
SIGMA_DEG = 18.0
MAX_DEG = 30.0
N_BANDS = 5
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

# Electrode layouts of common headsets (used for montage augmentation and cross-device evaluation).
DEVICES = {
    "muse": ["TP9", "AF7", "AF8", "TP10"],
    "emotiv_insight": ["AF3", "AF4", "T7", "T8", "Pz"],
    "emotiv_epoc": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"],
    "openbci_cyton": ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"],
    "neurosity_crown": ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"],
    "unicorn": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"],
    "brainbit": ["T3", "T4", "O1", "O2"],
    "frontal_2ch": ["Fp1", "Fp2"],
}

_PREFIX = re.compile(r"^(EEG|EXG)[\s_:-]*", re.I)


def canonical(name):
    """'EEG Fp1-REF' -> 'FP1'. Returns the upper-case key used in POSITIONS, or None if unknown."""
    n = _PREFIX.sub("", name.strip())
    n = re.split(r"[-/\s]", n)[0].upper() if n else ""
    return n if n in POSITIONS else None


def position(name):
    k = canonical(name)
    return None if k is None else POSITIONS[k]


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def region_weights(channels):
    """Returns W (R x C) with rows summing to 1 for observed regions (0 otherwise) and mask (R,)."""
    R, C = len(REGIONS), len(channels)
    W = np.zeros((R, C))
    pos = [position(c) for c in channels]
    for r, reg in enumerate(REGIONS):
        a = POSITIONS[reg.upper()]
        for c, p in enumerate(pos):
            if p is None:
                continue
            d = _angle_deg(a, p)
            if d <= MAX_DEG:
                W[r, c] = np.exp(-d * d / (2 * SIGMA_DEG * SIGMA_DEG))
    s = W.sum(axis=1)
    mask = (s > 0).astype(float)
    W[s > 0] /= s[s > 0, None]
    return W, mask


def region_features(zde, W, mask):
    """zde: (..., C, 5) per-session z-scored channel DE. Returns (..., 137)."""
    reg = np.einsum("rc,...cb->...rb", W, zde)                     # (..., R, 5)
    idx = {r: i for i, r in enumerate(REGIONS)}
    asym = []
    for l, r in REGION_PAIRS:
        both = mask[idx[l]] * mask[idx[r]]
        asym.append((reg[..., idx[l], :] - reg[..., idx[r], :]) * both)
    asym = np.stack(asym, axis=-2)                                  # (..., 7, 5)
    lead = reg.shape[:-2]
    return np.concatenate([reg.reshape(*lead, -1), asym.reshape(*lead, -1),
                           np.broadcast_to(mask, (*lead, len(REGIONS)))], axis=-1)


def feature_names():
    names = [f"{r}_{b}" for r in REGIONS for b in BAND_NAMES]
    names += [f"asym_{l}-{r}_{b}" for l, r in REGION_PAIRS for b in BAND_NAMES]
    names += [f"mask_{r}" for r in REGIONS]
    return names


N_FEATURES = len(REGIONS) * N_BANDS + len(REGION_PAIRS) * N_BANDS + len(REGIONS)


GRAPH_SIGMA_DEG = 30.0
GRAPH_HOPS = 2


def region_adjacency(sigma_deg=GRAPH_SIGMA_DEG):
    """(R, R) scalp graph between region anchors: Gaussian kernel of great-circle distance (self-loops = 1),
    the distance prior used by DGCNN-style emotion models (Song et al. 2018)."""
    P = np.array([POSITIONS[r.upper()] for r in REGIONS])
    d = np.degrees(np.arccos(np.clip(P @ P.T, -1.0, 1.0)))
    return np.exp(-d * d / (2 * sigma_deg * sigma_deg))


def graph_features(x, adjacency=None, hops=GRAPH_HOPS):
    """Append mask-aware graph diffusion to region features. x: (..., N_FEATURES).
    Each hop replaces every region by the kernel-weighted mean of its *observed* neighbours (itself included),
    so regions a headset does not cover are filled from the nearest covered ones. Returns
    (..., N_FEATURES + hops * R * 5): [x, A x, A^2 x] over the region DE block."""
    A = region_adjacency() if adjacency is None else np.asarray(adjacency)
    R = len(REGIONS)
    x = np.asarray(x, dtype=np.float64)
    reg = x[..., :R * N_BANDS].reshape(*x.shape[:-1], R, N_BANDS)
    mask = x[..., -R:]
    Wm = A * mask[..., None, :]                                          # (..., R, R) keep observed sources
    s = Wm.sum(axis=-1, keepdims=True)
    Wm = np.where(s > 0, Wm / np.where(s > 0, s, 1.0), 0.0)
    out, h = [x], reg
    for _ in range(hops):
        h = Wm @ h
        out.append(h.reshape(*x.shape[:-1], R * N_BANDS))
    return np.concatenate(out, axis=-1)


def device_subset(dataset_channels, device, max_deg=35.0, min_coverage=0.5):
    """Indices of dataset channels that stand in for `device` electrodes (nearest within max_deg, no reuse).
    Returns [] when fewer than min_coverage of the device's electrodes have a stand-in."""
    pos = [position(c) for c in dataset_channels]
    used, out = set(), []
    for e in DEVICES[device]:
        pe = position(e)
        best, bd = None, max_deg
        for i, p in enumerate(pos):
            if p is None or i in used:
                continue
            d = _angle_deg(pe, p)
            if d <= bd:
                best, bd = i, d
        if best is not None:
            used.add(best)
            out.append(best)
    if len(out) < min_coverage * len(DEVICES[device]):
        return []
    return sorted(out)


def describe(channels):
    """Human-readable coverage report."""
    W, mask = region_weights(channels)
    unknown = [c for c in channels if position(c) is None]
    return {"observed_regions": [r for r, m in zip(REGIONS, mask) if m],
            "missing_regions": [r for r, m in zip(REGIONS, mask) if not m],
            "unknown_channels": unknown}
