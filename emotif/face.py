"""Facial-activity indices that the live app can compute exactly as training does (mirror: app/src/face.ts).

Per channel and 2 s window, from the filtered signal *before* artifact repair:
  eye    log peak-to-peak of the 0-3 Hz component (blinks, eye movements): the signal is already 0.5 Hz
         high-passed, then two cascaded 3 Hz low-pass biquads (RBJ, Q = 1/sqrt 2) run causally over the whole
         recording from a zero state, so there are no window-edge transients and the app reproduces it sample
         for sample
  jaw1   log power 20-30 Hz  } scalp power above ~20 Hz is dominated by muscle (Whitham et al. 2007)
  jaw2   log power 30-45 Hz  }
Each is taken relative to the person's own median (training: over the recording session; app: over the resting
baseline), then pooled into the 17 scalp regions with the same weights and channel mask as the EEG features.
The decoder (models.BrainModel) removes from the EEG features everything these 51 numbers and their squares predict.
"""
import numpy as np
from scipy.signal import sosfilt

from . import features as F
from . import montage as M

EYE_LP_HZ = 3.0
MUSCLE_BANDS = ((20.0, 30.0), (30.0, 45.0))
N_IDX = 3
N_REGION_FACE = len(M.REGIONS) * N_IDX
N_FACE_INPUT = 2 * N_REGION_FACE          # pooled indices and their squares


def eye_sos(fs):
    w0 = 2.0 * np.pi * EYE_LP_HZ / fs
    cw, al = np.cos(w0), np.sin(w0) / (2.0 * (1.0 / np.sqrt(2.0)))
    b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2]) / (1 + al)
    a = np.array([1.0, -2 * cw / (1 + al), (1 - al) / (1 + al)])
    sec = np.concatenate([b, a])
    return np.vstack([sec, sec])


def slow_signal(x, fs):
    """Causal 0-3 Hz component of a filtered recording (C, T), zero initial state."""
    return sosfilt(eye_sos(fs), np.asarray(x, dtype=np.float64), axis=1)


def window_indices(win, slow_win, fs):
    """(C, 3): eye, jaw1, jaw2 for one window of the filtered signal and the matching slow-signal window."""
    eye = np.log(np.ptp(slow_win, axis=1) + 1e-9)
    p = F.band_powers(win, fs, MUSCLE_BANDS)
    return np.column_stack([eye, np.log(p + F.EPS)])


def pool(face, W):
    """face: (..., C, 3) relative indices of the channels W was built for; W (R, C). Returns (..., R*3)."""
    reg = np.einsum("rc,...cb->...rb", W, face)
    return reg.reshape(*reg.shape[:-2], -1)


def design(face_region):
    """Decoder face input: pooled indices and their squares, (..., 102)."""
    f = np.asarray(face_region, dtype=np.float64)
    return np.concatenate([f, f ** 2], axis=-1)
