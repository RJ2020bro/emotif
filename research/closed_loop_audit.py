#!/usr/bin/env python3
"""Closed-loop audit: what does the face do to the brain-controlled *world*, not just to the classifier?

  python research/closed_loop_audit.py --gameemo data/external/gameemo --stress data/external/stress_insight \
                                       --out results/closed_loop [--jobs 2] [--perms 200]
  python research/closed_loop_audit.py --figure --out results/closed_loop

Every held-out person's real recording is replayed, window by window, through the decoders and through the app's
own closed loop, ported from app/src (engine.ts, director.ts, safety.ts, world.ts, interact.ts):

  2 s windows every 0.25 s (engine.ts HOP_SEC) -> amplitude gate 200 uV p-p after ASR (engine.ts default rejectUv)
  -> decoder (value = expected class, conf = top probability) -> Kalman1 (q 0.002, r = 0.03 / max(0.2, conf); a
  rejected window or a brain-only face hold only predicts) -> director, mirror mode (slew 0.25 per s) -> world
  slewToward (tau 1.2 s, 0.5 per s) at 60 frames per s -> cloud cover, storm, fog density, saturation (world.ts
  updateWorld, style "natural", mood 0.8 = DEFAULT_SPEC) and the displayed weather (moodWeather with the 3 s dwell
  and the particle fade of updateWeather).
  The focus/relax control (lantern / beacon; formerly a crystal): AlphaControl (interact.ts: threshold 0.7, dwell 1 s, tau 1.5 s, hysteresis at half the threshold,
  hold) on the cleaned region features, posterior alpha regions (Pz, P7, P8, O1, O2; Pz on the Insight).

Decoders (LOSO on GAMEEMO: fitted on the other 27 people with the same features, windows (2 s hop) and settings as
research/brain_decoder.py, which this pipeline reproduces; on the Insight stress data the GAMEEMO models fitted on
all 28 people at the four shared electrodes AF3, AF4, T7, T8, i.e. other people from another lab):
  standard_filt   region features after filtering only (no ASR, no detector), logistic regression
  standard_clean  region features after ASR + artifact detector (what the app's standard model reads)
  brain           the shipped brain-only decoder: ridge prediction from the live face indices and their squares
                  subtracted, then logistic regression; holds when a standardized pooled face index > the training
                  99.5th percentile (10.2 SD on GAMEEMO)
Normalization: the z-score of every feature, the detector baseline, the face-index centre and the ASR calibration
come from the person's own clean session (as in training) and are kept fixed when artifacts are added, so an
injected artifact changes only the signal (the app's "baseline" normalization).

Analyses
 1  World-level face share. Out-of-person R^2 of each world parameter from an independent face measure: the audit's
    zero-phase face indices (per electrode: 0.5-3 Hz peak-to-peak, 20-30 and 30-45 Hz log power) causally averaged
    over the last 2, 8 and 30 s, with their squares (ridge, fitted on the other people). The same measure is applied
    to the per-window decoder output, the smoothed value the app shows and every world parameter. Significance:
    recording-swap permutations (each recording gets the face series of a random recording of another person).
    Per-person comparisons standard vs brain-only: Wilcoxon signed-rank. The audit's original per-window face share
    (in-sample adjusted R^2 on unsmoothed face indices, research/brain_decoder.face_leakage) is reported too.
 2  Counterfactual artifact injection into real recordings: 30 s with, 30 s without, alternating. Blinks (Gaussian
    deflections, sigma 60-100 ms, every 2.5 +- 0.5 s, peak 100 / 150 / 200 uV at AF3/AF4, decaying over the scalp),
    jaw clenches (20-45 Hz noise, bursts 1.5 s on / 1 s off, RMS 10 / 20 / 40 uV at T7/T8) and a wider 13-45 Hz
    clench; positive controls: genuine alpha change (8-13 Hz component of every channel x2 or x0.5) and alpha x2 with
    blinks. Measured against the same recording replayed without injection: how far each decoder's output, what the
    app shows and each world parameter move, how often the displayed weather or the affect quadrant differs.
 3  False command rate of the focus/relax control: AlphaControl over the same replays, with the brain-only hold (the app) and
    without it; commands per minute during injected epochs minus the same epochs of the clean replay. Neither
    dataset has eyes-closed segments, so the genuine-change controls are the alpha-scaled epochs.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from emotif import artifact as A  # noqa: E402
from emotif import asr as ASR  # noqa: E402
from emotif import audit as AU  # noqa: E402
from emotif import datasets as D  # noqa: E402
from emotif import face as FACE  # noqa: E402
from emotif import features as F  # noqa: E402
from emotif import montage as M  # noqa: E402

# ---------------------------------------------------------------- the app's constants (app/src)
HOP = 0.25                      # engine.ts HOP_SEC
TRAIN_HOP = 2.0                 # research/brain_decoder.py --hop 2 (results/brain_gameemo_v2)
REJECT_UV = 200.0               # engine.ts settings.rejectUv (app default)
KALMAN_Q = 0.002                # engine.ts Kalman1(0.002)
DIRECTOR_RATE = 0.25            # director.ts maxRate
SLEW_RATE, SLEW_TAU = 0.5, 1.2  # safety.ts MAX_RATE, TAU
FPS = 60                        # frames per second of the simulated render loop
MOOD = 0.8                      # worldspec.ts DEFAULT_SPEC.mood (style "natural")
WEATHER_DWELL = 3.0             # world.ts updateWeather
WEATHERS = ["none", "fireflies", "petals", "rain", "snow"]
WX_OPACITY = np.array([0.0, 0.95, 0.9, 0.55, 0.9])   # world.ts WX[...].opacity
CTRL_THRESHOLD, CTRL_DWELL, CTRL_TAU = 0.7, 1.0, 1.5  # interact.ts AlphaControl defaults
CTRL_POSTERIOR = [M.REGIONS.index(r) for r in ("Pz", "P7", "P8", "O1", "O2")]
ALPHA_BAND = 2
MASK0 = len(M.REGIONS) * M.N_BANDS + len(M.REGION_PAIRS) * M.N_BANDS
LR_C = 0.3
DECODERS = ["standard_filt", "standard_clean", "brain"]
TARGETS = ["valence", "arousal"]
PARAMS = ["cloud", "storm", "fog", "saturation"]
EPOCH_SEC = 30.0                # artifact epochs: 30 s off, 30 s on, alternating
INSIGHT_DEC = ["AF3", "T7", "T8", "AF4"]   # electrodes GAMEEMO and the Insight share

# ---------------------------------------------------------------- injected artifacts
# Blink potential relative to AF3/AF4 (fronto-polar sites, which GAMEEMO and the Insight lack, would be larger);
# jaw-muscle (temporalis / masseter) EMG relative to T7/T8. Gains fall off with distance from the source.
BLINK_GAIN = {"FP1": 1.25, "FP2": 1.25, "AF3": 1.0, "AF4": 1.0, "F7": 0.7, "F8": 0.7, "F3": 0.65, "F4": 0.65,
              "FZ": 0.65, "FC5": 0.4, "FC6": 0.4, "T7": 0.2, "T8": 0.2, "C3": 0.25, "C4": 0.25, "CZ": 0.25,
              "P7": 0.08, "P8": 0.08, "PZ": 0.08, "O1": 0.05, "O2": 0.05}
JAW_GAIN = {"T7": 1.0, "T8": 1.0, "F7": 0.7, "F8": 0.7, "FC5": 0.5, "FC6": 0.5, "P7": 0.4, "P8": 0.4,
            "AF3": 0.25, "AF4": 0.25, "F3": 0.25, "F4": 0.25, "FP1": 0.3, "FP2": 0.3, "C3": 0.3, "C4": 0.3,
            "O1": 0.15, "O2": 0.15, "PZ": 0.15, "FZ": 0.15, "CZ": 0.15}
CONDITIONS = {   # name: (kind, amplitude or gain, extra)
    "clean": ("none", 0.0),
    "blink100": ("blink", 100.0), "blink150": ("blink", 150.0), "blink200": ("blink", 200.0),
    "jaw10": ("jaw", 10.0), "jaw20": ("jaw", 20.0), "jaw40": ("jaw", 40.0),
    "jawwide20": ("jawwide", 20.0),
    "alpha_up": ("alpha", 2.0), "alpha_down": ("alpha", 0.5),
    "alpha_up_blink150": ("alpha+blink", 2.0),
}
ARTIFACT_CONDS = ["blink100", "blink150", "blink200", "jaw10", "jaw20", "jaw40", "jawwide20"]
BRAIN_CONDS = ["alpha_up", "alpha_down"]


def on_epochs(t):
    """True during injected epochs (the second, fourth, ... 30 s of a recording)."""
    return (np.floor(np.asarray(t) / EPOCH_SEC) % 2) == 1


def _ramp(on, fs, ramp_sec):
    """Raised-cosine edges for a 0/1 sequence."""
    k = max(1, int(round(ramp_sec * fs)))
    w = np.sin(np.linspace(0, np.pi, 2 * k + 1)) ** 2
    w /= w.sum()
    return np.clip(np.convolve(on.astype(float), w, mode="same"), 0, 1)


def artifact_signal(kind, amp, channels, n, fs, rng_seed):
    """(C, n) additive artifact for one recording, on the injected epochs only. rng_seed: tuple identifying the
    recording and artifact kind, so every amplitude of one kind uses the same blink times and the same noise."""
    t = np.arange(n) / fs
    on = on_epochs(t)
    names = [M.canonical(c) for c in channels]
    out = np.zeros((len(channels), n))
    if kind == "blink":
        rng = np.random.default_rng([*rng_seed, 1])
        g = np.array([BLINK_GAIN.get(c, 0.05) for c in names])
        times, s = [], 1.0
        while s < n / fs:
            times.append(s)
            s += 2.5 + rng.uniform(-0.5, 0.5)
        sig = np.zeros(n)
        for b in times:
            if not on_epochs(b):
                continue
            sd = rng.uniform(0.06, 0.10)
            i0, i1 = max(0, int((b - 5 * sd) * fs)), min(n, int((b + 5 * sd) * fs) + 1)
            sig[i0:i1] += np.exp(-((t[i0:i1] - b) ** 2) / (2 * sd * sd))
        out = amp * g[:, None] * sig[None]
    elif kind in ("jaw", "jawwide"):
        rng = np.random.default_rng([*rng_seed, 2 if kind == "jaw" else 3])
        lo = 20.0 if kind == "jaw" else 13.0
        g = np.array([JAW_GAIN.get(c, 0.15) for c in names])
        sos = butter(4, [lo, min(45.0, 0.45 * fs)], "bandpass", fs=fs, output="sos")
        noise = sosfiltfilt(sos, rng.normal(0, 1, (len(channels), n)), axis=1)
        noise /= noise.std(axis=1, keepdims=True) + 1e-12
        burst = (np.mod(t, 2.5) < 1.5) & on
        env = _ramp(burst, fs, 0.05)
        out = amp * g[:, None] * noise * env[None]
    return out


def inject(x_raw, fs, channels, cond, rng_seed):
    kind, amp = CONDITIONS[cond]
    if kind == "none":
        return x_raw
    n = x_raw.shape[1]
    x = x_raw.copy()
    if kind.startswith("alpha"):
        sos = butter(4, [8.0, 13.0], "bandpass", fs=fs, output="sos")
        alpha = sosfiltfilt(sos, x_raw - x_raw.mean(axis=1, keepdims=True), axis=1)
        env = _ramp(on_epochs(np.arange(n) / fs), fs, 0.5)
        x = x + (amp - 1.0) * env[None] * alpha
        if kind == "alpha+blink":
            x = x + artifact_signal("blink", 150.0, channels, n, fs, rng_seed)
        return x
    return x + artifact_signal(kind, amp, channels, n, fs, rng_seed)


# ---------------------------------------------------------------- vectorized mirrors of features.py / face.py / audit.py
def frames(x, N, hop_n):
    """(C, T) -> (W, C, N) windows starting at 0, hop_n, ... (features.sliding_windows)."""
    v = np.lib.stride_tricks.sliding_window_view(x, N, axis=1)[:, ::hop_n]
    return np.ascontiguousarray(v.transpose(1, 0, 2))


def psd_batch(win, fs):
    """Welch PSD exactly as features.band_powers computes it, for (..., N)."""
    N = win.shape[-1]
    L = int(round(F.SEG_SEC * fs))
    step = L // 2
    starts = list(range(0, N - L + 1, step))
    n = np.arange(L)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / L)
    scale = 2.0 / (fs * np.sum(w * w))
    psd = np.zeros(win.shape[:-1] + (L // 2 + 1,))
    for s in starts:
        seg = win[..., s:s + L]
        seg = seg - seg.mean(axis=-1, keepdims=True)
        X = np.fft.rfft(seg * w, axis=-1)
        psd += (X.real ** 2 + X.imag ** 2) * scale
    psd /= len(starts)
    return psd, np.arange(L // 2 + 1) * fs / L, fs / L


def bands(psd, freqs, df, bands_):
    return np.stack([psd[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) * df for lo, hi in bands_], axis=-1)


EEG_BANDS = [(lo, hi) for _, lo, hi in F.BANDS]


def detrend_stats(win):
    """features.channel_quality's detrended peak-to-peak and SD, (W, C)."""
    N = win.shape[-1]
    t = np.arange(N) - (N - 1) / 2.0
    slope = (win @ t) / np.sum(t * t)
    d = win - win.mean(axis=-1, keepdims=True) - slope[..., None] * t
    return d.max(axis=-1) - d.min(axis=-1), d.std(axis=-1)


def quality(ptp, sd, reject_uv):
    """features.channel_quality (and dsp.ts channelQuality) for every window: good (W, C), reject (W,)."""
    flat = sd < F.FLAT_UV
    if reject_uv <= 0:
        good = ~flat
        return good, ~good.any(axis=1)
    good = ~flat & (ptp <= F.BAD_UV)
    reject = ~good.any(axis=1) | np.any(good & (ptp > reject_uv), axis=1)
    return good, reject


class Pooler:
    """Region pooling with the weights of whichever channels are usable in each window (models.featurize)."""
    def __init__(self, channels):
        self.channels = list(channels)
        self.cache = {}

    def weights(self, pat):
        key = pat.tobytes()
        if key not in self.cache:
            cols = np.flatnonzero(pat)
            self.cache[key] = (cols, *M.region_weights([self.channels[c] for c in cols]))
        return self.cache[key]

    def pool(self, z, avail, face=None):
        """z (W, C, 5) z-scored DE; avail (W, C). Returns region features (W, 137) and, if face (W, C, 3) is given,
        the face indices pooled the same way (W, 51)."""
        W_ = z.shape[0]
        X = np.zeros((W_, M.N_FEATURES))
        Q = np.zeros((W_, FACE.N_REGION_FACE)) if face is not None else None
        if W_ == 0:
            return X, Q
        pats, inv = np.unique(avail, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        for k, pat in enumerate(pats):
            rows = np.flatnonzero(inv == k)
            if not pat.any():
                continue
            cols, Wr, mask = self.weights(pat)
            X[rows] = M.region_features(z[rows][:, cols, :], Wr, mask)
            if face is not None:
                Q[rows] = FACE.pool(face[rows][:, cols, :], Wr)
        return X, Q


def alpha_z(X):
    """interact.ts alphaZ for every row of region features (NaN where no region is observed)."""
    obs = X[:, MASK0:MASK0 + len(M.REGIONS)] > 0.5
    a = X[:, ALPHA_BAND:len(M.REGIONS) * M.N_BANDS:M.N_BANDS]
    post = np.zeros(len(M.REGIONS), bool)
    post[CTRL_POSTERIOR] = True
    use = obs & post
    none = ~use.any(axis=1)
    use[none] = obs[none]
    n = use.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (a * use).sum(axis=1) / np.maximum(n, 1), np.nan)


# ---------------------------------------------------------------- one person's pipeline
class Person:
    """Filtering, ASR, detector, z-scored DE, region features and both kinds of face indices for one person's
    recordings, with the normalization learned on the person's clean session (models.extract, audit.artifact_evidence)."""

    def __init__(self, trials, detector, asr_k=ASR.DEFAULT_K):
        trials = sorted(trials, key=lambda t: t.trial_id)
        self.trials = trials
        self.subject = trials[0].subject
        self.fs = float(trials[0].fs)
        self.channels = list(trials[0].channels)
        self.det = detector
        self.det_layers = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "act": L["act"]} for L in detector["layers"]]
        self.det_thr = detector.get("mask_threshold", 0.8)
        self.pool = Pooler(self.channels)
        self.N = F.window_len(self.fs)
        xs = [t.x if t.filtered else F.filter_signal(t.x, self.fs) for t in trials]
        cal = np.concatenate(xs, axis=1)[:, : int(600.0 * self.fs)]
        try:
            self.asr = ASR.calibrate(cal, self.fs, asr_k)
        except ValueError:
            self.asr = None
        self._calibrate(xs)

    def _detector(self, feats):
        z = feats
        for L in self.det_layers:
            z = z @ L["W"] + L["b"]
            if L["act"] == "relu":
                z = np.maximum(z, 0)
            else:
                z = np.exp(z - z.max(axis=1, keepdims=True))
                z /= z.sum(axis=1, keepdims=True)
        return z

    def _raw(self, xf, hop):
        """Calibration-free per-window quantities of one filtered recording."""
        fs, N = self.fs, self.N
        hn = max(1, int(round(hop * fs)))
        xc = ASR.clean(xf, self.asr, fs)[0] if self.asr is not None else xf
        Wf = frames(xf, N, hn)
        Wc = frames(xc, N, hn) if self.asr is not None else Wf
        out = {"start": np.arange(Wf.shape[0]) * hn}
        psd, fr, df = psd_batch(Wf, fs)
        out["bp_f"] = bands(psd, fr, df, EEG_BANDS)
        mus = bands(psd, fr, df, FACE.MUSCLE_BANDS)
        slow_live = frames(FACE.slow_signal(xf, fs), N, hn)
        out["face_live"] = np.concatenate([np.log(np.ptp(slow_live, axis=-1) + 1e-9)[..., None], np.log(mus + F.EPS)], axis=-1)
        slow_aud = frames(AU.slow_component(xf, fs), N, hn)
        f_, p_ = welch(Wf, fs, nperseg=int(round(F.SEG_SEC * fs)), axis=-1)
        aud_m = np.stack([np.log(p_[..., (f_ >= lo) & (f_ < hi)].sum(axis=-1) + 1e-12) for lo, hi in AU.MUSCLE_BANDS], axis=-1)
        out["face_audit"] = np.concatenate([np.log(np.ptp(slow_aud, axis=-1) + 1e-9)[..., None], aud_m], axis=-1)
        out["ptp_f"], out["sd_f"] = detrend_stats(Wf)
        if Wc is Wf:
            out["bp_c"], out["ptp_c"], out["sd_c"] = out["bp_f"], out["ptp_f"], out["sd_f"]
        else:
            psd_c, _, _ = psd_batch(Wc, fs)
            out["bp_c"] = bands(psd_c, fr, df, EEG_BANDS)
            out["ptp_c"], out["sd_c"] = detrend_stats(Wc)
        W_, C = Wc.shape[:2]
        out["art"] = A.channel_features(Wc.reshape(W_ * C, N), fs, out["bp_c"].reshape(W_ * C, -1)).reshape(W_, C, -1)
        return out

    @staticmethod
    def _zparams(de):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu, sd = np.nanmean(de, axis=0), np.nanstd(de, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where(np.isnan(sd) | (sd < 1e-6), 1.0, sd)
        return mu, sd

    def _calibrate(self, xs):
        """Normalization from the clean session at the training hop (models.extract with reject_uv = 0, as trained),
        and the training rows (windows kept by the filter-only and the cleaned pipeline)."""
        raws = [self._raw(x, TRAIN_HOP) for x in xs]
        cat = {k: np.concatenate([r[k] for r in raws]) for k in raws[0] if k != "start"}
        rec = np.concatenate([np.full(len(r["start"]), i) for i, r in enumerate(raws)])
        start = np.concatenate([r["start"] for r in raws])
        # filter only
        good_f, rej_f = quality(cat["ptp_f"], cat["sd_f"], 0.0)
        de_f = F.de_from_power(cat["bp_f"])
        de_f[~good_f] = np.nan
        self.z_f = self._zparams(de_f[~rej_f])
        # cleaned: detector baseline over windows that pass the quality rule, then the detector's channel mask
        good_c, rej_c = quality(cat["ptp_c"], cat["sd_c"], 0.0)
        self.det_base = A.baseline(cat["art"][~rej_c])
        bad = self._bad(cat["art"])
        mask_c = good_c & ~bad
        keep_c = ~rej_c & mask_c.any(axis=1)
        de_c = F.de_from_power(cat["bp_c"])
        de_c[~mask_c] = np.nan
        self.z_c = self._zparams(de_c[keep_c])
        self.face_live_med = np.median(cat["face_live"][keep_c], axis=0)
        self.face_audit_med = np.median(cat["face_audit"], axis=0)
        keep = keep_c & ~rej_f
        zf = (de_f - self.z_f[0]) / self.z_f[1]
        zc = (de_c - self.z_c[0]) / self.z_c[1]
        Xf, _ = self.pool.pool(zf[keep], good_f[keep])
        Xc, Q = self.pool.pool(zc[keep], mask_c[keep], cat["face_live"][keep] - self.face_live_med)
        tr = self.trials
        self.train = {"X_filt": Xf, "X_clean": Xc, "face_live": Q,
                      "face": (cat["face_audit"][keep] - self.face_audit_med).reshape(int(keep.sum()), -1),
                      "rec": rec[keep], "start": start[keep],
                      "y": {t: np.array([tr[i].labels.get(t, np.nan) for i in rec[keep]], float) for t in TARGETS + ["stress"]}}

    def _bad(self, art):
        W_, C, K = art.shape
        p = self._detector((art - self.det_base[None]).reshape(W_ * C, K))
        return ((1.0 - p[:, 0]) >= self.det_thr).reshape(W_, C)

    def replay(self, x_raw, hop=HOP, gates=(REJECT_UV, 0.0)):
        """One (possibly injected) raw recording through the live pipeline with the clean normalization.
        Returns {gate: {X_filt, use_filt, X_clean, use_clean, face_live}} plus t (window end, s) and face_audit."""
        xf = F.filter_signal(x_raw, self.fs)
        r = self._raw(xf, hop)
        bad = self._bad(r["art"])
        de_f, de_c = F.de_from_power(r["bp_f"]), F.de_from_power(r["bp_c"])
        out = {"t": (r["start"] + self.N) / self.fs,
               "face_audit": (r["face_audit"] - self.face_audit_med).reshape(len(r["start"]), -1)}
        fl = r["face_live"] - self.face_live_med
        for g in gates:
            good_f, rej_f = quality(r["ptp_f"], r["sd_f"], g)
            good_c, rej_c = quality(r["ptp_c"], r["sd_c"], g)
            mask_c = good_c & ~bad
            use_c = ~rej_c & mask_c.any(axis=1)
            zf = (np.where(good_f[..., None], de_f, np.nan) - self.z_f[0]) / self.z_f[1]
            zc = (np.where(mask_c[..., None], de_c, np.nan) - self.z_c[0]) / self.z_c[1]
            Xf, _ = self.pool.pool(zf, good_f)
            Xc, Q = self.pool.pool(zc, mask_c, fl)
            out[g] = {"X_filt": Xf, "use_filt": ~rej_f, "X_clean": Xc, "use_clean": use_c, "face_live": Q}
        return out


# ---------------------------------------------------------------- decoders (research/brain_decoder.py, per fold)
class Logit:
    def __init__(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.mu, sd = X.mean(0), X.std(0)
        self.sd = np.where(sd < 1e-9, 1.0, sd)
        self.clf = LogisticRegression(C=LR_C, max_iter=3000).fit((X - self.mu) / self.sd, y)
        self.cls = self.clf.classes_.astype(float)

    def __call__(self, X):
        P = self.clf.predict_proba((X - self.mu) / self.sd)
        return P @ np.clip(self.cls, -1, 1), P.max(axis=1)


class FoldModels:
    """standard_filt, standard_clean and brain for one target, fitted on training rows (audit._residualize, alpha 1;
    brain_decoder.hold_mask)."""

    def __init__(self, tr, target):
        y = tr["y"][target]
        ok = ~np.isnan(y)
        y = y[ok].astype(int)
        self.std_f = Logit(tr["X_filt"][ok], y)
        self.std_c = Logit(tr["X_clean"][ok], y)
        Q = tr["face_live"][ok]
        Aq = np.hstack([Q, Q ** 2])
        self.a_mu, sd = Aq.mean(0), Aq.std(0)
        self.a_sd = np.where(sd < 1e-9, 1.0, sd)
        Z = self._z(Q)
        reg = np.eye(Z.shape[1])
        reg[-1, -1] = 0.0
        E = tr["X_clean"][ok]
        self.B = np.linalg.solve(Z.T @ Z + reg, Z.T @ E)
        self.brain = Logit(E - Z @ self.B, y)
        self.h_mu, sd = Q.mean(0), Q.std(0)
        self.h_sd = np.where(sd < 1e-9, 1.0, sd)
        self.hold_z = float(np.percentile(np.abs((Q - self.h_mu) / self.h_sd).max(axis=1), 99.5))

    def _z(self, Q):
        Aq = (np.hstack([Q, Q ** 2]) - self.a_mu) / self.a_sd
        return np.hstack([Aq, np.ones((len(Aq), 1))])

    def held(self, Q):
        return np.abs((Q - self.h_mu) / self.h_sd).max(axis=1) > self.hold_z

    def predict(self, Xf, Xc, Q):
        return {"standard_filt": self.std_f(Xf), "standard_clean": self.std_c(Xc),
                "brain": self.brain(Xc - self._z(Q) @ self.B)}


def stack_train(persons, exclude=None):
    ps = [p for p in persons if p.subject != exclude]
    out = {k: np.concatenate([p.train[k] for p in ps]) for k in ("X_filt", "X_clean", "face_live", "face")}
    out["y"] = {t: np.concatenate([p.train["y"][t] for p in ps]) for t in ps[0].train["y"]}
    out["subject"] = np.concatenate([np.full(len(p.train["rec"]), p.subject) for p in ps])
    return out


# ---------------------------------------------------------------- the app's closed loop
def kalman(z, r, ok, q=KALMAN_Q):
    """engine.ts Kalman1 over windows: update(z, r) where ok, predict() elsewhere. (S, W) arrays."""
    S, W_ = z.shape
    x, P = np.zeros(S), np.ones(S)
    out = np.empty((S, W_))
    z = np.where(ok, z, 0.0)
    for i in range(W_):
        P = P + q
        K = P / (P + r[:, i])
        u = ok[:, i]
        x = np.where(u, x + K * (z[:, i] - x), x)
        P = np.where(u, P * (1 - K), P)
        out[:, i] = x
    return out


def mood_weather(v, a):
    """world.ts moodWeather as an index into WEATHERS."""
    return np.where(v >= 0.2, np.where(a >= 0.1, 2, 1), np.where(v <= -0.2, np.where(a >= 0.1, 3, 4), 0))


def slew_toward(cur, target, dt, max_rate=SLEW_RATE, tau=SLEW_TAU):
    smoothed = cur + (target - cur) * (1 - np.exp(-dt / tau))
    return cur + np.clip(smoothed - cur, -max_rate * dt, max_rate * dt)


def world(dec_v, dec_a, hop=HOP, fps=FPS):
    """director.ts (mirror) + world.ts frame loop driven by the engine output after each window. Returns the world's
    displayed state (cur v, a) and the displayed weather index, sampled when the next window arrives."""
    S, W_ = dec_v.shape
    sub = int(round(hop * fps))
    dt = hop / sub
    env_v, env_a, cur_v, cur_a = (np.zeros(S) for _ in range(4))
    w_type, w_pend = np.zeros(S, int), np.zeros(S, int)
    w_for, w_op = np.zeros(S), np.zeros(S)
    out_v, out_a, out_w = np.empty((S, W_)), np.empty((S, W_)), np.empty((S, W_), int)
    step = DIRECTOR_RATE * dt
    fade = 0.8 * dt
    for i in range(W_):
        dv, da = dec_v[:, i], dec_a[:, i]
        for _ in range(sub):
            env_v = env_v + np.clip(dv - env_v, -step, step)
            env_a = env_a + np.clip(da - env_a, -step, step)
            cur_v = slew_toward(cur_v, env_v, dt)
            cur_a = slew_toward(cur_a, env_a, dt)
            want = mood_weather(cur_v, cur_a)
            new = want != w_pend
            w_pend = np.where(new, want, w_pend)
            w_for = np.where(new, 0.0, w_for) + dt
            want = np.where((w_for < WEATHER_DWELL) & (want != w_type), w_type, want)
            diff = want != w_type
            op_down = np.maximum(0.0, w_op - fade)
            inten = np.clip(0.35 + np.hypot(cur_v, cur_a), 0.35, 1.0)
            op_same = w_op + np.clip(WX_OPACITY[w_type] * inten - w_op, -fade, fade)
            w_op = np.where(diff, op_down, op_same)
            w_type = np.where(diff & (op_down == 0), want, w_type)
        out_v[:, i], out_a[:, i], out_w[:, i] = cur_v, cur_a, w_type
    return out_v, out_a, out_w


def world_params(v, a, m=MOOD):
    """world.ts updateWorld, style "natural", no personal state: cloud cover, storm, fog density (relative to the
    neutral world's) and colour saturation."""
    ar, neg = (a + 1) / 2, np.maximum(0, -v)
    return {"cloud": np.clip(0.22 + 0.55 * m * neg + 0.12 * m * ar * neg, 0, 0.95),
            "storm": np.clip(m * neg * ar * 1.6, 0, 1),
            "fog": 1 + 0.9 * m * neg,
            "saturation": np.clip(1 + 0.4 * m * v, 0.55, 1.4)}


def alpha_control(z, held, dt=HOP, threshold=CTRL_THRESHOLD, dwell=CTRL_DWELL, tau=CTRL_TAU):
    """interact.ts AlphaControl.update over windows. z (S, W): alphaZ (NaN when no features); held (S, W).
    Returns level (S, W) and command (S, W): 0 idle, 1 focus, 2 relax."""
    S, W_ = z.shape
    level, since = np.zeros(S), np.zeros(S)
    cmd, pend = np.zeros(S, int), np.zeros(S, int)
    k = 1 - np.exp(-dt / tau)
    L, Cm = np.empty((S, W_)), np.empty((S, W_), int)
    for i in range(W_):
        h = held[:, i]
        zi = z[:, i]
        upd = ~h & np.isfinite(zi)
        level = np.where(upd, level + k * (-np.where(upd, zi, 0.0) - level), level)
        want = np.where(h, cmd, np.where(level > threshold, 1, np.where(level < -threshold, 2, 0)))
        new = want != pend
        pend = np.where(new, want, pend)
        since = np.where(new, 0.0, since + dt)
        inside = np.where(cmd == 1, level < threshold / 2, level > -threshold / 2)
        cmd = np.where((cmd != 0) & ~h & inside, 0, cmd)
        cmd = np.where((cmd == 0) & (want != 0) & (since >= dwell), want, cmd)
        L[:, i], Cm[:, i] = level, cmd
    return L, Cm


# ---------------------------------------------------------------- replay of one person under every condition
def replay_person(person, models, conds, seed, dec_person=None, gates=(REJECT_UV, 0.0)):
    """models: {target: FoldModels}. dec_person: a second Person on the decoder's electrodes (Insight: the decoder
    uses AF3, AF4, T7, T8 while the focus/relax control reads Pz too); the injected signal is the same on shared electrodes.
    Returns {cond: {gate: [per recording dict]}} with decoder outputs, holds, usable flags and alphaZ."""
    dp = dec_person or person
    res = {}
    for ci, cond in enumerate(conds):
        res[cond] = {g: [] for g in gates}
        for ri, (t_ctl, t_dec) in enumerate(zip(person.trials, dp.trials)):
            key = (seed, _stable(person.subject), ri)
            x_ctl = inject(t_ctl.x, person.fs, person.channels, cond, key)
            r_ctl = person.replay(x_ctl, gates=gates)
            if dp is person:
                r_dec = r_ctl
            else:
                idx = [person.channels.index(c) for c in dp.channels]
                r_dec = dp.replay(x_ctl[idx], gates=gates)
                assert np.allclose(r_dec["t"], r_ctl["t"])
            for g in gates:
                d, c = r_dec[g], r_ctl[g]
                rec = {"t": r_ctl["t"], "use_filt": d["use_filt"], "use_clean": d["use_clean"],
                       "alpha_z": np.where(c["use_clean"], alpha_z(c["X_clean"]), np.nan),
                       "alpha_z_filt": np.where(c["use_filt"], alpha_z(c["X_filt"]), np.nan)}
                held = None
                for tgt, fm in models.items():
                    out = fm.predict(d["X_filt"], d["X_clean"], d["face_live"])
                    for dec, (val, conf) in out.items():
                        rec[f"{dec}/{tgt}"] = val
                        rec[f"{dec}/{tgt}/conf"] = conf
                    h = fm.held(d["face_live"])
                    held = h if held is None else (held | h)
                rec["hold"] = held & d["use_clean"]
                if cond == "clean" and g == gates[0]:
                    rec["face_audit"] = r_ctl["face_audit"]
                res[cond][g].append(rec)
    return res


def _stable(s):
    """Deterministic small integer from a string (Python's hash() is salted per process)."""
    return int.from_bytes(s.encode()[-8:].rjust(8, b"\0"), "big") % 2_147_483_647


# ---------------------------------------------------------------- closed-loop simulation of many streams
def pad(arrs, fill):
    n = max(len(a) for a in arrs)
    out = np.full((len(arrs), n), fill, dtype=np.result_type(*[a.dtype for a in arrs], type(fill)))
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


def simulate(recs, targets=("valence", "arousal")):
    """Closed loop for a list of per-recording replay dicts (one condition, one gate). Returns per decoder padded
    arrays: raw output, decoded flag, shown (Kalman) value, world state, parameters, weather; and the focus/relax control."""
    lens = np.array([len(r["t"]) for r in recs])
    valid = np.arange(lens.max())[None] < lens[:, None]
    out = {"len": lens, "valid": valid, "t": pad([r["t"] for r in recs], np.nan)}
    for dec in DECODERS:
        use = pad([r["use_filt"] if dec == "standard_filt" else r["use_clean"] for r in recs], False)
        ok = use & valid
        if dec == "brain":
            ok &= ~pad([r["hold"] for r in recs], False)
        conf = np.mean([pad([r[f"{dec}/{t}/conf"] for r in recs], 1.0) for t in targets], axis=0)
        rr = 0.03 / np.maximum(0.2, conf)
        o = {"ok": ok}
        for t in targets:
            o["raw_" + t] = pad([r[f"{dec}/{t}"] for r in recs], np.nan)
            o["shown_" + t] = kalman(o["raw_" + t], rr, ok)
        if set(targets) == {"valence", "arousal"}:
            cv, ca, w = world(o["shown_valence"], o["shown_arousal"])
            o["cur_valence"], o["cur_arousal"], o["weather"] = cv, ca, w
            o["quadrant"] = mood_weather(cv, ca)
            o.update(world_params(cv, ca))
        out[dec] = o
    held_u = ~pad([r["use_clean"] for r in recs], False)
    held_g = held_u | pad([r["hold"] for r in recs], False)
    az = pad([r["alpha_z"] for r in recs], np.nan)
    azf = pad([r["alpha_z_filt"] for r in recs], np.nan)
    for name, z, h in (("ctrl_guarded", az, held_g), ("ctrl_unguarded", az, held_u),
                       ("ctrl_filt", azf, ~pad([r["use_filt"] for r in recs], False))):
        L, Cm = alpha_control(z, h)
        out[name] = {"level": L, "command": Cm, "held": h}
    return out


# ---------------------------------------------------------------- face share
EMA_TAUS = (2.0, 8.0, 30.0)


def face_history(face, hop=HOP, taus=EMA_TAUS):
    """Causal exponential averages of a recording's face indices (W, K) at several time scales -> (W, K * len)."""
    outs = []
    for tau in taus:
        k = 1 - np.exp(-hop / tau)
        e = np.empty_like(face)
        acc = face[0].copy()
        for i in range(len(face)):
            acc += k * (face[i] - acc)
            e[i] = acc
        outs.append(e)
    return np.hstack(outs)


def face_design(H):
    Z = np.hstack([H, H ** 2])
    mu, sd = Z.mean(0), Z.std(0)
    return np.hstack([(Z - mu) / np.where(sd < 1e-9, 1.0, sd), np.ones((len(Z), 1))])


def cv_r2(Zs, Ys, persons, lam=1.0, grams=None):
    """Out-of-person ridge R^2. Zs, Ys: per recording design (n_r, K) and targets (n_r, T); persons: id per recording;
    grams: optional precomputed Zs[i].T @ Zs[i]. Returns pooled R^2 (T,), per-person (ssres, sstot) arrays (P, T)
    and person ids. The mean in SS_tot is the pooled mean, so the pooled value is the SS_tot-weighted mean of the
    per-person values."""
    persons = np.asarray(persons)
    ids = list(dict.fromkeys(persons))
    if grams is None:
        grams = [Z.T @ Z for Z in Zs]
    G = {p: sum(grams[i] for i in np.flatnonzero(persons == p)) for p in ids}
    H = {p: sum(Zs[i].T @ Ys[i] for i in np.flatnonzero(persons == p)) for p in ids}
    Gt, Ht = sum(G.values()), sum(H.values())
    K = Gt.shape[0]
    reg = lam * np.eye(K)
    reg[-1, -1] = 0.0
    ybar = np.concatenate(Ys).mean(axis=0)
    ssr = np.zeros((len(ids), Ys[0].shape[1]))
    sst = np.zeros_like(ssr)
    for j, p in enumerate(ids):
        B = np.linalg.solve(Gt - G[p] + reg, Ht - H[p])
        for i in np.flatnonzero(persons == p):
            ssr[j] += ((Ys[i] - Zs[i] @ B) ** 2).sum(axis=0)
            sst[j] += ((Ys[i] - ybar) ** 2).sum(axis=0)
    return 1 - ssr.sum(0) / np.maximum(sst.sum(0), 1e-300), ssr, sst, ids


def in_sample_share(face, dec):
    """research/brain_decoder.face_leakage: adjusted R^2 of the output from face indices and squares, in sample."""
    Z = np.hstack([face, face ** 2, np.ones((len(face), 1))])
    B, *_ = np.linalg.lstsq(Z, dec, rcond=None)
    r2 = 1 - np.sum((dec - Z @ B) ** 2) / np.sum((dec - dec.mean()) ** 2)
    n, k = Z.shape
    return float(1 - (1 - r2) * (n - 1) / (n - k))


def world_targets(sim, names):
    """Per recording (n_r, T) matrix of the analysed signals over valid windows, with column names."""
    cols, mats = [], []
    for dec in DECODERS:
        o = sim[dec]
        for t in TARGETS:
            for kind in ("shown", "cur"):
                cols.append(f"{dec}/{kind}_{t}")
                mats.append(o[f"{kind}_{t}"])
        for p in PARAMS:
            cols.append(f"{dec}/{p}")
            mats.append(o[p])
        for k, w in enumerate(WEATHERS):
            cols.append(f"{dec}/weather={w}")
            mats.append((o["weather"] == k).astype(float))
    for c in ("ctrl_guarded", "ctrl_unguarded"):
        cols.append(f"{c}/level")
        mats.append(sim[c]["level"])
    M_ = np.stack(mats, axis=-1)
    return [M_[i, :n] for i, n in enumerate(sim["len"])], cols


def group_r2(r2cols, ssr, sst, cols, prefix):
    """R^2 of the one-hot weather block (multivariate: summed over its columns)."""
    j = [i for i, c in enumerate(cols) if c.startswith(prefix + "/weather=")]
    return 1 - ssr[:, j].sum() / max(sst[:, j].sum(), 1e-300), ssr[:, j].sum(axis=1), sst[:, j].sum(axis=1)


def wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank on paired per-person values, with the matched-pairs rank-biserial r
    (positive: a > b) and the median difference."""
    from scipy.stats import wilcoxon as W
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    nz = d[d != 0]
    if len(nz) == 0:
        return {"n": int(ok.sum()), "p": 1.0, "r_rb": 0.0, "median_diff": 0.0, "a_gt_b": 0}
    from scipy.stats import rankdata
    rk = rankdata(np.abs(nz))
    r = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())
    p = float(W(a[ok], b[ok]).pvalue) if len(nz) > 1 else 1.0
    return {"n": int(ok.sum()), "p": p, "r_rb": r, "median_diff": float(np.median(d)), "a_gt_b": int((d > 0).sum())}


def boot_ci(ssr, sst, rng, n=2000):
    """95 % CI of pooled R^2, resampling people."""
    P = len(ssr)
    v = []
    for _ in range(n):
        i = rng.integers(0, P, P)
        v.append(1 - ssr[i].sum() / max(sst[i].sum(), 1e-300))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def face_share_analysis(sims_clean, persons_of_rec, n_perm, seed):
    """sims_clean: simulate() output on the clean replay (gate = app). Face measure: audit face indices."""
    rng = np.random.default_rng(seed)
    face = sims_clean["face_audit"]
    Ys, cols = world_targets(sims_clean, None)
    Hs = [face_history(f) for f in face]
    Zall = face_design(np.concatenate(Hs))
    cut = np.cumsum([0] + [len(h) for h in Hs])
    Zs = [Zall[cut[i]:cut[i + 1]] for i in range(len(Hs))]
    grams = [Z.T @ Z for Z in Zs]
    r2, ssr, sst, ids = cv_r2(Zs, Ys, persons_of_rec, grams=grams)
    res = {"columns": {}, "people": ids, "n_face_regressors": int(Zs[0].shape[1] - 1)}
    for j, c in enumerate(cols):
        if "/weather=" in c:
            continue
        res["columns"][c] = {"r2": float(r2[j]), "per_person": (1 - ssr[:, j] / np.maximum(sst[:, j], 1e-300)).tolist(),
                             "ci95": boot_ci(ssr[:, j], sst[:, j], rng, 1000)}
    for dec in DECODERS:
        g, a_, b_ = group_r2(r2, ssr, sst, cols, dec)
        res["columns"][f"{dec}/weather"] = {"r2": float(g), "per_person": (1 - a_ / np.maximum(b_, 1e-300)).tolist(),
                                            "ci95": boot_ci(a_, b_, rng, 1000)}
    # per-window decoder output (decoded windows only), same out-of-person measure, and the audit's in-sample measure
    for dec in DECODERS:
        for t in TARGETS:
            o = sims_clean[dec]
            Zw = [Zs[i][o["ok"][i, :n]] for i, n in enumerate(sims_clean["len"])]
            Yw = [o["raw_" + t][i, :n][o["ok"][i, :n]][:, None] for i, n in enumerate(sims_clean["len"])]
            r2w, sr, st, _ = cv_r2(Zw, Yw, persons_of_rec)
            res["columns"][f"{dec}/raw_{t}"] = {"r2": float(r2w[0]), "per_person": (1 - sr[:, 0] / np.maximum(st[:, 0], 1e-300)).tolist(),
                                                "ci95": boot_ci(sr[:, 0], st[:, 0], rng, 1000)}
            fw = np.concatenate([face[i][o["ok"][i, :n]] for i, n in enumerate(sims_clean["len"])])
            res["columns"][f"{dec}/raw_{t}"]["in_sample_unsmoothed"] = in_sample_share(fw, np.concatenate(Yw)[:, 0])
    # permutation null: every recording gets the face series of a random recording of another person
    if n_perm:
        persons = np.asarray(persons_of_rec)
        nrec = len(Ys)
        null = []
        for _ in range(n_perm):
            src = np.empty(nrec, int)
            for i in range(nrec):
                cand = np.flatnonzero(persons != persons[i])
                src[i] = rng.choice(cand)
            same = [len(Zs[src[i]]) == len(Zs[i]) for i in range(nrec)]
            Zp = [Zs[src[i]] if same[i] else np.resize(Zs[src[i]], Zs[i].shape) for i in range(nrec)]
            gp = [grams[src[i]] if same[i] else Zp[i].T @ Zp[i] for i in range(nrec)]
            r2p, sp, tp, _ = cv_r2(Zp, Ys, persons, grams=gp)
            row = {c: float(r2p[j]) for j, c in enumerate(cols) if "/weather=" not in c}
            for dec in DECODERS:
                row[f"{dec}/weather"] = float(group_r2(r2p, sp, tp, cols, dec)[0])
            null.append(row)
        for c, v in res["columns"].items():
            if c in null[0]:
                nv = np.array([r[c] for r in null])
                v["p_perm"] = float((1 + np.sum(nv >= v["r2"])) / (1 + len(nv)))
                v["null_95"] = float(np.percentile(nv, 95))
                v["null_mean"] = float(nv.mean())
    return res


# ---------------------------------------------------------------- injection and focus/relax control metrics
def injection_metrics(sim_c, sim_i, rec_person, ids):
    """Per person, over injected epochs: mean |change| of each decoder's raw output (windows decoded in both
    replays), of the shown value, of the world state and parameters; fraction of time the displayed weather and the
    affect quadrant differ; fraction of windows decoded; signed shift of the shown value."""
    valid = sim_c["valid"] & on_epochs(np.nan_to_num(sim_c["t"]))
    rp = np.asarray(rec_person)
    out = {}
    for dec in DECODERS:
        c, i = sim_c[dec], sim_i[dec]
        m = {}
        for t in TARGETS:
            both = valid & c["ok"] & i["ok"]
            m[f"raw_{t}"] = _per_person(np.abs(i["raw_" + t] - c["raw_" + t]), both, rp, ids)
            m[f"shown_{t}"] = _per_person(np.abs(i["shown_" + t] - c["shown_" + t]), valid, rp, ids)
            m[f"shown_{t}_signed"] = _per_person(i["shown_" + t] - c["shown_" + t], valid, rp, ids)
        for p in PARAMS:
            m[p] = _per_person(np.abs(i[p] - c[p]), valid, rp, ids)
        m["weather_differs"] = _per_person((i["weather"] != c["weather"]).astype(float), valid, rp, ids)
        m["quadrant_differs"] = _per_person((i["quadrant"] != c["quadrant"]).astype(float), valid, rp, ids)
        m["decoded"] = _per_person(i["ok"].astype(float), valid, rp, ids)
        out[dec] = m
    return out


def _per_person(v, mask, rp, ids):
    res = []
    for p in ids:
        sel = mask[rp == p]
        vals = v[rp == p][sel]
        res.append(float(np.mean(vals)) if len(vals) else np.nan)
    return np.array(res)


def command_onsets(cmd):
    """(S, W) bool: a command (focus or relax) starts at this window."""
    prev = np.concatenate([np.zeros((cmd.shape[0], 1), int), cmd[:, :-1]], axis=1)
    return (cmd != 0) & (prev == 0)


def control_metrics(sim_c, sim_i, rec_person, ids, ctrl):
    """Commands per minute of injected-epoch time (focus, relax) in the injected and the clean replay, per person;
    fraction of injected-epoch time the control's command differs from the clean replay; fraction held."""
    valid = sim_c["valid"] & on_epochs(np.nan_to_num(sim_c["t"]))
    rp = np.asarray(rec_person)
    minutes = np.array([valid[rp == p].sum() * HOP / 60.0 for p in ids])
    res = {"minutes": minutes}
    for tag, s in (("inj", sim_i), ("clean", sim_c)):
        cm = s[ctrl]["command"]
        on = command_onsets(cm) & valid
        for k, name in ((1, "focus"), (2, "relax")):
            res[f"{tag}_{name}_per_min"] = np.array([(on & (cm == k))[rp == p].sum() for p in ids]) / np.maximum(minutes, 1e-9)
    res["differs"] = _per_person((sim_i[ctrl]["command"] != sim_c[ctrl]["command"]).astype(float), valid, rp, ids)
    res["held"] = _per_person(sim_i[ctrl]["held"].astype(float), valid, rp, ids)
    res["excess_per_min"] = (res["inj_focus_per_min"] + res["inj_relax_per_min"]
                             - res["clean_focus_per_min"] - res["clean_relax_per_min"])
    # epoch-level hit: does the expected command appear during an injected epoch (positive controls)
    return res


def epoch_hits(sim, rec_person, ids, ctrl, k):
    """Per person, fraction of injected epochs (>= 20 s long) during which command k is on at some point."""
    rp = np.asarray(rec_person)
    cm = sim[ctrl]["command"]
    hits = {p: [] for p in ids}
    for i in range(cm.shape[0]):
        t = sim["t"][i, :sim["len"][i]]
        e = np.floor(t / EPOCH_SEC)
        for ep in np.unique(e[(e % 2) == 1]):
            sel = e == ep
            if sel.sum() * HOP < 20:
                continue
            hits[rp[i]].append(bool(np.any(cm[i, :sim["len"][i]][sel] == k)))
    return np.array([np.mean(hits[p]) if hits[p] else np.nan for p in ids])


# ---------------------------------------------------------------- dataset runner
def run_dataset(name, persons, models, conds, seed, n_perm, dec_persons=None, log=print, jobs=1):
    """persons: list of Person (control electrodes); dec_persons: matching Person list on the decoder's electrodes
    (or None); models: per person {target: FoldModels} (fitted without that person)."""
    t0 = time.time()
    dps = dec_persons or [None] * len(persons)
    if jobs > 1:
        from joblib import Parallel, delayed
        reps = Parallel(n_jobs=jobs)(delayed(replay_person)(p, m, conds, seed, dp) for p, m, dp in zip(persons, models, dps))
    else:
        reps = []
        for p, m, dp in zip(persons, models, dps):
            reps.append(replay_person(p, m, conds, seed, dp))
            log(f"  {name}: {p.subject} replayed ({time.time() - t0:.0f} s)")
    log(f"  {name}: replayed ({time.time() - t0:.0f} s)")
    ids = [p.subject for p in persons]
    rec_person = [p.subject for p, r in zip(persons, reps) for _ in r["clean"][REJECT_UV]]
    labels = {t: [float(tr.labels.get(t, np.nan)) for p in persons for tr in p.trials] for t in TARGETS}
    sims = {g: simulate([rr for r in reps for rr in r["clean"][g]]) for g in (REJECT_UV, 0.0)}
    clean = sims[REJECT_UV]
    clean["face_audit"] = [rr["face_audit"] for r in reps for rr in r["clean"][REJECT_UV]]
    log(f"  {name}: closed loop simulated ({time.time() - t0:.0f} s)")
    res = {"n_people": len(ids), "n_recordings": len(rec_person),
           "minutes_replayed": float(sum(clean["len"]) * HOP / 60.0)}
    # decoder accuracy on the replay (2 s windows every 0.25 s; label of the recording), clean, app gate
    acc = {}
    for dec in DECODERS:
        for t in TARGETS:
            y = np.concatenate([np.full(n, labels[t][i]) for i, n in enumerate(clean["len"])])
            if np.all(np.isnan(y)):
                continue
            raw = np.concatenate([clean[dec]["raw_" + t][i, :n] for i, n in enumerate(clean["len"])])
            ok = np.concatenate([clean[dec]["ok"][i, :n] for i, n in enumerate(clean["len"])]) & ~np.isnan(y)
            pred = np.where(raw >= 0, 1, -1)
            acc[f"{dec}/{t}"] = float(np.mean([np.mean(pred[ok & (y == c)] == c) for c in (-1, 1)]))
            acc[f"{dec}/{t}/decoded_fraction"] = float(np.mean(ok[~np.isnan(y)]))
    res["replay_balanced_accuracy"] = acc
    res["face_share"] = face_share_analysis(clean, rec_person, n_perm, seed)
    log(f"  {name}: face share done ({time.time() - t0:.0f} s)")
    fs_ = res["face_share"]["columns"]
    comp = {}
    for col in ["raw_valence", "raw_arousal", "shown_valence", "shown_arousal", "cur_valence", "cur_arousal"] + PARAMS + ["weather"]:
        for ref in ("standard_filt", "standard_clean"):
            if f"{ref}/{col}" in fs_:
                comp[f"{ref}_vs_brain/{col}"] = wilcoxon(fs_[f"{ref}/{col}"]["per_person"], fs_[f"brain/{col}"]["per_person"])
    comp["ctrl_unguarded_vs_guarded/level"] = wilcoxon(fs_["ctrl_unguarded/level"]["per_person"], fs_["ctrl_guarded/level"]["per_person"])
    res["face_share_tests"] = comp
    # injection
    inj = {}
    for gate, gname in ((REJECT_UV, "app"), (0.0, "no_gate")):
        base = sims[gate]
        for cond in conds:
            if cond == "clean":
                continue
            sim = simulate([rr for r in reps for rr in r[cond][gate]])
            m = injection_metrics(base, sim, rec_person, ids)
            ctl = {c: control_metrics(base, sim, rec_person, ids, c)
                   for c in ("ctrl_guarded", "ctrl_unguarded", "ctrl_filt")}
            entry = {"decoders": {d: {k: _summ(v) for k, v in mm.items()} for d, mm in m.items()},
                     "control": {c: {k: _summ(v) for k, v in cc.items()} for c, cc in ctl.items()}}
            tests = {}
            for key in ["raw_valence", "raw_arousal", "shown_valence", "shown_arousal"] + PARAMS + ["weather_differs", "quadrant_differs"]:
                for ref in ("standard_filt", "standard_clean"):
                    tests[f"{ref}_vs_brain/{key}"] = wilcoxon(m[ref][key], m["brain"][key])
            tests["control_unguarded_vs_guarded/excess_per_min"] = wilcoxon(ctl["ctrl_unguarded"]["excess_per_min"], ctl["ctrl_guarded"]["excess_per_min"])
            tests["control_unguarded_vs_guarded/differs"] = wilcoxon(ctl["ctrl_unguarded"]["differs"], ctl["ctrl_guarded"]["differs"])
            tests["control_guarded_excess_vs_zero"] = wilcoxon(ctl["ctrl_guarded"]["excess_per_min"], np.zeros(len(ids)))
            tests["control_unguarded_excess_vs_zero"] = wilcoxon(ctl["ctrl_unguarded"]["excess_per_min"], np.zeros(len(ids)))
            entry["tests"] = tests
            if cond in ("alpha_up", "alpha_down", "alpha_up_blink150"):
                k = 2 if cond != "alpha_down" else 1
                entry["control_epoch_hits"] = {c: _summ(epoch_hits(sim, rec_person, ids, c, k))
                                               for c in ("ctrl_guarded", "ctrl_unguarded", "ctrl_filt")}
                entry["control_epoch_hits_clean"] = {c: _summ(epoch_hits(base, rec_person, ids, c, k))
                                                     for c in ("ctrl_guarded", "ctrl_unguarded", "ctrl_filt")}
            inj.setdefault(gname, {})[cond] = entry
    res["injection"] = inj
    # spontaneous focus/relax commands on the clean replay (whole recordings)
    sp = {}
    for c in ("ctrl_guarded", "ctrl_unguarded", "ctrl_filt"):
        cm = clean[c]["command"]
        on = command_onsets(cm) & clean["valid"]
        rp = np.asarray(rec_person)
        mins = np.array([clean["valid"][rp == p].sum() * HOP / 60 for p in ids])
        sp[c] = {"commands_per_min": _summ(np.array([on[rp == p].sum() for p in ids]) / mins),
                 "time_commanded": _summ(_per_person((cm != 0).astype(float), clean["valid"], rp, ids))}
    res["control_clean_replay"] = sp
    log(f"  {name}: done ({time.time() - t0:.0f} s)")
    return res


def _summ(v):
    v = np.asarray(v, float)
    f = v[np.isfinite(v)]
    return {"mean": float(f.mean()) if len(f) else None, "median": float(np.median(f)) if len(f) else None,
            "q25": float(np.percentile(f, 25)) if len(f) else None, "q75": float(np.percentile(f, 75)) if len(f) else None,
            "per_person": [None if not np.isfinite(x) else float(x) for x in v]}


# ---------------------------------------------------------------- checks against the published pipeline
def check_against_cache(persons, cache_path, log=print):
    """Max |difference| between this pipeline's training rows and research/brain_decoder.py's cache."""
    if not os.path.exists(cache_path):
        return None
    c = np.load(cache_path)
    out = {}
    for p in persons:
        sel = c["subject"] == p.subject
        if sel.sum() != len(p.train["rec"]):
            out[p.subject] = "row count differs"
            continue
        for k in ("eeg_filtered", "eeg_cleaned", "face", "face_live"):
            mine = {"eeg_filtered": "X_filt", "eeg_cleaned": "X_clean", "face": "face", "face_live": "face_live"}[k]
            out.setdefault(k, 0.0)
            out[k] = max(out[k], float(np.abs(c[k][sel] - p.train[mine]).max()))
    log(f"  pipeline vs {cache_path}: {out}")
    return out


def loso_published(persons, folds, log=print):
    """LOSO balanced accuracy and face share at the training hop (2 s windows every 2 s), to compare with
    docs/RESULTS.md 3c (research/brain_decoder.py on the same windows)."""
    res = {}
    for t in TARGETS:
        preds = {d: [] for d in DECODERS}
        ys, faces = [], []
        for p in persons:
            te = p.train
            out = folds[p.subject][t].predict(te["X_filt"], te["X_clean"], te["face_live"])
            for d in DECODERS:
                preds[d].append(out[d][0])
            ys.append(te["y"][t])
            faces.append(te["face"])
        y = np.concatenate(ys)
        Fa = np.concatenate(faces)
        held = np.concatenate([folds[p.subject][t].held(p.train["face_live"]) for p in persons])
        for d in DECODERS:
            v = np.concatenate(preds[d])
            pr = np.where(v >= 0, 1, -1)
            res[f"{d}/{t}"] = {"bacc": float(np.mean([np.mean(pr[y == c] == c) for c in (-1, 1)])),
                               "face_r2": in_sample_share(Fa, v)}
        res[f"brain/{t}"]["held_fraction"] = float(held.mean())
    log("  LOSO at the training hop: " + ", ".join(f"{k} {v['bacc']:.3f} / {v['face_r2']:.2f}" for k, v in res.items()))
    return res


# ---------------------------------------------------------------- main
def build_persons(trials, det, log=print, jobs=1):
    by = {}
    for t in trials:
        by.setdefault(t.subject, []).append(t)
    keys = sorted(by)
    if jobs > 1:
        from joblib import Parallel, delayed
        return Parallel(n_jobs=jobs)(delayed(Person)(by[k], det) for k in keys)
    out = []
    for k in keys:
        out.append(Person(by[k], det))
    return out


def run(a, log=print):
    warnings.filterwarnings("ignore")
    det = A.load()
    conds = list(CONDITIONS) if not a.conds else ["clean"] + [c for c in a.conds.split(",") if c and c != "clean"]
    out = {"settings": {"hop_s": HOP, "train_hop_s": TRAIN_HOP, "reject_uv": REJECT_UV, "kalman_q": KALMAN_Q,
                        "director_rate": DIRECTOR_RATE, "slew": [SLEW_RATE, SLEW_TAU], "fps": FPS, "mood": MOOD,
                        "weather_dwell_s": WEATHER_DWELL, "control": [CTRL_THRESHOLD, CTRL_DWELL, CTRL_TAU],
                        "epoch_s": EPOCH_SEC, "conditions": {k: list(v) for k, v in CONDITIONS.items() if k in conds},
                        "face_history_taus_s": list(EMA_TAUS), "seed": a.seed, "n_perm": a.perms}}
    if a.gameemo:
        log("GAMEEMO (14 channels): building people")
        trials = D.load_gameemo_raw(a.gameemo)
        if a.max_people:
            keep = sorted({t.subject for t in trials})[:a.max_people]
            trials = [t for t in trials if t.subject in keep]
        persons = build_persons(trials, det, log, a.jobs)
        out["gameemo_pipeline_check"] = check_against_cache(persons, os.path.join(a.cache_root, "brain_gameemo_v2", "cache.npz"), log)
        folds = {}
        for p in persons:
            tr = stack_train(persons, exclude=p.subject)
            folds[p.subject] = {t: FoldModels(tr, t) for t in TARGETS}
        log("  LOSO models fitted")
        out["gameemo_loso_training_hop"] = loso_published(persons, folds, log)
        out["gameemo_hold_z"] = float(np.median([f["valence"].hold_z for f in folds.values()]))
        out["gameemo"] = run_dataset("GAMEEMO", persons, [folds[p.subject] for p in persons], conds, a.seed, a.perms,
                                     log=log, jobs=a.jobs)
        del persons, folds
    if a.stress and a.gameemo:
        log("Insight stress data (5 channels; decoders on AF3, AF4, T7, T8, trained on all GAMEEMO people)")
        g4 = [t for t in D.to_device(D.load_gameemo_raw(a.gameemo), "emotiv_insight")]
        if a.max_people:
            keep = sorted({t.subject for t in g4})[:a.max_people]
            g4 = [t for t in g4 if t.subject in keep]
        gp = build_persons(g4, det, log, a.jobs)
        assert sorted(gp[0].channels) == sorted(INSIGHT_DEC), gp[0].channels
        tr = stack_train(gp)
        fms = {t: FoldModels(tr, t) for t in TARGETS}
        out["insight_hold_z"] = fms["valence"].hold_z
        st = D.load_stress_insight(a.stress)
        if a.max_people:
            keep = sorted({t.subject for t in st})[:a.max_people]
            st = [t for t in st if t.subject in keep]
        sp = build_persons(st, det, log, a.jobs)
        st4 = [D.Trial(t.subject, t.trial_id, t.x[[t.channels.index(c) for c in INSIGHT_DEC]], t.fs, list(INSIGHT_DEC),
                       dict(t.labels), t.filtered, t.dataset, t.group) for t in st]
        sp4 = build_persons(st4, det, log, a.jobs)
        out["stress"] = run_dataset("stress", sp, [fms] * len(sp), conds, a.seed, a.perms, dec_persons=sp4, log=log, jobs=a.jobs)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gameemo", default=None)
    p.add_argument("--stress", default=None)
    p.add_argument("--out", default="results/closed_loop")
    p.add_argument("--cache-root", default=os.path.join(ROOT, "results"), help="where results/brain_gameemo_v2/cache.npz is")
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--max-people", type=int, default=0)
    p.add_argument("--conds", default="")
    p.add_argument("--figure", action="store_true", help="only redraw the figure from closed_loop.json")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "closed_loop.json")
    if not a.figure:
        res = run(a)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1, default=_json_default)
    figure(path, a.out)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


COL = {"standard_filt": "#1baf7a", "standard_clean": "#eb6834", "brain": "#2a78d6"}   # dataviz reference slots 3, 2, 1
NAME = {"standard_filt": "Standard, filter only", "standard_clean": "Standard, cleaned", "brain": "Emotif brain-only"}
COND_LABEL = {"blink100": "blinks\n100", "blink150": "blinks\n150", "blink200": "blinks\n200", "jaw10": "jaw\n10",
              "jaw20": "jaw\n20", "jaw40": "jaw\n40", "jawwide20": "jaw 13-45\n20", "alpha_up": "alpha\nx2",
              "alpha_down": "alpha\nx0.5", "alpha_up_blink150": "alpha x2\n+ blinks"}


def figure(path, out):
    """Row per dataset: face share of the world (GAMEEMO; per-window audit measure for the Insight data, where the
    out-of-person world regression does not generalize), world change under injected artifacts vs genuine alpha
    changes, and focus/relax commands added by each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = json.load(open(path))
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.titlesize": 8, "axes.titleweight": "bold"})
    sets = [k for k in ("gameemo", "stress") if k in R]
    fig, axes = plt.subplots(len(sets), 3, figsize=(13, 3.9 * len(sets)), dpi=170, squeeze=False,
                             gridspec_kw={"width_ratios": [1.1, 1.35, 1.0]})
    rows = [("raw_valence", "Decoder output, valence"), ("raw_arousal", "Decoder output, arousal"),
            ("shown_valence", "Valence the app shows"), ("shown_arousal", "Arousal the app shows"),
            ("cloud", "Cloud cover"), ("storm", "Storm"), ("fog", "Fog density"), ("saturation", "Colour saturation"),
            ("weather", "Weather (5 kinds)")]
    tag = {"gameemo": "GAMEEMO (28 people, 14 ch, LOSO)", "stress": "Insight stress (17 people, GAMEEMO models)"}
    letters = iter("abcdef")

    def style(ax, axis):
        ax.grid(axis=axis, color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for r_, ds in enumerate(sets):
        D_ = R[ds]
        fs_ = D_["face_share"]["columns"]
        ax = axes[r_, 0]
        h = 0.26
        world_ok = ds == "gameemo"
        use = rows if world_ok else rows[:2]
        for k, dec in enumerate(DECODERS):
            ys = np.arange(len(use))[::-1] + (1 - k) * h
            if world_ok:
                vals = [fs_[f"{dec}/{c}"]["r2"] for c, _ in use]
            else:
                vals = [fs_[f"{dec}/{c}"]["in_sample_unsmoothed"] for c, _ in use]
            ax.barh(ys, np.clip(vals, 0, None), height=h - 0.03, color=COL[dec], label=NAME[dec])
            for y, v, c in zip(ys, vals, use):
                pv = fs_[f"{dec}/{c[0]}"].get("p_perm")
                txt = f"{v:.2f}" if v > 0 else f"<0 ({v:.2f})"
                ax.text(max(v, 0) + 0.008, y, txt + ("*" if (pv is not None and pv < 0.05 and v > 0) else ""),
                        va="center", fontsize=5.6, color=muted)
        ax.set_yticks(np.arange(len(use))[::-1], [n for _, n in use])
        ax.set_xlim(0, 0.62)
        if world_ok:
            ax.set_xlabel("share explained by the face (out-of-person R²; * permutation p < 0.05)")
            ax.set_title(f"({next(letters)}) {tag[ds]}: face share of the world", loc="left")
        else:
            ax.set_xlabel("share explained by the face (audit measure, per 2 s window, in sample)")
            ax.set_title(f"({next(letters)}) Insight stress, 17 people: face share of the output", loc="left")
        style(ax, "x")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        inj = D_["injection"]["app"]
        conds = [c for c in COND_LABEL if c in inj]
        art = [i for i, c in enumerate(conds) if not c.startswith("alpha")]
        x = np.arange(len(conds))
        ax = axes[r_, 1]
        for k, dec in enumerate(DECODERS):
            v = [0.5 * (inj[c]["decoders"][dec]["shown_valence"]["mean"] + inj[c]["decoders"][dec]["shown_arousal"]["mean"]) for c in conds]
            wv = [100 * inj[c]["decoders"][dec]["weather_differs"]["mean"] for c in conds]
            ax.bar(x + (k - 1) * 0.27, v, width=0.25, color=COL[dec], label=NAME[dec])
            for xi, vi, wi in zip(x + (k - 1) * 0.27, v, wv):
                ax.text(xi, vi + 0.004, f"{wi:.0f}", ha="center", va="bottom", fontsize=4.8, color=muted)
        ax.axvline(max(art) + 0.5, color=muted, lw=0.6, ls=":")
        ax.set_xticks(x, [COND_LABEL[c] for c in conds], fontsize=6)
        ax.set_ylabel("shown state moved, mean |change|")
        ax.set_xlabel("injected: face artifacts (uV)  |  genuine alpha change (right of the line)")
        ax.set_title(f"({next(letters)}) World vs same recording without injection (numbers: % time weather differs)", loc="left")
        style(ax, "y")
        ax = axes[r_, 2]
        for k, (c, lab, colr) in enumerate((("ctrl_unguarded", "Without face hold", "#9a9993"),
                                            ("ctrl_guarded", "Emotif (face hold)", "#2a78d6"))):
            v = [inj[cn]["control"][c]["excess_per_min"]["mean"] for cn in conds]
            ax.bar(x + (k - 0.5) * 0.38, v, width=0.36, color=colr, label=lab)
        ax.axhline(0, color=muted, lw=0.6)
        ax.axvline(max(art) + 0.5, color=muted, lw=0.6, ls=":")
        ax.set_xticks(x, [COND_LABEL[c] for c in conds], fontsize=6)
        ax.set_ylabel("commands per minute added (injected - clean replay)")
        sp = D_["control_clean_replay"]["ctrl_guarded"]["commands_per_min"]["mean"]
        ax.set_title(f"({next(letters)}) Focus/relax control (clean replay {sp:.1f} commands/min)", loc="left")
        style(ax, "y")
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_closed_loop.png"))
    fig.savefig(os.path.join(out, "fig_closed_loop.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
