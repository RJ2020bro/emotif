"""Artifact Subspace Reconstruction (after Mullen et al. 2015, IEEE TBME; Kothe & Jung 2016).

Calibration on (mostly) clean, high-pass-filtered data X (C x N):
  C_cal  = geometric median of 0.5 s window covariances (robust to occasional blinks), plus a small ridge
  V      = eigenvectors of C_cal (ascending)
  rms_i  = RMS of component V_i^T x in 0.5 s windows, step 0.25 s
  thr_i  = max(median(rms_i) + k * 1.4826 * MAD(rms_i), 2.5 * median(rms_i))
  T      = diag(thr) V^T

Processing in blocks of 0.125 s, each using the covariance of a 0.5 s window centred on the block
(so output lags input by (W + S) / 2 samples in streaming):
  D, U   = eig(cov(window)), ascending
  keep_j = D_j < ||T u_j||^2, and the smallest  C - floor(0.66 C)  components are always kept
  R      = I if all kept, else  C_cal U_k (U_k^T C_cal U_k)^-1 U_k^T
  out    = raised-cosine blend from the previous block's R to this block's R
  (if R would increase the block's energy, the block passes through unchanged)

R is the least-squares reconstruction of the full signal from the components that look like the
calibration data. It is invariant to eigenvector signs and to rotations inside the kept subspace, so this
module (numpy eigh) and app/src/asr.ts (Jacobi) agree to rounding error.
"""
import numpy as np

WIN_SEC = 0.5
STEP_SEC = 0.125
CAL_STEP_SEC = 0.25
MAX_DROP = 0.66
DEFAULT_K = 20.0
MIN_CAL_SEC = 10.0
RIDGE = 1e-3          # calibration covariance += RIDGE * mean eigenvalue * I
MIN_RATIO = 2.5       # never repair a component below 2.5x its typical baseline amplitude: genuine state
                      # changes (alpha can double with relaxation) must survive; blinks and EMG are far larger


def _sizes(fs):
    W = max(4, int(round(WIN_SEC * fs)))
    S = max(1, int(round(STEP_SEC * fs)))
    L = (W - S) // 2
    return W, S, L


def min_keep(C):
    return max(1, C - int(np.floor(MAX_DROP * C)))


def geometric_median(mats, iters=100, tol=1e-12):
    """Weiszfeld iteration in Frobenius norm. A convex combination of the inputs, so a geometric median
    of covariance matrices is itself positive semi-definite (an element-wise median is not)."""
    A = np.asarray(mats, dtype=np.float64)
    y = A.mean(axis=0)
    for _ in range(iters):
        d = np.sqrt(np.sum((A - y) ** 2, axis=(1, 2)))
        w = 1.0 / np.maximum(d, 1e-12 * max(1.0, float(np.sqrt(np.sum(y ** 2)))))
        y_new = np.tensordot(w, A, axes=1) / w.sum()
        if np.sqrt(np.sum((y_new - y) ** 2)) <= tol * max(1e-300, np.sqrt(np.sum(y ** 2))):
            y = y_new
            break
        y = y_new
    return y


def calibrate(X, fs, k=DEFAULT_K):
    """X: (C, N) filtered calibration data. Returns a state dict (all numpy arrays)."""
    X = np.asarray(X, dtype=np.float64)
    C, N = X.shape
    W, _, _ = _sizes(fs)
    if N < MIN_CAL_SEC * fs:
        raise ValueError(f"ASR calibration needs at least {MIN_CAL_SEC:.0f} s of data")
    nwin = N // W
    covs = np.empty((nwin, C, C))
    for i in range(nwin):
        seg = X[:, i * W:(i + 1) * W]
        covs[i] = seg @ seg.T / W
    Ccal = geometric_median(covs)
    Ccal = (Ccal + Ccal.T) / 2
    Ccal = Ccal + RIDGE * np.trace(Ccal) / C * np.eye(C)   # rank-deficient data (average reference, bridges)
    d, V = np.linalg.eigh(Ccal)
    Y = V.T @ X
    step = max(1, int(round(CAL_STEP_SEC * fs)))
    starts = range(0, N - W + 1, step)
    rms = np.array([np.sqrt(np.mean(Y[:, s:s + W] ** 2, axis=1)) for s in starts]).T   # (C, nwin)
    mu = np.median(rms, axis=1)
    sig = 1.4826 * np.median(np.abs(rms - mu[:, None]), axis=1)
    sig = np.maximum(sig, 1e-6 * np.maximum(mu, 1e-12))
    thr = np.maximum(mu + k * sig, MIN_RATIO * mu)
    return {"C": Ccal, "V": V, "T": thr[:, None] * V.T, "k": float(k), "fs": float(fs), "n_channels": C}


def reconstruction(state, cov):
    """Reconstruction matrix R (C x C) for one block covariance. Also returns the number of removed components."""
    Ccal, T = state["C"], state["T"]
    C = Ccal.shape[0]
    D, U = np.linalg.eigh((cov + cov.T) / 2)
    thr = np.sum((T @ U) ** 2, axis=0)
    keep = D < thr
    keep[:min_keep(C)] = True
    if keep.all():
        return np.eye(C), 0
    Uk = U[:, keep]
    B = Uk.T @ Ccal @ Uk
    R = Ccal @ Uk @ np.linalg.solve(B, Uk.T)
    # repair must not add energy: fall back to passing the block through unchanged
    if np.trace(R @ cov @ R.T) > np.trace(cov):
        return np.eye(C), 0
    return R, int((~keep).sum())


def _blend(S):
    j = np.arange(1, S + 1)
    return (1 - np.cos(np.pi * j / S)) / 2


def clean(X, state, fs):
    """Offline cleaning, aligned with the input (no delay). Identical arithmetic to ASRStream for every
    block whose look-ahead window lies inside the data. Returns (Y, removed_per_block)."""
    X = np.asarray(X, dtype=np.float64)
    C, N = X.shape
    W, S, L = _sizes(fs)
    Y = np.empty_like(X)
    Rprev = np.eye(C)
    removed = []
    for b0 in range(0, N, S):
        b1 = min(N, b0 + S)
        we = min(N, b0 + S + L)
        ws = max(0, we - W)
        seg = X[:, ws:we]
        R, nrem = reconstruction(state, seg @ seg.T / seg.shape[1])
        removed.append(nrem)
        w = _blend(S)[: b1 - b0]
        blk = X[:, b0:b1]
        Y[:, b0:b1] = w * (R @ blk) + (1 - w) * (Rprev @ blk)
        Rprev = R
    return Y, np.array(removed)


class ASRStream:
    """Causal streaming version: push raw (filtered) rows, receive cleaned rows delayed by S + L samples."""

    def __init__(self, state, fs):
        self.state = state
        self.W, self.S, self.L = _sizes(fs)
        self.C = state["n_channels"]
        self.buf = np.zeros((self.C, 0))
        self.offset = 0          # absolute index of buf[:, 0]
        self.next_block = 0      # absolute start of the next block to emit
        self.Rprev = np.eye(self.C)
        self.removed = 0

    def push(self, rows):
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, self.C).T
        self.buf = np.concatenate([self.buf, rows], axis=1)
        total = self.offset + self.buf.shape[1]
        out = []
        while self.next_block + self.S + self.L <= total:
            b0 = self.next_block
            we = b0 + self.S + self.L
            ws = max(0, we - self.W)
            seg = self.buf[:, ws - self.offset: we - self.offset]
            R, self.removed = reconstruction(self.state, seg @ seg.T / seg.shape[1])
            blk = self.buf[:, b0 - self.offset: b0 - self.offset + self.S]
            w = _blend(self.S)
            out.append(w * (R @ blk) + (1 - w) * (self.Rprev @ blk))
            self.Rprev = R
            self.next_block += self.S
        keep_from = max(0, self.next_block + self.S + self.L - self.W) - self.offset
        if keep_from > 0:
            self.buf = self.buf[:, keep_from:]
            self.offset += keep_from
        return np.concatenate(out, axis=1).T if out else np.zeros((0, self.C))

    @property
    def delay(self):
        return self.S + self.L
