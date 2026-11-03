"""Face-subspace erasure: make EEG features carry no linear trace of facial activity.

FaceEraser is LEACE (least-squares concept erasure; Belrose et al. 2023, NeurIPS) applied to EEG artifacts. Given
features X and a concept Z (here: per-window eye and muscle indices), it finds the affine map x -> mu + P (x - mu)
that

  * leaves the features with zero cross-covariance with every concept column (so no linear function of the erased
    features can predict any linear combination of the face indices on the training data), and
  * moves the features as little as possible in the Mahalanobis metric of X (the smallest edit that achieves it).

With W the whitening of X (cov S = W^-T W^-1 on its support) and U an orthonormal basis of the whitened
cross-covariance W^T Cov(X, Z):  P = I - W^+ U U^T W^T.

At run time the map needs no face measurement: it is a fixed matrix, and in front of a linear (or MLP) layer it folds
into that layer's weights (W1' = P^T W1, b1' = b1 + W1^T (mu - P mu)), so the deployed decoder costs nothing extra.
"""
import numpy as np


def face_concept(face_region, kind="region", n_regions=17):
    """Concept matrix from face indices pooled into regions: (n, R*3) in, (n, k) out.
    kind = "global": scalp mean of the 3 indices over observed regions + squares (6 columns);
    "region": all 51 pooled indices + squares (102 columns; unobserved regions are constant and drop out)."""
    F = np.asarray(face_region, dtype=np.float64).reshape(len(face_region), n_regions, 3)
    if kind == "global":
        obs = np.any(F != 0, axis=2, keepdims=True)
        g = (F * obs).sum(axis=1) / np.maximum(obs.sum(axis=1), 1)
        return np.hstack([g, g ** 2])
    if kind == "region":
        f = F.reshape(len(F), -1)
        return np.hstack([f, f ** 2])
    raise ValueError(kind)


class FaceEraser:
    def __init__(self, tol=1e-9, rank=None):
        self.tol = tol
        self.max_rank = rank          # erase only the top-k face directions (None = all: the full guarantee)

    def fit(self, X, Z):
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        n, dim = X.shape
        self.mu = X.mean(axis=0)
        Xc, Zc = X - self.mu, Z - Z.mean(axis=0)
        S = Xc.T @ Xc / n
        ev, V = np.linalg.eigh(S)
        keep = ev > max(ev.max(), 0.0) * self.tol
        W = V[:, keep] / np.sqrt(ev[keep])          # whitened coordinates: W^T (x - mu)
        Winv = V[:, keep] * np.sqrt(ev[keep])       # back to feature space
        Mw = W.T @ (Xc.T @ Zc / n)
        U, s, _ = np.linalg.svd(Mw, full_matrices=False)
        r = s > (s.max() * 1e-8 if s.size and s.max() > 0 else np.inf)
        if self.max_rank is not None:
            r &= np.arange(len(r)) < self.max_rank
        U = U[:, r]
        self.rank = int(r.sum())
        self.P = np.eye(dim) - Winv @ U @ U.T @ W.T
        return self

    def transform(self, X):
        return self.mu + (np.asarray(X, dtype=np.float64) - self.mu) @ self.P.T

    def affine(self):
        """(A, b) with transform(x) = A x + b."""
        return self.P, self.mu - self.P @ self.mu
