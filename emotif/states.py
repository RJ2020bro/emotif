"""Personal state decoder, Python twin of app/src/states.ts (same maths; tests/test_parity.py checks agreement).

Per person: brain-only residual (region features minus their ridge prediction from the pooled face indices and their
squares) -> standardize -> shrinkage LDA with the total covariance. Each state is scored by its one-vs-rest AUC on
held-out calibration blocks (leave one block out, block-mean probabilities) and tested against permutations of the
state labels between blocks. A face-only reference (the same classifier on the face indices alone) audits how much
the face could have done.

Used by research/personal_states.py on recorded calibration sessions.
"""
import numpy as np

from . import face as FACE
from . import montage as M

SHRINK, RIDGE, ALPHA = 0.5, 1.0, 0.05
N_FEATURES = M.N_FEATURES


def face_design(F):
    F = np.asarray(F, dtype=np.float64)
    return np.hstack([F, F ** 2])


def prepare(X, F, remove_face=True):
    """Label-free part fitted on training windows. Returns a dict and the standardized training rows."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    Q = face_design(F)
    qmu, qsd = Q.mean(0), Q.std(0)
    qsd = np.where(qsd < 1e-9, 1.0, qsd)
    Z = (Q - qmu) / qsd
    P = Z.shape[1] + 1
    if remove_face:
        Z1 = np.hstack([Z, np.ones((n, 1))])
        A = Z1.T @ Z1
        A[np.arange(P - 1), np.arange(P - 1)] += RIDGE
        A[P - 1, P - 1] += 1e-9
        B = np.linalg.solve(A, Z1.T @ X)
    else:
        B = np.zeros((P, X.shape[1]))
        B[-1] = X.mean(0)
    R = X - Z @ B[:-1] - B[-1]
    rmu, rsd = R.mean(0), R.std(0)
    idx = np.flatnonzero(rsd > 1e-9)
    if not len(idx):
        raise ValueError("no usable features")
    S = (R[:, idx] - rmu[idx]) / rsd[idx]
    C = (1 - SHRINK) * (S.T @ S) / n + SHRINK * np.eye(len(idx))
    mx = np.sort(np.abs(Z[:, :FACE.N_REGION_FACE]).max(axis=1))
    hold_z = max(3.0, float(mx[min(len(mx) - 1, int(np.floor(0.995 * len(mx))))]))
    return {"qmu": qmu, "qsd": qsd, "B": B, "rmu": rmu, "rsd": rsd, "idx": idx, "C": C, "hold_z": hold_z}, S


def standardize(p, X, F):
    Z = (face_design(F) - p["qmu"]) / p["qsd"]
    R = np.asarray(X, dtype=np.float64) - Z @ p["B"][:-1] - p["B"][-1]
    return (R[:, p["idx"]] - p["rmu"][p["idx"]]) / p["rsd"][p["idx"]]


def lda(p, S, y, K):
    means = np.array([S[y == k].mean(0) for k in range(K)])
    W = np.linalg.solve(p["C"], means.T)            # D x K
    b = -0.5 * np.einsum("kd,dk->k", means, W)
    return W, b


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_layer(X, F, y, K):
    """One softmax layer over [features, standardized face design] (the app's head input), plus face statistics."""
    p, S = prepare(X, F, True)
    W, b0 = lda(p, S, np.asarray(y), K)
    idx = p["idx"]
    Wx = np.zeros((N_FEATURES, K))
    Wx[idx] = W / p["rsd"][idx, None]
    Wq = -p["B"][:-1][:, idx] @ Wx[idx]
    b = b0 - (p["B"][-1][idx] + p["rmu"][idx]) @ Wx[idx]
    return {"W": np.vstack([Wx, Wq]), "b": b}, {"mu": p["qmu"], "sd": p["qsd"], "hold_z": p["hold_z"]}


def state_input(X, F, face):
    return np.hstack([np.asarray(X, dtype=np.float64), (face_design(F) - face["mu"]) / face["sd"]])


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def _folds(X, F, blocks, kind):
    if kind == "face":
        Q = face_design(F)
        Xk = np.zeros((len(X), N_FEATURES))
        Xk[:, :Q.shape[1]] = Q[:, :N_FEATURES]
        Fk = np.zeros_like(np.asarray(F, dtype=np.float64))
    else:
        Xk, Fk = np.asarray(X, dtype=np.float64), np.asarray(F, dtype=np.float64)
    out = []
    for blk in np.unique(blocks):
        te, tr = blocks == blk, blocks != blk
        p, S = prepare(Xk[tr], Fk[tr], kind == "brain")
        out.append((blk, p, S, np.flatnonzero(tr), standardize(p, Xk[te], Fk[te])))
    return out


def _evaluate(folds, block_label, blocks, K):
    rows = []
    for blk, p, S, tr, T in folds:
        y = np.array([block_label[b] for b in blocks[tr]])
        if len(np.unique(y)) < K:
            continue
        W, b = lda(p, S, y, K)
        rows.append((block_label[blk], softmax(T @ W + b).mean(0)))
    ys = np.array([r[0] for r in rows])
    P = np.array([r[1] for r in rows])
    aucs = [auc(P[ys == k, k], P[ys != k, k]) for k in range(K)]
    rec = [float((P[ys == k].argmax(1) == k).mean()) for k in range(K) if (ys == k).any()]
    return np.array(aucs), float(np.mean(rec)) if rec else float("nan")


def calibrate(X, F, y, blocks, states, perms=199, seed=1):
    """Fit, check every state on held-out blocks, permutation-test. Returns a dict like the app's PersonalStates."""
    y, blocks = np.asarray(y), np.asarray(blocks)
    K = len(states)
    per = [len(np.unique(blocks[y == k])) for k in range(K)]
    if min(per) < 2:
        raise ValueError("each state needs at least two calibration blocks")
    ub = np.unique(blocks)
    truth = {b: int(y[blocks == b][0]) for b in ub}
    fb, ff = _folds(X, F, blocks, "brain"), _folds(X, F, blocks, "face")
    real_auc, real_b = _evaluate(fb, truth, blocks, K)
    _, face_b = _evaluate(ff, truth, blocks, K)
    rng = np.random.default_rng(seed)
    ge_auc, ge_b, ge_f = np.zeros(K), 0, 0
    lab = np.array([truth[b] for b in ub])
    for _ in range(perms):
        m = dict(zip(ub, rng.permutation(lab)))
        a_, b_ = _evaluate(fb, m, blocks, K)
        ge_auc += ~(a_ < real_auc)
        ge_b += not (b_ < real_b)
        ge_f += not (_evaluate(ff, m, blocks, K)[1] < face_b)
    layer, face = fit_layer(X, F, y, K)
    p_auc = (1 + ge_auc) / (1 + perms)
    checks = {s: {"auc": float(real_auc[k]), "p": float(p_auc[k]), "on": bool(real_auc[k] > 0.5 and p_auc[k] <= ALPHA),
                  "blocks": int(per[k])} for k, s in enumerate(states)}
    return {"states": list(states), "layer": layer, "face": face, "checks": checks, "bacc": real_b,
            "bacc_p": (1 + ge_b) / (1 + perms), "face_only": {"bacc": face_b, "p": (1 + ge_f) / (1 + perms)},
            "blocks": len(ub), "windows": len(y)}


def probs(model, X, F):
    return softmax(state_input(X, F, model["face"]) @ model["layer"]["W"] + model["layer"]["b"])
