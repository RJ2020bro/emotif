"""Known-answer tests: python -m pytest facecheck/tests (or python facecheck/tests/test_facecheck.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import facecheck as fc  # noqa: E402


def make(signal, n_sub=12, n_win=60, n_feat=20, n_face=6, seed=0):
    """Each subject has one recording per class (label constant within a recording) and every recording has its own
    random offset in feature space (a 'fingerprint'). signal: 'none' (labels in nothing but the fingerprint),
    'brain' (labels shift the features the same way in every subject), 'face' (labels shift only the face, which
    then leaks into the features)."""
    rng = np.random.default_rng(seed)
    X, y, rec, sub, face = [], [], [], [], []
    w_brain = rng.normal(0, 1, n_feat)
    w_leak = rng.normal(0, 1, (n_face, n_feat))
    for s in range(n_sub):
        for c in (0, 1):
            fp = rng.normal(0, 1.5, n_feat)
            f = rng.normal(0, 1, (n_win, n_face)) + (1.5 * c if signal == "face" else 0.0)
            x = fp + rng.normal(0, 1, (n_win, n_feat)) + f @ w_leak * (0.8 if signal == "face" else 0.0)
            if signal == "brain":
                x = x + 1.2 * c * w_brain
            X.append(x); face.append(f)
            y += [c] * n_win; rec += [f"s{s}r{c}"] * n_win; sub += [f"s{s}"] * n_win
    return np.vstack(X), np.array(y), np.array(rec), np.array(sub), np.vstack(face)


def test_split_leakage_detected():
    X, y, rec, sub, face = make("none")
    r = fc.audit(X, y, rec, sub, face, model="logreg")
    assert r["random_split"] > 0.8, r
    assert r["scrambled_labels"] > 0.8, r            # still high with meaningless labels: recording identity
    assert abs(r["subject_split"] - 0.5) < 0.12, r    # nothing about the label transfers to new people
    assert any(v.startswith("SPLIT LEAKAGE") for v in r["verdicts"]), r["verdicts"]
    assert not any(v.startswith("FACE") for v in r["verdicts"]), r["verdicts"]


def test_brain_signal_not_flagged_as_face():
    X, y, rec, sub, face = make("brain")
    r = fc.audit(X, y, rec, sub, face, model="logreg")
    assert r["subject_split"] > 0.8, r
    assert abs(r["face_only_subject"] - 0.5) < 0.12, r
    assert r["face_share"] < 0.1, r
    assert not any(v.startswith("FACE") for v in r["verdicts"]), r["verdicts"]


def test_face_signal_flagged():
    X, y, rec, sub, face = make("face")
    r = fc.audit(X, y, rec, sub, face, model="logreg")
    assert r["subject_split"] > 0.7, r                # the EEG decoder 'works' on new people...
    assert r["face_only_subject"] > 0.8, r            # ...but the face alone does as well
    assert r["leakage_ratio"] > 0.8 and r["face_share"] > 0.3, r
    assert any(v.startswith("FACE LEAKAGE") for v in r["verdicts"]), r["verdicts"]


def test_face_indices():
    fs, T = 128.0, 60
    rng = np.random.default_rng(1)
    t = np.arange(int(T * fs)) / fs
    x = rng.normal(0, 5, (2, len(t)))
    half = t >= T / 2
    blinks = sum(150 * np.exp(-((t - b) ** 2) / (2 * 0.06 ** 2)) for b in np.arange(T / 2 + 1, T, 2.0))
    x[0] += blinks                                    # channel 0: blinks in the second half
    x[1, half] += 30 * rng.normal(0, 1, half.sum())   # channel 1: broadband 'muscle' in the second half
    F, starts = fc.face_indices(x, fs)
    F = F.reshape(len(starts), 2, 3)
    late = starts / fs >= T / 2 + 2
    early = starts / fs + 2 <= T / 2
    assert F[late, 0, 0].mean() - F[early, 0, 0].mean() > 1.0       # eye index rises with blinks
    assert F[late, 1, 1].mean() - F[early, 1, 1].mean() > 1.0       # jaw index rises with muscle
    assert abs(F[late, 1, 0].mean() - F[early, 1, 0].mean()) < abs(F[late, 0, 0].mean() - F[early, 0, 0].mean())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "OK")
