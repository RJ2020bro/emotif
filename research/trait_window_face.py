#!/usr/bin/env python3
"""Per-window face indices for the trait datasets (depression, anxiety, sleepiness), for the leaderboard's face-only
controls.

The trait caches (results/states/trait_<name>_emotiv_insight_cache.npz) keep per-window EEG features (win_reg) but
only one face summary per person. This rebuilds the exact same windows as states.person_features (same cleaning,
hop and "every electrode usable" rule), checks that its EEG features equal the cached ones, and stores the face
indices of those windows the way the GAMEEMO and stress caches do (audit.artifact_evidence: per electrode, eye
0.5-3 Hz peak-to-peak and 20-30 / 30-45 Hz muscle power, relative to the person's median over all windows).

  python research/trait_window_face.py [--datasets depression,anxiety,sleepiness]
Writes results/states/trait_<name>_emotiv_insight_winface.npz with face (n_windows, 3C) aligned to win_reg.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from emotif import asr as asr_mod  # noqa: E402
from emotif import audit  # noqa: E402
from emotif import features as F  # noqa: E402
from emotif import montage as M  # noqa: E402
import states  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "states")


def window_starts(t, hop=2.0, asr_k=asr_mod.DEFAULT_K):
    """Start samples and regional features of the windows states.person_features keeps in win_reg."""
    fs = t.fs
    x = t.x if t.filtered else F.filter_signal(t.x, fs)
    try:
        st = asr_mod.calibrate(x[:, : int(600 * fs)], fs, asr_k)
        x = asr_mod.clean(x, st, fs)[0]
    except ValueError:
        pass
    N = F.window_len(fs)
    Wall, _ = M.region_weights(list(t.channels))
    starts, reg = [], []
    for s0 in F.sliding_windows(x, fs, hop):
        win = x[:, s0:s0 + N]
        good, reject = F.channel_quality(win, 0.0, F.BAD_UV)
        if reject or not good.any() or not good.all():
            continue
        bp = F.band_powers(win, fs)
        starts.append(s0)
        reg.append(Wall @ np.log(bp / bp.sum(axis=1, keepdims=True)))
    return starts, np.array(reg).reshape(-1, len(M.REGIONS), len(M.BAND_NAMES))


def build(name, hop=2.0):
    cache = np.load(os.path.join(OUT, f"trait_{name}_emotiv_insight_cache.npz"))
    trials = states.load(name)
    assert [t.subject for t in trials] == list(cache["subject"]), f"{name}: recordings differ from the cache"
    art = audit.artifact_evidence(trials, hop)
    faces, regs = [], []
    for t in trials:
        starts, reg = window_starts(t, hop)
        faces += [art[(t.trial_id, s)]["index"] for s in starts]
        regs.append(reg)
    reg = np.concatenate(regs)
    assert reg.shape == cache["win_reg"].shape, f"{name}: {reg.shape} windows vs cache {cache['win_reg'].shape}"
    assert np.allclose(reg, cache["win_reg"], atol=1e-6), f"{name}: window features differ from the cache"
    face = np.array(faces)
    assert len(face) == len(cache["win_person"]) and np.isfinite(face).all()
    path = os.path.join(OUT, f"trait_{name}_emotiv_insight_winface.npz")
    np.savez_compressed(path, face=face, win_person=cache["win_person"])
    print(f"{name}: {len(face)} windows, face {face.shape[1]} indices per window, aligned with the cache -> {path}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="depression,anxiety,sleepiness")
    a = p.parse_args(argv)
    for n in a.datasets.split(","):
        build(n)


if __name__ == "__main__":
    main()
