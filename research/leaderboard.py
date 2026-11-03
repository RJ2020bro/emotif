#!/usr/bin/env python3
"""Why published EEG emotion accuracies on these public datasets are so high: the same windows, the same features,
evaluated the way many papers do (windows split at random) and with controls that tell recording identity, the face
and the brain apart.

  python research/leaderboard.py run    --out results/leaderboard [--fast]
  python research/leaderboard.py figure --out results/leaderboard

Uses the feature caches written by research/brain_decoder.py (GAMEEMO) and research/states.py (Insight stress,
trait screens). Each recording has a single label in all of these datasets, so a window-level random split puts
windows of the same recording in training and test.

Conditions (balanced accuracy; random forest and logistic regression on the same standardised features):
  random            5-fold, windows assigned at random (the common published protocol)
  random_permuted   same split, but every recording is given a random class (balanced, independent of its true
                    label) before training and testing. The labels no longer mean anything, so any accuracy above
                    chance here is recording (or person) identification, not emotion.
  random_face       random split, face indices only (eye 0-3 Hz peak-to-peak, 20-30 and 30-45 Hz power per
                    electrode; no brain rhythms)
  temporal          each recording's first 70 % of windows train, last 30 % test, 2 windows dropped between them
  subject           7-fold, all windows of a person in one fold (new people)
  subject_face      subject-wise, face indices only
  subject_brain     subject-wise, Emotif brain-only decoder on the cleaned features (what the live face indices
                    predict removed)
  record_id         random split, predicting WHICH recording a window came from (chance 1 / recordings)
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from emotif import models  # noqa: E402

CACHE = {"gameemo": "results/brain_gameemo_v2/cache.npz",
         "stress": "results/states/stress_shared_cache.npz",
         "depression": "results/states/trait_depression_emotiv_insight_cache.npz",
         "anxiety": "results/states/trait_anxiety_emotiv_insight_cache.npz",
         "sleepiness": "results/states/trait_sleepiness_emotiv_insight_cache.npz"}
TITLE = {"gameemo": "GAMEEMO, 4 games (28 people, EPOC+ 14 ch)",
         "stress": "Stress, horror vs music (17 people, Insight)",
         "depression": "Depression PHQ-9 (31 people, Insight layout)",
         "anxiety": "Anxiety GAD-7 (38 people, Insight layout)",
         "sleepiness": "Sleepiness VAS (49 people, Insight layout)"}
CONDITIONS = ["random", "random_permuted", "random_face", "temporal", "subject", "subject_face", "subject_brain", "record_id"]


def bacc(y, p):
    cls = np.unique(y)
    return float(np.mean([np.mean(p[y == c] == c) for c in cls]))


def make(kind, fast):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "forest":
        return RandomForestClassifier(100 if fast else 300, min_samples_leaf=1, n_jobs=-1, random_state=0,
                                      class_weight="balanced")
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))


def load(name):
    """Returns dict X (windows x features), y (window labels, ints), rec (recording id per window), subj (person per
    window), face (window face indices or None), face_live (pooled live indices or None)."""
    d = dict(np.load(CACHE[name], allow_pickle=False))
    if name == "gameemo":
        y = ((d["y_valence"] > 0).astype(int) * 2 + (d["y_arousal"] > 0).astype(int))
        return {"X": d["eeg_filtered"], "y": y, "rec": d["group"], "subj": d["subject"], "face": d["face"],
                "face_live": d["face_live"], "Xc": d["eeg_cleaned"]}
    if name == "stress":
        return {"X": d["eeg_filtered"], "y": (d["y_stress"] > 0).astype(int), "rec": d["group"], "subj": d["subject"],
                "face": d["face"], "face_live": d["face_live"], "Xc": d["eeg_cleaned"]}
    from states import trait_design
    mask = d["mask"][0]
    X, _ = trait_design(d["win_reg"], mask)
    person = d["win_person"]
    y = (d["y"][person] > 0).astype(int)
    rec = np.array([f"p{p}" for p in person])
    return {"X": X, "y": y, "rec": rec, "subj": rec, "face": None, "face_live": None}


def permuted_labels(y, rec, subj, rng):
    """Meaningless labels: every recording gets a class drawn from a balanced, shuffled list (independent of its true
    label), and all its windows carry it. Any accuracy above chance on these labels comes from recognising
    recordings, not states."""
    recs = np.unique(rec)
    cls = np.unique(y)
    lab = np.resize(cls, len(recs))[rng.permutation(len(recs))]
    m = dict(zip(recs, lab))
    return np.array([m[r] for r in rec])


def temporal_split(rec, frac=0.7, gap=2):
    tr, te = [], []
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        k = int(round(len(idx) * frac))
        tr += idx[:k].tolist()
        te += idx[k + gap:].tolist()
    return np.array(tr), np.array(te)


def cv_predict(model_kind, X, y, splits, fast, brain=None):
    pred = np.full(len(y), -1)
    for tr, te in splits:
        if brain is not None:
            m = models.BrainModel(make(model_kind, fast), alpha=1.0).fit(X[tr], y[tr], brain[tr])
            pred[te] = m.predict(X[te], brain[te])
        else:
            pred[te] = make(model_kind, fast).fit(X[tr], y[tr]).predict(X[te])
    return pred


def run_dataset(name, fast, perms, seed=0):
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    D = load(name)
    X, y, rec, subj = D["X"], D["y"], D["rec"], D["subj"]
    rng = np.random.default_rng(seed)
    out = {"n_windows": int(len(y)), "n_recordings": int(len(np.unique(rec))), "n_people": int(len(np.unique(subj))),
           "n_classes": int(len(np.unique(y))), "chance": 1.0 / len(np.unique(y)), "results": {}}
    rnd = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    gk = list(GroupKFold(min(7, len(np.unique(subj)))).split(X, y, subj))
    tr, te = temporal_split(rec)
    for kind in ("forest", "logreg"):
        R = out["results"].setdefault(kind, {})
        p = cv_predict(kind, X, y, rnd, fast)
        R["random"] = bacc(y, p)
        accs = []
        for k in range(perms):
            yp = permuted_labels(y, rec, subj, rng)
            rs = list(StratifiedKFold(5, shuffle=True, random_state=seed + k).split(X, yp))
            accs.append(bacc(yp, cv_predict(kind, X, yp, rs, fast)))
        R["random_permuted"] = float(np.mean(accs))
        R["random_permuted_all"] = accs
        if D["face"] is not None:
            R["random_face"] = bacc(y, cv_predict(kind, D["face"], y, rnd, fast))
        m = make(kind, fast).fit(X[tr], y[tr])
        R["temporal"] = bacc(y[te], m.predict(X[te]))
        R["subject"] = bacc(y, cv_predict(kind, X, y, gk, fast))
        if D["face"] is not None:
            R["subject_face"] = bacc(y, cv_predict(kind, D["face"], y, gk, fast))
        print(f"  {name} {kind}: " + ", ".join(f"{k} {v:.3f}" for k, v in R.items() if isinstance(v, float)), flush=True)
    if D["face_live"] is not None:
        for kind in ("logreg", "forest"):
            out["results"][kind]["subject_brain"] = bacc(y, cv_predict(kind, D["Xc"], y, gk, fast, brain=D["face_live"]))
            print(f"  {name} brain-only subject-wise ({kind}): {out['results'][kind]['subject_brain']:.3f}", flush=True)
    # how identifiable is a recording from one 2 s window?
    rid = np.unique(rec, return_inverse=True)[1]
    rs = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, rid))
    out["record_id"] = {"accuracy": float(np.mean(cv_predict("forest", X, rid, rs, fast) == rid)),
                        "chance": 1.0 / out["n_recordings"]}
    print(f"  {name} recording identity: {out['record_id']['accuracy']:.3f} (chance {out['record_id']['chance']:.4f})",
          flush=True)
    return out


def figure(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(os.path.join(out_dir, "leaderboard.json")))
    names = [n for n in CACHE if n in res]
    rows = [("random", "Random window split\n(published protocol)"), ("random_permuted", "Same split,\nlabels scrambled"),
            ("random_face", "Same split,\nface only"), ("temporal", "Later part of\neach recording"),
            ("subject", "New people"), ("subject_face", "New people,\nface only"),
            ("subject_brain", "New people,\nbrain-only")]
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 4.4), sharey=True)
    colour = {"random": "#b23a48", "random_permuted": "#e07a5f", "random_face": "#f2cc8f", "temporal": "#81b29a",
              "subject": "#3d5a80", "subject_face": "#98c1d9", "subject_brain": "#293241"}
    ypos = {key: len(rows) - 1 - i for i, (key, _) in enumerate(rows)}
    for ax, n in zip(np.atleast_1d(axes), names):
        r = res[n]
        src = r["results"]["forest"]
        for key, _ in rows:
            yy = ypos[key]
            if key in src:
                v = src[key]
                ax.barh(yy, v, color=colour[key])
                ax.text(min(v + 0.01, 0.97), yy, f"{v:.2f}", va="center", fontsize=8)
            else:
                ax.text(0.02, yy, "no face data per window", va="center", fontsize=7, color="0.5")
        ax.axvline(r["chance"], color="k", lw=1, ls="--")
        ax.set_xlim(0, 1.05)
        ax.set_title(TITLE[n], fontsize=9)
        ax.set_xlabel("balanced accuracy (dashed: chance)", fontsize=8)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks([ypos[k] for k, _ in rows])
    ax0.set_yticklabels([lab for _, lab in rows], fontsize=8)
    fig.suptitle("Random window splits score as well with scrambled labels; on GAMEEMO the face alone names the game "
                 "for new people (random forest, 2 s windows)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_leaderboard.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "fig_leaderboard.pdf"))
    print("wrote", os.path.join(out_dir, "fig_leaderboard.png"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["run", "figure"])
    p.add_argument("--out", default="results/leaderboard")
    p.add_argument("--datasets", default=",".join(CACHE))
    p.add_argument("--perms", type=int, default=3, help="label permutations per model for the scrambled control")
    p.add_argument("--fast", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "figure":
        return figure(a.out)
    warnings.filterwarnings("ignore")
    path = os.path.join(a.out, "leaderboard.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for n in a.datasets.split(","):
        print(n, flush=True)
        res[n] = run_dataset(n, a.fast, a.perms)
        json.dump(res, open(path, "w"), indent=1)
    figure(a.out)
    return res


if __name__ == "__main__":
    main()
