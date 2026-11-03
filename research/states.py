#!/usr/bin/env python3
"""Brain-only decoding of stress and of three clinical/trait screens on the Emotiv Insight layout
(AF3, AF4, T7, T8, Pz), with a face-leakage audit for every result.

  python research/states.py stress   --out results/states      # acute stress vs relaxation, 17 people, real Insight
  python research/states.py transfer --out results/states      # GAMEEMO-trained valence/arousal on that Insight data
  python research/states.py traits   --out results/states      # depression (PHQ-9), anxiety (GAD-7), sleepiness (VAS)
  python research/states.py figure   --out results/states

stress    Within-person: windows of the horror-video and relaxing-music recordings, leave-one-person-out, decoders as in
          brain_decoder.py (standard filtered, standard cleaned, Emotif brain-only). Paired test: in how many people
          is the mean output higher for the horror recording? Order control: horror came second for 12 of 17
          people, so the same decoders are also trained to tell the second recording from the first, and the paired
          stress test is repeated for the 5 people who heard the music second.
transfer  The brain-only valence/arousal decoder trained on GAMEEMO (28 people, Emotiv EPOC, restricted to the four
          Insight electrodes it shares: AF3, AF4, T7, T8) applied without retraining to the stress data (another
          lab, another headset, another stimulus). Pre-registered direction: horror lower valence, higher arousal.
traits    One label per person, so features describe the person: relative log band power (band / 1-45 Hz total; gain-
          free) per scalp region, median over 2 s windows of the cleaned signal, plus left-right asymmetry. Face
          covariates per person: median eye and jaw indices over windows and electrodes. Leave-one-person-out
          logistic regression (standard, and brain-only = features minus their ridge prediction from the face
          covariates, fit on training people), person-level permutation test (1000), and a small set of markers
          chosen from the literature before looking at these data, tested with Holm correction.
All results are research measurements on screening-score groups, not diagnoses.
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
from emotif import asr as asr_mod  # noqa: E402
from emotif import audit, models  # noqa: E402
from emotif import datasets as D  # noqa: E402
from emotif import face as FACE  # noqa: E402
from emotif import features as F  # noqa: E402
from emotif import montage as M  # noqa: E402

DATA = {"stress": ("stress_insight", "data/external/stress_insight"),
        "depression": ("depression_phq9", "data/external/depression_phq9"),
        "anxiety": ("anxiety_gad7", "data/external/anxiety_gad7"),
        "sleepiness": ("sleepiness", "data/external/sleepiness"),
        "gameemo": ("gameemo_raw", "data/external/gameemo")}
CONDS = ["eeg_filtered", "eeg_cleaned", "brain_live"]


DEVICE = "emotiv_insight"


def load(name, device="insight"):
    device = DEVICE if device == "insight" else device
    loader, path = DATA[name]
    tr = D.LOADERS[loader](path)
    return D.to_device(tr, device) if device else tr


def drop_channels(trials, names):
    out = []
    for t in trials:
        keep = [i for i, c in enumerate(t.channels) if M.canonical(c) not in {n.upper() for n in names}]
        out.append(D.Trial(t.subject, t.trial_id, t.x[keep], t.fs, [t.channels[i] for i in keep], dict(t.labels),
                           t.filtered, t.dataset, t.group))
    return out


def cached(path, build):
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=False))
    d = build()
    np.savez_compressed(path, **d)
    return d


def binom_two_sided(k, n):
    from scipy.stats import binomtest
    return float(binomtest(k, n, 0.5).pvalue)


def paired(d, dec, y, sel_subjects=None):
    """Per person: mean output over the +1 recording minus over the -1 recording."""
    out = {}
    for s in np.unique(d["subject"]):
        if sel_subjects is not None and s not in sel_subjects:
            continue
        ms = d["subject"] == s
        a, b = dec[ms & (y == 1)], dec[ms & (y == -1)]
        if len(a) and len(b):
            out[str(s)] = float(a.mean() - b.mean())
    v = np.array(list(out.values()))
    from scipy.stats import wilcoxon
    k = int((v > 0).sum())
    return {"n": len(v), "higher": k, "sign_p": binom_two_sided(k, len(v)),
            "wilcoxon_p": float(wilcoxon(v).pvalue) if len(v) >= 5 else None, "mean_diff": float(v.mean()),
            "per_person": out}


# ----------------------------------------------------------------------------------------------- within-person stress
def run_stress(a):
    import brain_decoder as BD
    os.makedirs(a.out, exist_ok=True)
    d = cached(os.path.join(a.out, "stress_cache.npz"),
               lambda: BD.assemble_all(load("stress"), a.hop, ["stress", "later"]))
    print(f"stress: {len(d['subject'])} windows, {len(np.unique(d['subject']))} people")
    rng = np.random.default_rng(0)
    res = {"hop": a.hop, "smooth_windows": a.smooth}
    # people who heard the music second: the order control subgroup
    music_second = sorted({str(s) for s in np.unique(d["subject"])
                           if (d["y_later"][(d["subject"] == s) & (d["y_stress"] == -1)] == 1).all()})
    res["music_second"] = music_second
    for t in ("stress", "later"):
        y = d["y_" + t]
        ok = ~np.isnan(y)
        classes = np.unique(y[ok])
        perms = [audit.permute_recordings(dict(d, y=np.where(ok, y, 0)), rng) for _ in range(a.perms)]
        res[t] = {}
        for cond in CONDS:
            r = {}
            for sm in (1, a.smooth):
                BD.SMOOTH = sm
                pred, dec = BD.loso(d, cond, y)
                r[f"bacc_s{sm}"] = audit.balanced_accuracy(y[ok], pred[ok], classes)
            BD.SMOOTH = 1
            pred, dec = BD.loso(d, cond, y)
            r["face_r2"] = BD.face_leakage(d, dec, ok)
            r["paired"] = paired(d, dec, y)
            if t == "stress":
                r["paired_music_second"] = paired(d, dec, y, set(music_second))
            if perms:
                nv = np.array([audit.balanced_accuracy(yp[ok], BD.loso(d, cond, np.where(ok, yp, np.nan))[0][ok], classes)
                               for yp in perms])
                r["p_perm"] = float((1 + np.sum(nv >= r["bacc_s1"])) / (1 + len(nv)))
            res[t][cond] = r
            pp = r["paired"]
            print(f"{t:7s} {cond:13s} bacc {r['bacc_s1']:.3f} / {r[f'bacc_s{a.smooth}']:.3f} ({a.smooth * a.hop:.0f} s)  "
                  f"face R2 {r['face_r2']:+.3f}  paired {pp['higher']}/{pp['n']} p={pp['sign_p']:.3f}"
                  + (f"  perm p={r['p_perm']:.3f}" if "p_perm" in r else "")
                  + (f"  music-second {r['paired_music_second']['higher']}/{r['paired_music_second']['n']}" if t == "stress" else ""),
                  flush=True)
    with open(os.path.join(a.out, "stress.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


# ------------------------------------------------------------------------------------------ cross-dataset transfer
def run_transfer(a):
    import brain_decoder as BD
    from sklearn.linear_model import LogisticRegression
    os.makedirs(a.out, exist_ok=True)
    shared = ["AF3", "AF4", "T7", "T8"]
    g = cached(os.path.join(a.out, "gameemo_insight_cache.npz"),
               lambda: BD.assemble_all(load("gameemo"), a.hop, ["valence", "arousal"]))
    s = cached(os.path.join(a.out, "stress_shared_cache.npz"),
               lambda: BD.assemble_all(drop_channels(load("stress"), ["Pz"]), a.hop, ["stress", "later"]))
    print(f"GAMEEMO (AF3, AF4, T7, T8): {len(g['subject'])} windows; stress data: {len(s['subject'])} windows")
    res = {"electrodes": shared}
    for t, expect in (("valence", -1), ("arousal", 1)):
        y = g["y_" + t]
        ok = ~np.isnan(y)
        out = {}
        for cond, (Etr, Ete) in {
            "eeg_cleaned": (g["eeg_cleaned"][ok], s["eeg_cleaned"]),
            "brain_live": audit._residualize(g["eeg_cleaned"][ok], s["eeg_cleaned"], g["face_live"][ok], s["face_live"]),
        }.items():
            Xtr, Xte = audit._standardize(Etr, Ete)
            clf = LogisticRegression(C=0.3, max_iter=3000).fit(Xtr, y[ok])
            dec = clf.predict_proba(Xte) @ clf.classes_.astype(float)
            pr = paired(s, dec, s["y_stress"])
            pr_ms = paired(s, dec, s["y_stress"], {p for p in np.unique(s["subject"]).astype(str)
                                                   if (s["y_later"][(s["subject"] == p) & (s["y_stress"] == -1)] == 1).all()})
            hits = pr["higher"] if expect > 0 else pr["n"] - pr["higher"]
            out[cond] = {"paired": pr, "paired_music_second": pr_ms, "in_expected_direction": hits,
                         "face_r2": BD.face_leakage(s, dec, np.ones(len(dec), bool))}
            print(f"{t:8s} {cond:12s} horror {'<' if expect < 0 else '>'} relax in {hits}/{pr['n']} people "
                  f"(sign p={pr['sign_p']:.3f}, Wilcoxon p={pr['wilcoxon_p']:.3f}, mean diff {pr['mean_diff']:+.3f}); "
                  f"face R2 {out[cond]['face_r2']:+.3f}", flush=True)
        res[t] = out
    with open(os.path.join(a.out, "transfer.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


# ------------------------------------------------------------------------------------------------ person-level traits
BAND_IX = {b: i for i, b in enumerate(M.BAND_NAMES)}


def person_features(t, asr_k=asr_mod.DEFAULT_K, hop=2.0):
    """(regional relative log power (R, 5), region mask (R,), face covariates (3,), n windows, per-window regional
    relative log power (n_clean, R, 5) over windows where every electrode was usable) for one recording."""
    fs = t.fs
    x = t.x if t.filtered else F.filter_signal(t.x, fs)
    xf = x
    try:
        state = asr_mod.calibrate(x[:, : int(600 * fs)], fs, asr_k)
        x = asr_mod.clean(x, state, fs)[0]
    except ValueError:
        pass
    slow = FACE.slow_signal(xf, fs)
    N = F.window_len(fs)
    rel, faces, goods = [], [], []
    Wall, _ = M.region_weights(list(t.channels))
    win_reg = []                                                      # windows with every electrode usable
    for s0 in F.sliding_windows(x, fs, hop):
        win = x[:, s0:s0 + N]
        good, reject = F.channel_quality(win, 0.0, F.BAD_UV)
        if reject or not good.any():
            continue
        bp = F.band_powers(win, fs)                                   # (C, 5)
        r = np.log(bp / bp.sum(axis=1, keepdims=True))
        r[~good] = np.nan
        rel.append(r)
        goods.append(good)
        if good.all():
            win_reg.append(Wall @ r)
        faces.append(FACE.window_indices(xf[:, s0:s0 + N], slow[:, s0:s0 + N], fs))
    if not rel:
        raise ValueError(f"{t.trial_id}: no usable windows")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ch = np.nanmedian(np.array(rel), axis=0)                      # (C, 5)
    usable = ~np.isnan(ch[:, 0])
    W, mask = M.region_weights([c for c, u in zip(t.channels, usable) if u])
    reg = W @ ch[usable]
    face = np.median(np.array(faces), axis=0).mean(axis=0)          # (3,) eye, jaw 20-30, jaw 30-45 (absolute, log)
    return reg, mask, face, len(rel), np.array(win_reg).reshape(-1, len(M.REGIONS), len(M.BAND_NAMES))


def trait_design(reg, mask):
    """Observed regions' relative power plus left-right asymmetry (right minus left) for observed pairs."""
    idx = {r: i for i, r in enumerate(M.REGIONS)}
    cols, names = [reg[:, mask > 0, :].reshape(len(reg), -1)], [f"{r}_{b}" for r in np.array(M.REGIONS)[mask > 0] for b in M.BAND_NAMES]
    for l, r in M.REGION_PAIRS:
        if mask[idx[l]] and mask[idx[r]]:
            cols.append(reg[:, idx[r], :] - reg[:, idx[l], :])
            names += [f"asym_{r}-{l}_{b}" for b in M.BAND_NAMES]
    return np.hstack(cols), names


# Markers fixed before analysis (sources in docs/LITERATURE.md). Each is (name, function of reg (R, 5)).
def _alpha_asym(reg, l, r):
    i = {x: k for k, x in enumerate(M.REGIONS)}
    return reg[i[r], BAND_IX["alpha"]] - reg[i[l], BAND_IX["alpha"]]


def _mean(reg, regions, band):
    i = {x: k for k, x in enumerate(M.REGIONS)}
    return float(np.mean([reg[i[r], BAND_IX[band]] for r in regions]))


FRONT = ["Fp1", "Fp2", "F3", "F4"]
MARKERS = {
    "depression": [("frontal alpha asymmetry (F4 - F3)", lambda g: _alpha_asym(g, "F3", "F4")),
                   ("frontal relative alpha", lambda g: _mean(g, FRONT, "alpha")),
                   ("frontal relative theta", lambda g: _mean(g, FRONT, "theta"))],
    "anxiety": [("frontal relative beta", lambda g: _mean(g, FRONT, "beta")),
                ("frontal alpha asymmetry (F4 - F3)", lambda g: _alpha_asym(g, "F3", "F4")),
                ("frontal relative alpha", lambda g: _mean(g, FRONT, "alpha"))],
    "sleepiness": [("relative theta, all electrodes", lambda g: float(np.nanmean(np.where(g[:, 0] != 0, g[:, BAND_IX["theta"]], np.nan)))),
                   ("relative alpha, all electrodes", lambda g: float(np.nanmean(np.where(g[:, 0] != 0, g[:, BAND_IX["alpha"]], np.nan)))),
                   ("relative delta, all electrodes", lambda g: float(np.nanmean(np.where(g[:, 0] != 0, g[:, BAND_IX["delta"]], np.nan))))],
}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def loso_person(X, y, Fc, brain, C):
    from sklearn.linear_model import LogisticRegression
    dec = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        Xtr, Xte = X[tr], X[i:i + 1]
        if brain:
            Xtr, Xte = audit._residualize(Xtr, Xte, Fc[tr], Fc[i:i + 1])
        Xtr, Xte = audit._standardize(Xtr, Xte)
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xtr, y[tr])
        dec[i] = clf.decision_function(Xte)[0]
    return dec


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def face_r2_person(Fc, dec):
    Z = np.hstack([Fc, Fc ** 2, np.ones((len(Fc), 1))])
    B, *_ = np.linalg.lstsq(Z, dec, rcond=None)
    r2 = 1 - np.sum((dec - Z @ B) ** 2) / np.sum((dec - dec.mean()) ** 2)
    n, k = Z.shape
    return float(1 - (1 - r2) * (n - 1) / (n - k))


def window_split_check(d, mask, y):
    """5-fold window-level balanced accuracy, (a) windows assigned to folds at random, as many published results on
    these datasets do, and (b) all windows of a person in one fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    X, _ = trait_design(d["win_reg"], mask)
    yw = y[d["win_person"]]
    from sklearn.ensemble import RandomForestClassifier
    out = {"n_windows": int(len(yw))}
    makers = {"logreg": lambda: LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"),
              "forest": lambda: RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", n_jobs=2, random_state=0)}
    for model, make in makers.items():
        for name, splits in (("random", StratifiedKFold(5, shuffle=True, random_state=0).split(X, yw)),
                             ("by_person", GroupKFold(5).split(X, yw, d["win_person"]))):
            pred = np.zeros(len(yw), int)
            for tr, te in splits:
                Xtr, Xte = audit._standardize(X[tr], X[te])
                pred[te] = make().fit(Xtr, yw[tr]).predict(Xte)
            out[f"{model}_{name}"] = audit.balanced_accuracy(yw, pred, np.array([-1, 1]))
    return out


def run_traits(a):
    from scipy.stats import mannwhitneyu, spearmanr
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {}
    for name in ("depression", "anxiety", "sleepiness"):
        def build():
            trials = load(name)
            rows = [person_features(t) for t in trials]
            return {"subject": np.array([t.subject for t in trials]), "y": np.array([t.labels[name] for t in trials], float),
                    "reg": np.array([r[0] for r in rows]), "mask": np.array([r[1] for r in rows]),
                    "face": np.array([r[2] for r in rows]), "n_windows": np.array([r[3] for r in rows]),
                    "win_reg": np.concatenate([r[4] for r in rows]),
                    "win_person": np.concatenate([np.full(len(r[4]), i) for i, r in enumerate(rows)])}
        d = cached(os.path.join(a.out, f"trait_{name}_{DEVICE or 'full'}_cache.npz"), build)
        masks = np.unique(d["mask"], axis=0)
        if len(masks) != 1:
            raise ValueError(f"{name}: recordings cover different regions; expected one layout")
        X, names = trait_design(d["reg"], masks[0])
        y, Fc = d["y"].astype(int), d["face"]
        r = {"n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == -1).sum()), "features": names,
             "windows_per_person_median": float(np.median(d["n_windows"]))}
        # markers fixed in advance
        mk = []
        for label, fn in MARKERS[name]:
            v = np.array([fn(g) for g in d["reg"]])
            st = mannwhitneyu(v[y == 1], v[y == -1])
            mk.append({"marker": label, "median_pos": float(np.median(v[y == 1])), "median_neg": float(np.median(v[y == -1])),
                       "auc": auc(y, v), "p": float(st.pvalue)})
        for m_, p_ in zip(mk, holm([m_["p"] for m_ in mk])):
            m_["p_holm"] = p_
        if name == "sleepiness":
            scores = np.array([D.SLEEPINESS_SCORES.get(s, {}).get("VASS", np.nan) for s in d["subject"]]) \
                if D.SLEEPINESS_SCORES else None
            if scores is None or np.isnan(scores).any():
                D.load_sleepiness(DATA["sleepiness"][1])
                scores = np.array([D.SLEEPINESS_SCORES[s]["VASS"] for s in d["subject"]])
            for (label, fn), m_ in zip(MARKERS[name], mk):
                v = np.array([fn(g) for g in d["reg"]])
                rho = spearmanr(v, scores)
                m_["spearman_vass"] = float(rho.statistic)
                m_["spearman_p"] = float(rho.pvalue)
        r["markers"] = mk
        # face covariates differ between groups?
        r["face_group_p"] = [float(mannwhitneyu(Fc[y == 1, j], Fc[y == -1, j]).pvalue) for j in range(3)]
        for cond, brain in (("standard", False), ("brain_only", True)):
            dec = loso_person(X, y, Fc, brain, a.C)
            pred = np.where(dec > 0, 1, -1)
            bacc = audit.balanced_accuracy(y, pred, np.array([-1, 1]))
            null = []
            for _ in range(a.perms):
                yp = rng.permutation(y)
                null.append(auc(yp, loso_person(X, yp, Fc, brain, a.C)))
            null = np.array(null)
            A_ = auc(y, dec)
            r[cond] = {"bacc": bacc, "auc": A_, "face_r2": face_r2_person(Fc, dec),
                       "p_perm": float((1 + np.sum(null >= A_)) / (1 + len(null))) if len(null) else None}
            print(f"{name:10s} {cond:10s} n={len(y)} ({r['n_pos']}+/{r['n_neg']}-)  AUC {A_:.3f}  bacc {bacc:.3f}  "
                  f"face R2 {r[cond]['face_r2']:+.3f}" + (f"  perm p={r[cond]['p_perm']:.3f}" if len(null) else ""), flush=True)
        # how the usual evaluation inflates accuracy: windows of the same person in training and test folds
        r["window_split"] = window_split_check(d, masks[0], y)
        ws = r["window_split"]
        print(f"   per-window accuracy, windows split at random (same people in train and test) vs split by person: "
              f"logistic {ws['logreg_random']:.3f} vs {ws['logreg_by_person']:.3f}; forest {ws['forest_random']:.3f} vs {ws['forest_by_person']:.3f}")
        for m_ in mk:
            print(f"   marker {m_['marker']:36s} AUC {m_['auc']:.3f}  p={m_['p']:.3f}  Holm p={m_['p_holm']:.3f}"
                  + (f"  rho(VAS)={m_['spearman_vass']:+.2f} p={m_['spearman_p']:.3f}" if "spearman_vass" in m_ else ""))
        res[name] = r
    with open(os.path.join(a.out, f"traits_{DEVICE or 'full'}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["stress", "transfer", "traits", "figure"])
    p.add_argument("--out", required=True)
    p.add_argument("--hop", type=float, default=2.0)
    p.add_argument("--smooth", type=int, default=15, help="windows averaged for the smoothed accuracy (15 x 2 s = 30 s)")
    p.add_argument("--perms", type=int, default=0)
    p.add_argument("--C", type=float, default=0.05, help="inverse regularization for the person-level models")
    p.add_argument("--full", action="store_true", help="traits: every electrode of the dataset instead of the Insight layout")
    a = p.parse_args(argv)
    global DEVICE
    if a.full:
        DEVICE = None
    if a.what == "stress":
        run_stress(a)
    elif a.what == "transfer":
        run_transfer(a)
    elif a.what == "traits":
        run_traits(a)
    else:
        figure(a.out)


def figure(out):
    """Three panels: within-person stress (paired), person-level traits (standard vs brain-only) and how random
    window splits inflate trait accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = json.load(open(os.path.join(out, "stress.json")))
    tr = json.load(open(os.path.join(out, "traits_emotiv_insight.json")))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=170, gridspec_kw={"width_ratios": [1.1, 1.2, 1.2]})
    names = {"eeg_filtered": "standard,\nfilter only", "eeg_cleaned": "standard,\ncleaned", "brain_live": "Emotif\nbrain-only"}
    col = {"eeg_filtered": "#9aa5b1", "eeg_cleaned": "#64748b", "brain_live": "#1d4e89"}
    rng = np.random.default_rng(1)
    allv = np.concatenate([list(st["stress"][c]["paired"]["per_person"].values()) for c in CONDS])
    top = float(allv.max())
    ax[0].set_ylim(float(allv.min()) - 0.05, top + 0.42)
    for i, c in enumerate(CONDS):
        r = st["stress"][c]
        v = np.array(list(r["paired"]["per_person"].values()))
        ms = set(st["music_second"])
        keys = list(r["paired"]["per_person"].keys())
        xs = i + rng.uniform(-0.12, 0.12, len(v))
        ax[0].scatter(xs, v, s=16, color=[("#d97706" if k in ms else col[c]) for k in keys], zorder=3)
        ax[0].text(i, top + 0.05, f"{r['paired']['higher']}/{r['paired']['n']} higher\nWilcoxon p={r['paired']['wilcoxon_p']:.3f}\nface R² {max(0, r['face_r2']):.2f}",
                   ha="center", va="bottom", fontsize=7)
    ax[0].axhline(0, color="#334155", lw=0.8)
    ax[0].set_xticks(range(len(CONDS)), [names[c] for c in CONDS], fontsize=7.5)
    ax[0].set_ylabel("output: horror minus relaxing music", fontsize=8)
    ax[0].set_title("A. Stress within a person (17, real Insight)\norange: music recorded second", fontsize=9)
    lab = {"depression": "Depression\n(PHQ-9, n=31)", "anxiety": "Anxiety\n(GAD-7, n=38)", "sleepiness": "Sleepiness\n(VAS, n=49)"}
    xs = np.arange(3)
    for j, (cond, cc) in enumerate((("standard", "#64748b"), ("brain_only", "#1d4e89"))):
        a = [tr[t][cond]["auc"] for t in lab]
        ax[1].bar(xs + (j - 0.5) * 0.36, a, 0.34, color=cc, label="standard" if j == 0 else "Emotif brain-only")
        for x, t in zip(xs, lab):
            r = tr[t][cond]
            ax[1].text(x + (j - 0.5) * 0.36, r["auc"] + 0.01, f"{r['auc']:.2f}\nface {max(0, r['face_r2']):.2f}"
                       + (f"\np={r['p_perm']:.2f}" if r.get("p_perm") is not None else ""), ha="center", va="bottom", fontsize=6.5)
    ax[1].axhline(0.5, color="#334155", ls="--", lw=0.8)
    ax[1].set_ylim(0, 1)
    ax[1].set_xticks(xs, list(lab.values()), fontsize=7.5)
    ax[1].set_ylabel("AUC, new person (leave-one-person-out)", fontsize=8)
    ax[1].set_title("B. Trait screens from 5 min of rest (Insight layout)", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False, loc="upper left")
    for j, (k, cc, nm) in enumerate((("forest_random", "#d97706", "windows split at random"), ("forest_by_person", "#1d4e89", "split by person"))):
        a = [tr[t]["window_split"][k] for t in lab]
        ax[2].bar(xs + (j - 0.5) * 0.36, a, 0.34, color=cc, label=nm)
        for x, v in zip(xs, a):
            ax[2].text(x + (j - 0.5) * 0.36, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax[2].axhline(0.5, color="#334155", ls="--", lw=0.8)
    ax[2].set_ylim(0, 1)
    ax[2].set_xticks(xs, list(lab.values()), fontsize=7.5)
    ax[2].set_ylabel("balanced accuracy per 2 s window", fontsize=8)
    ax[2].set_title("C. Same data, random forest: the split decides", fontsize=9)
    ax[2].legend(fontsize=7, frameon=False, loc="upper left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_states.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
