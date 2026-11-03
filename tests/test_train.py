"""Training pipeline on synthetic files in every supported dataset format, plus the real Muse data
(data/external/bird_muse) when present. Checks loaders, montage augmentation, CV, export and the
cross-dataset (mixed-headset) model."""
import json
import os
import pickle
import re
import sys
import tempfile
import warnings

import numpy as np
from scipy.io import savemat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train  # noqa: E402
from emotif import datasets, face as FACE, models, montage as M  # noqa: E402


def synth(channels, fs, seconds, v, a, rng):
    """1/f-ish background + oscillations; valence -> frontal alpha asymmetry, arousal -> beta/alpha."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    x = np.zeros((len(channels), n))
    for i, c in enumerate(channels):
        p = M.position(c)
        bg = np.convolve(rng.normal(0, 1, n), np.ones(8) / 8, mode="same") * 6 + rng.normal(0, 1.5, n)
        alpha = 9 * (1 - 0.4 * a)
        if p is not None and p[1] > 0.3:
            alpha *= (1 + 0.3 * v) if p[0] > 0 else (1 - 0.3 * v)
        env = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 6))
        x[i] = (bg + alpha * env * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))
                + 3 * (1 + 0.6 * a) * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 6))
                + 4 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6)))
    return x


def make_seed(d, rng):
    sd = os.path.join(d, "SEED", "Preprocessed_EEG")
    os.makedirs(sd)
    labels = np.array([[1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]])
    savemat(os.path.join(sd, "label.mat"), {"label": labels})
    for s in (1, 2):
        savemat(os.path.join(sd, f"{s}_20131027.mat"),
                {f"djc_eeg{k + 1}": synth(datasets.SEED_CHANNELS, 200, 10, labels[0, k], 0, rng) for k in range(15)})
    return os.path.join(d, "SEED")


def make_deap(d, rng):
    dd = os.path.join(d, "deap")
    os.makedirs(dd)
    for s in (1, 2):
        data, lab = np.zeros((8, 40, 128 * 20)), np.zeros((8, 4))
        for k in range(8):
            vr, ar = (7.5 if k % 2 else 2.5), (7.5 if (k // 2) % 2 else 2.5)
            lab[k] = [vr, ar, 5, 5]
            data[k, :32] = synth(datasets.DEAP_CHANNELS, 128, 20, 1 if vr > 5 else -1, 1 if ar > 5 else -1, rng)
        with open(os.path.join(dd, f"s{s:02d}.dat"), "wb") as fh:
            pickle.dump({"data": data, "labels": lab}, fh)
    return dd


def make_dreamer(d, rng):
    subjects = []
    for _ in range(2):
        stim, val, aro = np.empty(6, dtype=object), [], []
        for k in range(6):
            v, a = (1 if k % 2 else -1), (1 if k // 3 else -1)
            stim[k] = synth(datasets.EMOTIV_CHANNELS, 128, 16, v, a, rng).T     # (samples, 14) like the real file
            val.append(5 if v > 0 else 1)
            aro.append(5 if a > 0 else 1)
        subjects.append({"EEG": {"stimuli": stim, "baseline": stim}, "ScoreValence": np.array(val), "ScoreArousal": np.array(aro)})
    data = np.empty(len(subjects), dtype=object)
    for i, s in enumerate(subjects):
        data[i] = s
    p = os.path.join(d, "DREAMER.mat")
    savemat(p, {"DREAMER": {"Data": data, "EEG_SamplingRate": 128,
                            "EEG_Electrodes": np.array(datasets.EMOTIV_CHANNELS, dtype=object)}})
    return p


def make_gameemo(d, rng):
    root = os.path.join(d, "GAMEEMO")
    for s in (1, 2):
        sub = os.path.join(root, f"(S{s:02d})", "Preprocessed EEG Data", ".csv format")
        os.makedirs(sub)
        for g, (v, a) in datasets.GAMEEMO_VA.items():
            x = synth(datasets.EMOTIV_CHANNELS, 128, 20, v, a, rng)
            with open(os.path.join(sub, f"S{s:02d}G{g}AllChannels.csv"), "w") as fh:
                fh.write(",".join(datasets.EMOTIV_CHANNELS) + ",\n")
                for i in range(x.shape[1]):
                    fh.write(",".join(f"{val:.4f}" for val in x[:, i]) + ",\n")
    return root


def make_csv(d, rng):
    cd = os.path.join(d, "rec")
    os.makedirs(cd)
    chans, fs = ["TP9", "AF7", "AF8", "TP10"], 256.0
    for name in ("P01", "P02"):
        rows, tt = [], 0
        for v, a in [(1, 1), (0, -1), (-1, 1), (1, -1), (-1, -1), (0, 1)]:
            x = synth(chans, fs, 16, v, a, rng)
            for i in range(x.shape[1]):
                rows.append(f"{tt / fs:.6f}," + ",".join(f"{val:.3f}" for val in x[:, i]) + f",{v},{a}")
                tt += 1
        x = synth(chans, fs, 4, 0, 0, rng)
        for i in range(x.shape[1]):
            rows.append(f"{tt / fs:.6f}," + ",".join(f"{val:.3f}" for val in x[:, i]) + ",,")
            tt += 1
        with open(os.path.join(cd, f"{name}.csv"), "w") as fh:
            fh.write("time_s," + ",".join(chans) + ",valence_label,arousal_label\n" + "\n".join(rows) + "\n")
    return cd


def make_sessions(d, rng):
    """Two recorded sessions in the server's on-disk format, with protocol events and SAM ratings."""
    import json as _json
    from emotif.recorder import Recorder
    root = os.path.join(d, "sessions")
    os.makedirs(root)
    chans, fs = ["TP9", "AF7", "AF8", "TP10"], 256.0
    for p in ("P07", "P08"):
        rec = Recorder(root, p, fs, chans, "test")
        rec.samples(synth(chans, fs, 20, 0, 0, rng).T.tolist())
        for i, (blk, v, a, sv, sa) in enumerate([("happy", 1, 1, 8, 7), ("calm", 1, -1, 7, 2), ("stress", -1, 1, 2, 8), ("sad", -1, -1, 2, 2)] * 2):
            rec.event("block_start", _json.dumps({"block": f"{blk}{i}", "task": "x", "design": {"v": v, "a": a}}))
            rec.samples(synth(chans, fs, 16, v, a, rng).T.tolist())
            rec.event("block_end", _json.dumps({"block": f"{blk}{i}"}))
            rec.event("sam", _json.dumps({"block": f"{blk}{i}", "valence": sv, "arousal": sa}))
        rec.close()
        os.rename(rec.dir, os.path.join(root, f"{rec.id}_{p}"))   # distinct folder names within one second
    return root


def check_export(model):
    assert model["format"] == "emotif-model-v2"
    assert model["feature_space"]["n_features"] == M.N_FEATURES
    for h in model["heads"].values():
        X = np.random.default_rng(0).normal(0, 1, (4, M.N_FEATURES + (FACE.N_FACE_INPUT if h.get("face") else 0)))
        if h.get("face"):
            assert model["face_input"]["kind"] == "face-indices-v1" and h["face"]["hold_z"] > 0
        p = models.forward(h["layers"], X)
        assert p.shape == (4, len(h["classes"])) and np.allclose(p.sum(axis=1), 1)


def brain_known_answer():
    """BrainModel on synthetic data with a known answer. 'brain' features carry the label; face activity is random
    per window and leaks into every feature through a mixing shared across people (anatomy) with person-to-person
    variation, linearly and curved. Label in the brain: the brain-only decoder keeps the accuracy. Label in the face: the standard
    decoder reads it and its output tracks the face; the brain-only decoder falls to chance and doesn't."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(7)
    n_sub, n = 12, 300
    res = {}
    for where in ("brain", "face"):
        X, F_, y, sub = [], [], [], []
        mix0 = rng.normal(0, 0.6, (FACE.N_REGION_FACE, M.N_FEATURES))   # how face activity reaches each feature (anatomy)
        for s in range(n_sub):
            lab = rng.integers(0, 2, n) * 2 - 1
            face = rng.normal(0, 1, (n, FACE.N_REGION_FACE))
            if where == "face":
                face[:, :6] += 0.9 * lab[:, None]                 # the label shows up in eye/jaw activity
            x = rng.normal(0, 1, (n, M.N_FEATURES))
            if where == "brain":
                x[:, 10:16] += 0.6 * lab[:, None]                 # the label is in brain rhythms
            mix = mix0 * (1 + 0.15 * rng.normal(0, 1, mix0.shape))       # varies a little between people
            x += face @ mix + 0.3 * (face ** 2) @ mix              # the face leaks into the EEG features, curved
            X.append(x); F_.append(face); y.append(lab); sub.append(np.full(n, s))
        X, F_, y, sub = map(np.concatenate, (X, F_, y, sub))
        tr, te = sub < 8, sub >= 8
        std = LogisticRegression(C=0.3, max_iter=3000).fit(X[tr], y[tr])
        brain = models.fit_model("brain", X[tr], y[tr], F_[tr])
        acc_s = float(np.mean(std.predict(X[te]) == y[te]))
        acc_b = float(np.mean(brain.predict(X[te], F_[te]) == y[te]))
        out_s = std.predict_proba(X[te])[:, 1]
        out_b = brain.predict_proba(X[te], F_[te])[:, 1]
        Z = np.hstack([F_[te], F_[te] ** 2, np.ones((te.sum(), 1))])
        def face_var(v):       # variance of the output that face activity explains (linear + squares)
            fit = Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
            return float(np.var(fit))
        res[where] = (acc_s, acc_b, face_var(out_s), face_var(out_b))
    (sb, bb, _, _), (sf, bf, rs, rb) = res["brain"], res["face"]
    assert bb > 0.8 and bb >= sb - 0.05, res["brain"]        # brain information kept
    assert sf > 0.8 and bf < 0.6, res["face"]                # face information not decoded
    assert rb < 0.02 * rs, res["face"]                       # and its output barely moves with the face
    print(f"brain-only model, known answer: label in brain -> standard {sb:.2f}, brain-only {bb:.2f}; "
          f"label in face -> standard {sf:.2f}, brain-only {bf:.2f}, face-driven output variance {rs:.4f} -> {rb:.5f}")


def state_loaders(d, rng):
    """Loaders for the stress, depression/anxiety and sleepiness sets on synthetic files laid out like the
    originals; plus a model with an extra "stress" head trained through train.py."""
    import openpyxl
    from emotif.edf import write_edf
    root = os.path.join(d, "stress")
    folders = {"H": "Horrer Video Stimulation", "R": "Participants Listening to Relaxing Music"}
    for k, (h, r) in enumerate(datasets.STRESS_PAIRS):
        for stem, tag, minute in ((h, "H", 10), (r, "R", 20 if k % 3 else 5)):   # every third person hears music first
            os.makedirs(os.path.join(root, folders[tag]), exist_ok=True)
            f = os.path.join(root, folders[tag], stem + ".edf")
            x = synth(datasets.INSIGHT_CHANNELS, 128, 30, -0.8 if tag == "H" else 0.8, 0.8 if tag == "H" else -0.8, rng)
            write_edf(f, x, 128, datasets.INSIGHT_CHANNELS)
            with open(f, "r+b") as fh:
                fh.seek(168)
                fh.write(f"07.06.2212.{minute:02d}.00".encode("ascii"))
    tr = datasets.load_stress_insight(root)
    assert len(tr) == 34 and len({t.subject for t in tr}) == 17 and tr[0].channels == datasets.INSIGHT_CHANNELS
    for t in tr:
        music_first = int(t.subject[-2:]) % 3 == 1          # subject k = index + 1; music first when index % 3 == 0
        want_later = 1 if (t.labels["stress"] == 1) == music_first else -1
        assert t.labels["later"] == want_later, (t.trial_id, t.labels)
    # identical start times are refused (that is how the duplicated upload was found)
    with open(os.path.join(root, folders["R"], datasets.STRESS_PAIRS[0][1] + ".edf"), "r+b") as fh:
        fh.seek(168)
        fh.write(b"07.06.2212.10.00")
    try:
        datasets.load_stress_insight(root)
        raise AssertionError("same start time must be refused")
    except ValueError:
        pass
    with open(os.path.join(root, folders["R"], datasets.STRESS_PAIRS[0][1] + ".edf"), "r+b") as fh:
        fh.seek(168)
        fh.write(b"07.06.2212.20.00")
    # depression / anxiety: EEGData 38400 x 14 in Emotiv order
    dep = os.path.join(d, "dep")
    os.makedirs(dep)
    for name in ("DSub1.mat", "DSub2.mat", "DCSub1.mat", "notes.mat"):
        savemat(os.path.join(dep, name), {"EEGData": rng.normal(4200, 20, (38400, 14))})
    dt = datasets.load_depression_phq9(dep)
    assert sorted((t.subject, t.labels["depression"]) for t in dt) == [("depression_c01", -1), ("depression_p01", 1), ("depression_p02", 1)]
    ins = datasets.to_device(dt, "emotiv_insight")
    assert [c.upper() for c in ins[0].channels] == ["AF3", "T7", "T8", "AF4"], ins[0].channels   # no stand-in for Pz
    # sleepiness: table + EDFs with the two channel-naming schemes, EOG/EMG and reference channels left out
    sl = os.path.join(d, "sleep")
    os.makedirs(sl)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["participant code", "Age", "ESS", "SSS", "VASS"])
    ws.append(["AA", 20, 5, 2, 6.5])
    ws.append(["BB", 21, 8, 3, 1.0])
    wb.save(os.path.join(sl, "49 descriptive table.xlsx"))
    write_edf(os.path.join(sl, "AA.EDF"), rng.normal(0, 10, (5, 250 * 20)), 250, ["EEG Fp1-A1", "EEG Pz-A1", "EEG A2-A1", "EOG EOG_A1", "EMG EMG"])
    write_edf(os.path.join(sl, "BB.edf"), rng.normal(0, 10, (4, 250 * 20)), 250, ["Fp1", "Pz", "LEOG", "VEOG"])
    sp = datasets.load_sleepiness(sl)
    assert [(t.subject, t.labels["sleepiness"], t.channels) for t in sp] == [
        ("sleep_aa", 1, ["EEG Fp1-A1", "EEG Pz-A1"]), ("sleep_bb", -1, ["Fp1", "Pz"])], [(t.subject, t.channels) for t in sp]
    assert datasets.SLEEPINESS_SCORES["sleep_bb"]["VASS"] == 1.0
    # a model with valence/arousal heads from one dataset and a stress head from another
    os.makedirs(os.path.join(d, "g2"))
    g = make_gameemo(os.path.join(d, "g2"), rng)
    out = os.path.join(d, "insight.json")
    m = train.main(["--data", f"gameemo:{g}", "--data", f"stress_insight:{root}", "--targets", "valence,arousal,stress",
                    "--cv", "none", "--model", "brain", "--out", out])
    assert set(m["heads"]) == {"valence", "arousal", "stress"} and m["heads"]["stress"]["classes"] == [-1, 1], list(m["heads"])
    check_export(m)
    print("state loaders OK: stress pairs and order labels, duplicate start refused, PHQ/GAD mats, sleepiness table + EDF "
          "naming, Insight subset, 3-head model (valence, arousal, stress)")


def personal_states_pipeline(d, rng):
    """Recorded calibration sessions -> research/personal_states.py: loader by state, per-person calibration,
    group statistics and the participant's ratings."""
    import json as _json
    sys.path.insert(0, os.path.join(ROOT, "research"))
    import personal_states as PS
    from emotif.recorder import Recorder
    root = os.path.join(d, "cal_sessions")
    os.makedirs(root)
    chans, fs = datasets.INSIGHT_CHANNELS, 128.0
    va = {"calm": (1, -1), "stress": (-1, 1), "anxiety": (-1, 0.3), "anger": (-0.3, 1), "low_mood": (-1, -1), "drowsy": (0.2, -0.3)}
    for p in ("C01", "C02"):
        rec = Recorder(root, p, fs, chans, "test")
        rec.samples(synth(chans, fs, 20, 0, 0, rng).T.tolist())
        for rnd in range(3):
            for st in datasets.PERSONAL_STATES:
                blk = f"{st}_{rnd + 1}"
                rec.event("block_start", _json.dumps({"block": blk, "task": "x", "design": {"v": 0, "a": 0}, "state": st}))
                rec.samples(synth(chans, fs, 14, *va[st], rng).T.tolist())
                rec.event("block_end", _json.dumps({"block": blk}))
                rec.event("intensity", _json.dumps({"block": blk, "value": 7 if st == "stress" else 2}))
        rec.close()
        os.rename(rec.dir, os.path.join(root, f"{rec.id}_{p}"))
    tr = datasets.load_sessions(root, labels="state")
    assert len(tr) == 36 and sorted({t.labels["state"] for t in tr}) == list(range(6)), (len(tr), {t.labels.get("state") for t in tr})
    res = PS.main(["--sessions", root, "--out", os.path.join(d, "ps_out"), "--perms", "19"])
    assert res["n"] == 2 and set(res["by_state"]) == set(datasets.PERSONAL_STATES)
    for subj, r in res["participants"].items():
        assert r["blocks"] == 18 and 0 <= r["bacc"] <= 1 and r["checks"]["calm"]["blocks"] == 3
    assert max(r["checks"]["calm"]["auc"] for r in res["participants"].values()) > 0.6, res["participants"]
    assert all(v["stress"] == 7 for v in res["ratings"].values()), res["ratings"]
    print("personal states pipeline OK: state-labelled sessions, per-person calibration with block permutations, group table, ratings")


def session_qa_check(d, rng):
    """research/session_qa.py on a recorded headset check: a good recording passes all three checks, a recording
    without alpha reactivity fails the first."""
    import json as _json
    sys.path.insert(0, os.path.join(ROOT, "research"))
    import session_qa as QA
    from emotif.recorder import Recorder
    chans, fs = datasets.INSIGHT_CHANNELS, 128.0
    out = {}
    for name, ec_gain in (("good", 3.0), ("flat_alpha", 1.0), ("face_only", None)):
        root = os.path.join(d, "qa_" + name)
        os.makedirs(root)
        rec = Recorder(root, "Q01", fs, chans, "test")
        t = np.arange(int(60 * fs)) / fs
        def seg(alpha, blinks=False, clench=False, sec=60):
            n = int(sec * fs); tt = t[:n]
            x = rng.normal(0, 6, (5, n)) + alpha * np.sin(2 * np.pi * 10 * tt)[None] * np.array([0.6, 0.8, 1.0, 0.8, 0.6])[:, None]
            if blinks:
                for b in np.arange(1, sec, 1.0):
                    x[[0, 4]] += 140 * np.exp(-((tt - b) ** 2) / (2 * 0.06 ** 2))[None]
            if clench:
                on = (np.floor(tt / 2) % 2 == 0)
                x[[1, 3]] += 40 * rng.normal(0, 1, (2, n)) * on[None]
            return x
        rec.samples(seg(8).T.tolist()[: int(30 * fs)])
        blocks = ((("eyes_open", seg(8)), ("eyes_closed", seg(8 * ec_gain))) if ec_gain else (("eyes_open", seg(8, sec=20)),)) + \
            (("blinks", seg(8, blinks=True, sec=20)), ("clench", seg(8, clench=True, sec=20)))
        for blk, x in blocks:
            rec.event("block_start", _json.dumps({"block": blk, "task": "x", "design": {"v": 0, "a": 0}}))
            rec.samples(x.T.tolist())
            rec.event("block_end", _json.dumps({"block": blk}))
        rec.close()
        out[name] = QA.qa(rec.dir)
    g, f = out["good"], out["flat_alpha"]
    assert g["ready"] and all(c["pass"] for c in g["checks"].values()), g["checks"]
    assert not f["checks"]["alpha_reactivity"]["pass"] and not f["ready"], f["checks"]["alpha_reactivity"]
    assert g["checks"]["alpha_reactivity"]["best_posterior"] > 2.5
    fo = out["face_only"]   # the 1-minute "Blink and jaw check": no eyes-closed minute, ready on the face checks
    assert "alpha_reactivity" not in fo["checks"] and fo["ready"], fo["checks"]
    assert "| Alpha eyes closed / open | not run |" in QA.markdown(fo)
    print(f"session QA OK: good recording passes (alpha x{g['checks']['alpha_reactivity']['best_posterior']}, blink x{g['checks']['blink_index']['best_frontal']}, "
          f"jaw x{g['checks']['jaw_index']['best_temporal']}); no alpha reactivity fails; blink-and-jaw-only check ready")


def main():
    warnings.filterwarnings("ignore")
    brain_known_answer()
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as d:
        paths = {"seed": make_seed(d, rng), "deap": make_deap(d, rng), "dreamer": make_dreamer(d, rng),
                 "gameemo": make_gameemo(d, rng), "csv": make_csv(d, rng), "session": make_sessions(d, rng)}
        assert len(datasets.load_dreamer(paths["dreamer"])) == 12
        g = datasets.load_gameemo(paths["gameemo"])
        assert len(g) == 8 and g[0].labels == {"valence": -1, "arousal": -1} and g[0].x.shape[0] == 14
        assert len(datasets.load_csv_dir(paths["csv"])) == 12
        st = datasets.load_sessions(paths["session"])
        assert len(st) == 16 and st[1].labels == {"valence": 1, "arousal": -1}, (len(st), st[1].labels)
        assert datasets.load_sessions(paths["session"], labels="design")[2].labels == {"valence": -1, "arousal": 1}
        for name, p in paths.items():
            out = os.path.join(d, f"{name}.json")
            m = train.main(["--data", f"{name}:{p}", "--out", out, "--folds", "3"])
            check_export(m)
            for target, cv in m["cv"].items():
                # GAMEEMO has one recording per class per subject, so trial-wise within-subject CV can't
                # separate class from recording; LOSO is the meaningful test there
                best = max(s["bacc"] for mode in ("within", "loso") for s in cv.get(mode, {}).values())
                assert best > cv["chance_bacc"] + 0.15, (name, target, cv)
        state_loaders(d, rng)
        personal_states_pipeline(d, rng)
        session_qa_check(d, rng)
        # brain-only model end to end through train.py: face indices, folded export, hold threshold
        m = train.main(["--data", f"dreamer:{paths['dreamer']}", "--out", os.path.join(d, "brain.json"), "--model", "brain", "--cv", "loso"])
        check_export(m)
        assert m["cv"]["valence"]["loso"]["full"]["bacc"] > m["cv"]["valence"]["chance_bacc"] + 0.15, m["cv"]["valence"]
        # one model across three headsets/datasets (Emotiv 14, DEAP 32, Muse 4)
        m = train.main(["--data", f"dreamer:{paths['dreamer']}", "--data", f"deap:{paths['deap']}",
                        "--data", f"csv:{paths['csv']}", "--out", os.path.join(d, "mixed.json"), "--model", "mlp", "--cv", "loso"])
        check_export(m)
        assert set(m["datasets"]) == {"dreamer", "deap", "csv"} and m["n_subjects"] == 6
        assert m["cv"]["valence"]["loso"]["full"]["bacc"] > m["cv"]["valence"]["chance_bacc"] + 0.1, m["cv"]["valence"]
    bird = os.path.join(ROOT, "data", "external", "bird_muse")
    if os.path.isdir(bird):
        m = train.main(["--data", f"muse_mental:{bird}", "--out", os.path.join(tempfile.gettempdir(), "bird.json"), "--cv", "loso"])
        assert m["cv"]["arousal"]["loso"]["full"]["bacc"] > 0.5, m["cv"]["arousal"]
        print(f"real Muse data: LOSO balanced accuracy {m['cv']['arousal']['loso']['full']['bacc']:.3f} (chance 0.333)")
    print("training pipeline OK: seed, deap, dreamer, gameemo, csv, recorded sessions, mixed-headset model")


if __name__ == "__main__":
    main()
