"""Dataset loaders. Each returns a list of Trial: one continuous recording with constant labels.

Labels are on a common scale: valence / arousal in {-1, 0, +1}.

  seed         SJTU SEED, Preprocessed_EEG/*.mat + label.mat                 62 ch, 200 Hz, valence -1/0/1
  deap         DEAP data_preprocessed_python/sXX.dat                          32 ch, 128 Hz, rating > 5 -> +1 else -1
  dreamer      DREAMER.mat                                                    14 ch (Emotiv), 128 Hz, rating > 3 -> +1 else -1
  gameemo      GAMEEMO .../SxxGyAllChannels.csv                               14 ch (Emotiv), 128 Hz, game quadrant
  muse_mental  Bird et al. 2018 Muse recordings (relaxed/neutral/concentrating)  4 ch, 256 Hz, arousal proxy
  csv          recordings exported by the Emotif app                          any montage
"""
import csv
import glob
import os
import pickle
import re
from dataclasses import dataclass, field

import numpy as np

SEED_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7",
    "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6",
    "PO8", "CB1", "O1", "Oz", "O2", "CB2"]
DEAP_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1",
    "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2",
    "P4", "P8", "PO4", "O2"]
EMOTIV_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# GAMEEMO games (Alakus et al. 2020): G1 boring (LANV), G2 calm (LAPV), G3 horror (HANV), G4 funny (HAPV)
GAMEEMO_VA = {1: (-1, -1), 2: (1, -1), 3: (-1, 1), 4: (1, 1)}
MUSE_MENTAL_AROUSAL = {"relaxed": -1, "neutral": 0, "concentrating": 1}

def _walk_csv(root):
    """All .csv files under root, including inside dot-folders (GAMEEMO ships '.csv format/')."""
    out = []
    for d, _, files in os.walk(root):
        out.extend(os.path.join(d, f) for f in files if f.lower().endswith(".csv"))
    return sorted(out)


TIME_COLS = {"time", "time_s", "timestamp", "timestamps", "t"}
LABEL_COLS = {"valence_label": "valence", "arousal_label": "arousal"}


@dataclass
class Trial:
    subject: str
    trial_id: str
    x: np.ndarray          # (C, T) µV
    fs: float
    channels: list
    labels: dict = field(default_factory=dict)   # {"valence": -1|0|1, "arousal": -1|0|1}
    filtered: bool = False                       # True once features.filter_signal has been applied
    dataset: str = ""
    group: str = ""                              # CV group; pieces of one recording share it (default trial_id)


def load_seed(root):
    from scipy.io import loadmat
    d = os.path.join(root, "Preprocessed_EEG") if os.path.isdir(os.path.join(root, "Preprocessed_EEG")) else root
    label_path = os.path.join(d, "label.mat")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"label.mat not found in {d}")
    labels = np.asarray(loadmat(label_path)["label"]).reshape(-1).astype(int)
    trials = []
    files = sorted(f for f in glob.glob(os.path.join(d, "*.mat")) if os.path.basename(f) != "label.mat")
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        subject = "seed" + base.split("_")[0]
        mat = loadmat(f)
        keyed = sorted((int(m.group(1)), k) for k in mat for m in [re.match(r"^.*_eeg(\d+)$", k)] if m)
        for num, k in keyed:
            x = np.asarray(mat[k], dtype=np.float64)
            if x.shape[0] != len(SEED_CHANNELS):
                raise ValueError(f"{f}:{k} has {x.shape[0]} channels, expected 62")
            trials.append(Trial(subject, f"{base}:{num}", x, 200.0, list(SEED_CHANNELS),
                                {"valence": int(labels[num - 1])}, dataset="seed"))
    if not trials:
        raise FileNotFoundError(f"no SEED .mat files in {d}")
    return trials


def load_deap(root):
    files = sorted(glob.glob(os.path.join(root, "s*.dat")))
    if not files:
        raise FileNotFoundError(f"no sXX.dat files in {root}")
    trials = []
    for f in files:
        subject = "deap" + os.path.splitext(os.path.basename(f))[0]
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        data, lab = np.asarray(d["data"]), np.asarray(d["labels"])
        for t in range(data.shape[0]):
            x = data[t, :32, 3 * 128:].astype(np.float64)
            trials.append(Trial(subject, f"{subject}:{t}", x, 128.0, list(DEAP_CHANNELS),
                                {"valence": 1 if lab[t, 0] > 5 else -1, "arousal": 1 if lab[t, 1] > 5 else -1},
                                dataset="deap"))
    return trials


def _field(obj, name, default=None):
    """Read a field from a scipy.io mat_struct; default if absent."""
    return getattr(obj, name, default)


def load_dreamer(root):
    """DREAMER.mat (Katsigiannis & Ramzan 2018). 23 subjects x 18 film clips, Emotiv EPOC 128 Hz."""
    from scipy.io import loadmat
    path = root if root.lower().endswith(".mat") else os.path.join(root, "DREAMER.mat")
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    D = m["DREAMER"]
    data = np.atleast_1d(_field(D, "Data"))
    fs = float(_field(D, "EEG_SamplingRate", 128.0))
    channels = [str(e) for e in np.atleast_1d(_field(D, "EEG_Electrodes", EMOTIV_CHANNELS))]
    trials = []
    for si, subj in enumerate(data):
        eeg = _field(subj, "EEG")
        stim = np.atleast_1d(_field(eeg, "stimuli"))
        val = np.atleast_1d(_field(subj, "ScoreValence")).astype(float)
        aro = np.atleast_1d(_field(subj, "ScoreArousal")).astype(float)
        sid = f"dreamer{si + 1:02d}"
        for k, x in enumerate(stim):
            x = np.asarray(x, dtype=np.float64)
            if x.ndim != 2:
                raise ValueError(f"{path}: subject {si + 1} clip {k + 1} is not 2-D")
            if x.shape[1] == len(channels) and x.shape[0] != len(channels):
                x = x.T
            trials.append(Trial(sid, f"{sid}:{k}", x, fs, list(channels),
                                {"valence": 1 if val[k] > 3 else -1, "arousal": 1 if aro[k] > 3 else -1},
                                dataset="dreamer"))
    if not trials:
        raise ValueError(f"no trials in {path}")
    return trials


def load_gameemo(root, prefer="preprocessed"):
    """GAMEEMO (Alakus et al. 2020, CC BY 4.0). Finds SxxGyAllChannels.csv (preprocessed) or
    SxxGyAllRawChannels.csv anywhere under root. 5 min per game, 14 Emotiv channels, 128 Hz."""
    pats = {"preprocessed": re.compile(r"^S(\d+)G([1-4])AllChannels\.csv$", re.I),
            "raw": re.compile(r"^S(\d+)G([1-4])AllRawChannels\.csv$", re.I)}
    found = {}
    for f in _walk_csv(root):
        for kind, pat in pats.items():
            m = pat.match(os.path.basename(f))
            if m:
                found.setdefault((int(m.group(1)), int(m.group(2))), {})[kind] = f
    if not found:
        raise FileNotFoundError(f"no GAMEEMO SxxGy*.csv under {root}")
    wanted = {c.upper() for c in EMOTIV_CHANNELS}
    trials = []
    for (s, g), files in sorted(found.items()):
        f = files.get(prefer) or next(iter(files.values()))
        with open(f, newline="") as fh:
            rows = list(csv.reader(fh))
        header = [h.strip().strip('"') for h in rows[0]]
        idx = [i for i, h in enumerate(header) if h.upper() in wanted]
        if len(idx) < 4:
            raise ValueError(f"{f}: expected Emotiv channel columns, got {header}")
        body = []
        for r in rows[1:]:
            if len(r) < len(header):
                continue
            try:
                body.append([float(r[i]) for i in idx])
            except ValueError:
                continue
        v, a = GAMEEMO_VA[g]
        trials.append(Trial(f"gameemo{s:02d}", f"S{s:02d}G{g}", np.array(body).T, 128.0, [header[i] for i in idx],
                            {"valence": v, "arousal": a}, dataset="gameemo"))
    return trials


def load_muse_mental(root):
    """Bird et al. 2018 Muse recordings: subject<x>-<relaxed|neutral|concentrating>-<n>.csv (muselsl format)."""
    pat = re.compile(r"^subject([a-z])-(relaxed|neutral|concentrating)-(\d+)\.csv$")
    trials = []
    for f in _walk_csv(root):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        with open(f, newline="") as fh:
            rows = list(csv.reader(fh))
        header = [h.strip() for h in rows[0]]
        cols = [header.index("timestamps")] + [header.index(c) for c in MUSE_CHANNELS]
        body = np.array([[float(r[i]) for i in cols] for r in rows[1:] if len(r) == len(header)])
        t = body[:, 0]
        dt = np.diff(t)
        fs = float(1.0 / np.median(dt))
        # some files contain recording pauses; split there so no window spans a gap
        cuts = [0] + (np.flatnonzero((dt > 5.0 / fs) | (dt <= 0)) + 1).tolist() + [len(t)]
        base = os.path.basename(f)
        for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            if b - a < 2 * fs:
                continue
            trials.append(Trial(f"muse_{m.group(1)}", f"{base}#{k}", body[a:b, 1:].T, fs, list(MUSE_CHANNELS),
                                {"arousal": MUSE_MENTAL_AROUSAL[m.group(2)]}, dataset="muse_mental", group=base))
    if not trials:
        raise FileNotFoundError(f"no subject?-state-n.csv files under {root}")
    return trials


def _parse_label(s):
    s = s.strip()
    if s == "":
        return None
    v = int(float(s))
    if v not in (-1, 0, 1):
        raise ValueError(f"label {s!r} must be -1, 0 or 1")
    return v


def load_csv_file(path, fs=None):
    """Emotif recording: time_s, <channels...>, valence_label, arousal_label[, pred_*, event].
    The whole file is filtered once (as in the live app); each run of identical labels becomes a Trial."""
    from .features import filter_signal
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"{path}: empty")
    header = [h.strip() for h in rows[0]]
    tcol = next((i for i, h in enumerate(header) if h.lower() in TIME_COLS), None)
    lcols = {i: LABEL_COLS[h.lower()] for i, h in enumerate(header) if h.lower() in LABEL_COLS}
    chan_idx = [i for i in range(len(header))
                if i != tcol and i not in lcols and not header[i].lower().startswith(("pred_", "event", "marker"))]
    if not chan_idx:
        raise ValueError(f"{path}: no channel columns")
    body = [r for r in rows[1:] if len(r) == len(header)]
    x = np.array([[float(r[i]) for i in chan_idx] for r in body]).T
    if fs is None:
        if tcol is None:
            raise ValueError(f"{path}: no time column; pass fs")
        t = np.array([float(r[tcol]) for r in body])
        if not np.all(np.diff(t) > 0):
            raise ValueError(f"{path}: time column is not strictly increasing")
        fs = float((len(t) - 1) / (t[-1] - t[0]))
    x = filter_signal(x, fs)
    names = [lcols[i] for i in sorted(lcols)]
    labs = [tuple(_parse_label(r[i]) for i in sorted(lcols)) for r in body]
    subject = os.path.splitext(os.path.basename(path))[0]
    trials, start = [], 0
    for k in range(1, len(labs) + 1):
        if k == len(labs) or labs[k] != labs[start]:
            lab = {n: v for n, v in zip(names, labs[start]) if v is not None}
            if lab:
                trials.append(Trial(subject, f"{subject}:{start}", x[:, start:k], fs,
                                    [header[i] for i in chan_idx], lab, filtered=True, dataset="csv"))
            start = k
    return trials


def load_csv_dir(path, fs=None):
    files = [path] if os.path.isfile(path) else _walk_csv(path)
    if not files:
        raise FileNotFoundError(f"no .csv in {path}")
    out = []
    for f in files:
        out.extend(load_csv_file(f, fs))
    return out


def sam_class(x):
    """SAM 1-9 -> -1 / 0 / +1 (same rule as the app)."""
    x = float(x)
    return -1 if x <= 3 else (1 if x >= 7 else 0)


PERSONAL_STATES = ["calm", "stress", "anxiety", "anger", "low_mood", "drowsy"]     # app/src/states.ts STATES


def load_sessions(root, labels="sam"):
    """Sessions recorded by the Emotif server (folders with meta.json, eeg.csv, events.csv).
    Each protocol block becomes a Trial labelled by the participant's SAM rating (labels="sam"), by
    the block's intended valence/arousal (labels="design"), or by its calibration state (labels="state"). Each session is normalised on its own, as in the app."""
    import json as _json
    from .features import filter_signal
    dirs = [root] if os.path.isfile(os.path.join(root, "meta.json")) else \
        sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "meta.json")))
    trials = []
    for d in dirs:
        with open(os.path.join(d, "meta.json")) as fh:
            meta = _json.load(fh)
        with open(os.path.join(d, "eeg.csv"), newline="") as fh:
            rows = list(csv.reader(fh))
        header = rows[0]
        chans = [h for h in header[1:] if h not in LABEL_COLS]
        ci = [header.index(c) for c in chans]
        body = [r for r in rows[1:] if len(r) == len(header)]
        if len(body) < 2:
            continue
        x = filter_signal(np.array([[float(r[i]) for i in ci] for r in body]).T, float(meta["fs"]))
        fs = float(meta["fs"])
        starts, ends, sams = {}, {}, {}
        with open(os.path.join(d, "events.csv"), newline="") as fh:
            for r in list(csv.reader(fh))[1:]:
                if len(r) < 3:
                    continue
                t, kind, val = float(r[0]), r[1], r[2]
                try:
                    v = _json.loads(val) if val else {}
                except ValueError:
                    continue
                if not isinstance(v, dict) or "block" not in v:
                    continue
                if kind == "block_start":
                    starts[v["block"]] = (t, v)
                elif kind == "block_end":
                    ends[v["block"]] = t
                elif kind == "sam":
                    sams[v["block"]] = v
        for b, (t0, info) in starts.items():
            if b not in ends or info.get("task") == "rest":
                continue
            if labels == "state":          # personal state calibration: label = index in PERSONAL_STATES
                st = info.get("state")
                if st not in PERSONAL_STATES:
                    continue
                lab = {"state": PERSONAL_STATES.index(st)}
            elif labels == "sam":
                if b not in sams:
                    continue
                lab = {"valence": sam_class(sams[b]["valence"]), "arousal": sam_class(sams[b]["arousal"])}
            else:
                des = info.get("design", {})
                lab = {"valence": int(np.sign(des.get("v", 0))), "arousal": int(np.sign(des.get("a", 0)))}
            a, z = int(round(t0 * fs)), int(round(ends[b] * fs))
            if z - a < 2 * fs:
                continue
            trials.append(Trial(f"{meta.get('participant', 'anon')}@{meta['id']}", f"{meta['id']}:{b}", x[:, a:z], fs, chans,
                                lab, filtered=True, dataset="session", group=f"{meta['id']}:{b}"))
    if not trials:
        raise FileNotFoundError(f"no rated protocol blocks in sessions under {root}")
    return trials


def load_gameemo_raw(root):
    """GAMEEMO 'raw' export (DC removed by the headset software, not filtered or artifact-cleaned), so the
    cleaning pipeline under test does all of the cleaning."""
    return load_gameemo(root, prefer="raw")


# --- State and trait datasets (Mendeley Data; IDs and licences in docs/DATASETS.md) ---------------------------------

# Stress (Mendeley wnshbvdxs2, CC BY 4.0), recorded with the 5-channel Emotiv Insight. Only the horror-video and
# relaxing-music folders were recorded with the same people on the same days, so only they give a within-person
# contrast. Pairs (horror file, relaxing-music file) were matched from the EDF headers' recording field and start
# times; the names are not stored here. "002 (5)" / "001 (3)" are one recording uploaded twice and are left out.
# The last two pairs are matched by date, start time (8-12 min apart) and initial only.
STRESS_PAIRS = [("002 (10)", "001 (8)"), ("002 (11)", "001 (9)"), ("002 (12)", "001 (10)"), ("002 (14)", "001 (12)"),
                ("002 (15)", "001 (13)"), ("002 (16)", "001 (14)"), ("002 (17)", "001 (4)"), ("002 (18)", "001 (15)"),
                ("002 (19)", "001 (16)"), ("002 (2)", "001 (19)"), ("002 (20)", "001 (17)"), ("002 (21)", "001 (18)"),
                ("002 (3)", "001 (1)"), ("002 (4)", "001 (2)"), ("002 (9)", "001 (7)"), ("002 (7)", "001 (5)"),
                ("002 (8)", "001 (6)")]
INSIGHT_CHANNELS = ["AF3", "T7", "Pz", "T8", "AF4"]


def _find_file(root, stem, exts):
    for dirpath, _, files in os.walk(root):
        for f in files:
            b, e = os.path.splitext(f)
            if b == stem and e.lower() in exts:
                return os.path.join(dirpath, f)
    return None


def load_stress_insight(root):
    """Acute stress (horror video, label stress=+1) vs relaxation (relaxing music, stress=-1), 17 people, Emotiv
    Insight (AF3, T7, Pz, T8, AF4), 128 Hz. Each person contributes one recording of each. Label "later" marks the
    recording made second, for a control analysis (does a decoder just track time?)."""
    import datetime
    from .edf import read_edf
    trials = []
    for k, (h, r) in enumerate(STRESS_PAIRS, 1):
        pair = []
        for stem, lab, folder in ((h, 1, "Horrer Video Stimulation"), (r, -1, "Participants Listening to Relaxing Music")):
            f = _find_file(os.path.join(root, folder), stem, {".edf"})
            if f is None:
                raise FileNotFoundError(f"{folder}/{stem}.edf not found under {root}")
            with open(f, "rb") as fh:
                head = fh.read(184).decode("ascii", "replace")
            start = datetime.datetime.strptime(head[168:184], "%d.%m.%y%H.%M.%S")
            d = read_edf(f)
            idx = [d["channels"].index(c) for c in INSIGHT_CHANNELS]
            pair.append((start, Trial(f"stress{k:02d}", f"stress{k:02d}_{'horror' if lab > 0 else 'relax'}", d["x"][idx],
                                      float(d["fs"]), list(INSIGHT_CHANNELS), {"stress": lab}, dataset="stress_insight")))
        if pair[0][0] == pair[1][0]:
            raise ValueError(f"stress pair {k}: both recordings start at {pair[0][0]}")
        later = 0 if pair[0][0] > pair[1][0] else 1
        for i, (_, t) in enumerate(pair):
            t.labels["later"] = 1 if i == later else -1       # order control: the second recording of the pair
            trials.append(t)
    return trials


def _load_epoc_mats(root, pattern, target):
    """Kabir et al. screening sets (Emotiv EPOC+, 5 min, 128 Hz, EEGData 38400 x 14). Columns are in Emotiv's export
    order (EMOTIV_CHANNELS); checked on these files: alpha relative power is highest in columns 7-8 (O1, O2)."""
    from scipy.io import loadmat
    pat = re.compile(pattern)
    trials = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            m = pat.match(f)
            if not m:
                continue
            x = np.asarray(loadmat(os.path.join(dirpath, f))["EEGData"], dtype=float)
            if x.shape[1] != 14:
                raise ValueError(f"{f}: expected 14 columns, got {x.shape}")
            lab = -1 if m.group(1) else 1
            sid = f"{target}_{'c' if m.group(1) else 'p'}{int(m.group(2)):02d}"
            trials.append(Trial(sid, sid, x.T.copy(), 128.0, list(EMOTIV_CHANNELS), {target: lab}, dataset=target))
    if not trials:
        raise FileNotFoundError(f"no {pattern} files under {root}")
    return trials


def load_depression_phq9(root):
    """Depression screening set (Mendeley ct5vrc4k2t, CC BY 4.0): PHQ-9 >= 20 (DSub, depression=+1, 18 people) vs
    PHQ-9 <= 4 (DCSub, depression=-1, 13 people)."""
    return _load_epoc_mats(root, r"^D(C)?Sub(\d+)\.mat$", "depression")


def load_anxiety_gad7(root):
    """Anxiety screening set (Mendeley rh3fy75zdv, CC BY 4.0): GAD-7 >= 15 (ASub, anxiety=+1, 23 people) vs
    GAD-7 <= 4 (ACSub, anxiety=-1, 15 people)."""
    return _load_epoc_mats(root, r"^A(C)?Sub(\d+)\.mat$", "anxiety")


VASS_CUTOFF = 5.0    # visual analogue sleepiness scale (0-10) at the recording: >= 5 (the midpoint) counts as sleepy


def load_sleepiness(root):
    """Resting-state EEG with sleepiness ratings (Mendeley wnp3jnymvn, CC BY 4.0): 49 people, 10 min, 19-channel 10-20
    referenced to A1, 250 Hz. Every volunteer scored ESS <= 10 (no excessive daytime sleepiness), so the label is the
    state rating taken at the recording: sleepiness=+1 when VASS >= 5, else -1. ESS, SSS and VASS are stored in
    SLEEPINESS_SCORES[subject]."""
    import openpyxl
    from .edf import read_edf
    from .montage import canonical
    book = None
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".xlsx") and "descriptive" in f.lower():
                book = os.path.join(dirpath, f)
    if book is None:
        raise FileNotFoundError(f"no descriptive table .xlsx under {root}")
    rows = list(openpyxl.load_workbook(book, read_only=True, data_only=True).active.iter_rows(values_only=True))
    head = [str(h).strip() for h in rows[0]]
    ic, ie, iss, iv = head.index("participant code"), head.index("ESS"), head.index("SSS"), head.index("VASS")
    trials = []
    SLEEPINESS_SCORES.clear()
    for r in rows[1:]:
        if r[ic] is None:
            continue
        code = str(r[ic]).strip()
        f = _find_file(root, code, {".edf"})
        if f is None:
            continue
        d = read_edf(f)
        keep = [i for i, c in enumerate(d["channels"])
                if not c.strip().upper().startswith(("EOG", "EMG", "ECG")) and canonical(c) not in (None, "A1", "A2", "M1", "M2")]
        sid = f"sleep_{code.lower()}"
        SLEEPINESS_SCORES[sid] = {"ESS": float(r[ie]), "SSS": float(r[iss]), "VASS": float(r[iv])}
        trials.append(Trial(sid, sid, d["x"][keep], float(d["fs"]), [d["channels"][i] for i in keep],
                            {"sleepiness": 1 if float(r[iv]) >= VASS_CUTOFF else -1}, dataset="sleepiness"))
    if not trials:
        raise FileNotFoundError(f"no EDF files matching the table under {root}")
    return trials


SLEEPINESS_SCORES = {}


def to_device(trials, device):
    """The same recordings restricted to the electrodes nearest a headset's (montage.device_subset); recordings
    with fewer than half of the headset's electrodes are dropped."""
    from .montage import device_subset
    out = []
    for t in trials:
        idx = device_subset(t.channels, device)
        if idx:
            out.append(Trial(t.subject, t.trial_id, t.x[idx], t.fs, [t.channels[i] for i in idx], dict(t.labels),
                             t.filtered, t.dataset, t.group))
    return out


LOADERS = {"seed": load_seed, "deap": load_deap, "dreamer": load_dreamer, "gameemo": load_gameemo, "gameemo_raw": load_gameemo_raw,
           "muse_mental": load_muse_mental, "csv": load_csv_dir, "session": load_sessions,
           "stress_insight": load_stress_insight, "depression_phq9": load_depression_phq9, "anxiety_gad7": load_anxiety_gad7,
           "sleepiness": load_sleepiness}
