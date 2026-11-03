"""Session recording. One folder per session:

  meta.json    participant code, device, montage, fs, protocol, start time, app version
  eeg.csv      time_s, <channels...>, valence_label, arousal_label     (raw µV, sample-clocked)
  events.csv   time_s, type, value                                      (markers, SAM ratings, arm commands)
  affect.csv   time_s, valence, arousal, confidence, label, quality      (decoder output)

eeg.csv loads directly with `train.py --data csv:<folder>`.
"""
import csv
import datetime
import json
import os
import re

VERSION = "2.0.0"


def safe_code(s):
    s = re.sub(r"[^A-Za-z0-9_-]+", "", s or "")[:32]
    return s or "anon"


class Recorder:
    def __init__(self, root, participant, fs, channels, device, protocol="", notes="", extra=None):
        if fs <= 0 or not channels:
            raise ValueError("no active stream to record")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.id = f"{stamp}_{safe_code(participant)}"
        self.dir = os.path.join(root, self.id)
        os.makedirs(self.dir, exist_ok=False)
        self.fs = float(fs)
        self.channels = list(channels)
        self.n = 0
        self.labels = ["", ""]
        self.meta = {
            "id": self.id, "participant": safe_code(participant), "device": device, "fs": self.fs,
            "channels": self.channels, "protocol": protocol, "notes": notes,
            "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "app_version": VERSION, **(extra or {}),
        }
        self._eeg = open(os.path.join(self.dir, "eeg.csv"), "w", newline="")
        self._ev = open(os.path.join(self.dir, "events.csv"), "w", newline="")
        self._af = open(os.path.join(self.dir, "affect.csv"), "w", newline="")
        self.w_eeg, self.w_ev, self.w_af = csv.writer(self._eeg), csv.writer(self._ev), csv.writer(self._af)
        self.w_eeg.writerow(["time_s", *self.channels, "valence_label", "arousal_label"])
        self.w_ev.writerow(["time_s", "type", "value"])
        self.w_af.writerow(["time_s", "valence", "arousal", "confidence", "label", "quality"])
        self._write_meta()

    def _write_meta(self):
        with open(os.path.join(self.dir, "meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=1)

    @property
    def t(self):
        return self.n / self.fs

    def samples(self, rows):
        for r in rows:
            if len(r) != len(self.channels):
                continue
            self.w_eeg.writerow([f"{self.n / self.fs:.6f}", *r, *self.labels])
            self.n += 1

    def set_labels(self, valence, arousal):
        def norm(v):
            return "" if v in (None, "") else str(int(v))
        self.labels = [norm(valence), norm(arousal)]
        self.event("self_report", json.dumps({"valence": valence, "arousal": arousal}))

    def event(self, kind, value=""):
        self.w_ev.writerow([f"{self.t:.6f}", kind, value])

    def affect(self, v, a, conf, label, quality):
        self.w_af.writerow([f"{self.t:.6f}", f"{v:.4f}", f"{a:.4f}", "" if conf is None else f"{conf:.4f}",
                            label, f"{quality:.3f}"])

    def close(self):
        self.meta["ended"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        self.meta["n_samples"] = self.n
        self.meta["duration_s"] = round(self.t, 3)
        self._write_meta()
        for f in (self._eeg, self._ev, self._af):
            f.close()


def list_sessions(root):
    out = []
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root), reverse=True):
        p = os.path.join(root, d, "meta.json")
        if os.path.isfile(p):
            try:
                with open(p) as fh:
                    out.append(json.load(fh))
            except ValueError:
                continue
    return out
