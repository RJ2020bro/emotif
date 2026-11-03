"""EDF / EDF+ / BDF reader and writer (no dependencies beyond numpy). app/src/edf.ts mirrors read_edf.

read_edf keeps voltage signals (uV, mV, V) at the most common sampling rate and returns microvolts; annotation
and status channels and signals at other rates are listed in `skipped`. write_edf exists for tests and for
exporting Emotif recordings to other tools.
"""
import numpy as np

UNIT_TO_UV = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}


def _ascii(b, off, n):
    return b[off:off + n].decode("latin-1").strip()


def _num(b, off, n, what):
    try:
        return float(_ascii(b, off, n))
    except ValueError:
        raise ValueError(f"EDF header: {what} is not a number") from None


def read_edf(path_or_bytes):
    """Returns dict(fs, channels, x (C, N) in uV, skipped, format, duration_sec)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        b = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as fh:
            b = fh.read()
    if len(b) < 256:
        raise ValueError("file is too short to be EDF/BDF")
    bdf = b[0] == 255 and _ascii(b, 1, 7) == "BIOSEMI"
    if not bdf and _ascii(b, 0, 8) != "0":
        raise ValueError("not an EDF or BDF file (bad version field)")
    nbytes = 3 if bdf else 2
    header = int(_num(b, 184, 8, "header size"))
    nrec = int(_num(b, 236, 8, "number of records"))
    rec_dur = _num(b, 244, 8, "record duration")
    ns = int(_num(b, 252, 4, "number of signals"))
    if ns <= 0 or header != 256 * (ns + 1):
        raise ValueError("EDF header: signal count and header size disagree")
    if rec_dur <= 0:
        raise ValueError("EDF records with zero duration are not supported")

    def field(start, width, i):
        return 256 + start * ns + i * width

    sig = [{"label": _ascii(b, field(0, 16, i), 16), "unit": _ascii(b, field(96, 8, i), 8),
            "pmin": _num(b, field(104, 8, i), 8, "physical minimum"), "pmax": _num(b, field(112, 8, i), 8, "physical maximum"),
            "dmin": _num(b, field(120, 8, i), 8, "digital minimum"), "dmax": _num(b, field(128, 8, i), 8, "digital maximum"),
            "spr": int(_num(b, field(216, 8, i), 8, "samples per record"))} for i in range(ns)]
    rec_samples = sum(s["spr"] for s in sig)
    avail = (len(b) - header) // (rec_samples * nbytes)
    if nrec < 0 or nrec > avail:
        nrec = avail
    if nrec < 1:
        raise ValueError("EDF file has no complete data records")

    def is_annot(label):
        return label.lower() in ("edf annotations", "bdf annotations", "status")

    def scale(unit):
        return UNIT_TO_UV.get(unit.lower())

    counts = {}
    for s in sig:
        if not is_annot(s["label"]) and scale(s["unit"]) is not None and s["spr"] > 0:
            counts[s["spr"]] = counts.get(s["spr"], 0) + 1
    if not counts:
        raise ValueError("no voltage signals (uV, mV or V) in this file")
    spr = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]
    keep = [i for i, s in enumerate(sig) if not is_annot(s["label"]) and scale(s["unit"]) is not None and s["spr"] == spr]
    skipped = [s["label"] for i, s in enumerate(sig) if i not in keep and "annotations" not in s["label"].lower()]
    raw = np.frombuffer(b, dtype=np.uint8, count=nrec * rec_samples * nbytes, offset=header)
    raw = raw.reshape(nrec, rec_samples, nbytes).astype(np.int32)
    if nbytes == 2:
        d = raw[..., 0] | (raw[..., 1] << 8)
        d = np.where(d & 0x8000, d - 0x10000, d)
    else:
        d = raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16)
        d = np.where(d & 0x800000, d - 0x1000000, d)
    offsets = np.concatenate([[0], np.cumsum([s["spr"] for s in sig])[:-1]])
    x = np.empty((len(keep), nrec * spr))
    for c, i in enumerate(keep):
        s = sig[i]
        if s["dmax"] == s["dmin"]:
            raise ValueError(f"signal {s['label']}: digital minimum equals maximum")
        g = (s["pmax"] - s["pmin"]) / (s["dmax"] - s["dmin"]) * scale(s["unit"])
        x[c] = ((d[:, offsets[i]:offsets[i] + spr] - s["dmin"]) * g + s["pmin"] * scale(s["unit"])).reshape(-1)
    return {"fs": spr / rec_dur, "channels": [sig[i]["label"] for i in keep], "x": x, "skipped": skipped,
            "format": "BDF" if bdf else "EDF", "duration_sec": nrec * rec_dur}


def write_edf(path, x_uv, fs, channels, bdf=False, record_sec=1.0, units=None, extra=None):
    """Write (C, N) microvolt data as EDF (16-bit) or BDF (24-bit). Samples are quantised to the digital range
    over each channel's own min..max. extra: optional list of (label, unit, (n,) values, spr) signals at another
    rate, for testing. units: optional per-channel unit strings ("uV", "mV", "V")."""
    x_uv = np.asarray(x_uv, dtype=np.float64)
    C, N = x_uv.shape
    spr = int(round(fs * record_sec))
    if abs(spr - fs * record_sec) > 1e-9:
        raise ValueError("fs * record_sec must be an integer")
    nrec = N // spr
    units = units or ["uV"] * C
    dmax = 8388607 if bdf else 32767
    dmin = -dmax - 1

    def bound(v, up):
        """Round v outward to the most decimals that fit the 8-character header field; returns the string."""
        for k in range(7, -1, -1):
            q = (np.ceil if up else np.floor)(v * 10 ** k) / 10 ** k
            txt = f"{q:.{k}f}"
            if len(txt) <= 8:
                return txt
        raise ValueError(f"value {v} does not fit an EDF header field")

    sigs = []
    for c in range(C):
        f = {"uV": 1.0, "mV": 1e-3, "V": 1e-6}[units[c]]
        v = x_uv[c, :nrec * spr] * f
        lo, hi = bound(float(v.min()), False), bound(float(v.max()), True)
        if float(hi) <= float(lo):
            hi = bound(float(lo) + 1.0, True)
        sigs.append((channels[c], units[c], lo, hi, v, spr))
    for label, unit, vals, s_spr in (extra or []):
        vals = np.asarray(vals, dtype=np.float64)[:nrec * s_spr]
        sigs.append((label, unit, bound(float(vals.min()), False), bound(float(vals.max()) + 1.0, True), vals, s_spr))
    ns = len(sigs)

    def pad(s, n):
        s = str(s)
        if len(s) > n:
            raise ValueError(f"header field {s!r} longer than {n}")
        return s.ljust(n).encode("latin-1")

    h = bytearray()
    h += (b"\xffBIOSEMI" if bdf else pad("0", 8))
    h += pad("X X X X", 80) + pad("Startdate X X X X", 80) + pad("01.01.26", 8) + pad("00.00.00", 8)
    h += pad(256 * (ns + 1), 8) + pad("24BIT" if bdf else "", 44) + pad(nrec, 8) + pad(f"{record_sec:g}", 8) + pad(ns, 4)
    h += b"".join(pad(s[0], 16) for s in sigs)
    h += b"".join(pad("AgAgCl electrode", 80) for _ in sigs)
    h += b"".join(pad(s[1], 8) for s in sigs)
    h += b"".join(pad(s[2], 8) for s in sigs)
    h += b"".join(pad(s[3], 8) for s in sigs)
    h += b"".join(pad(dmin, 8) for _ in sigs)
    h += b"".join(pad(dmax, 8) for _ in sigs)
    h += b"".join(pad("HP:0.1Hz", 80) for _ in sigs)
    h += b"".join(pad(s[5], 8) for s in sigs)
    h += b"".join(pad("", 32) for _ in sigs)
    assert len(h) == 256 * (ns + 1)
    digs = []
    for (_, _, lo, hi, v, s_spr) in sigs:
        pmin, pmax = float(lo), float(hi)
        d = np.round((v - pmin) / (pmax - pmin) * (dmax - dmin) + dmin).astype(np.int64)
        digs.append(np.clip(d, dmin, dmax).reshape(nrec, s_spr))
    body = np.concatenate(digs, axis=1).reshape(-1)
    nb = 3 if bdf else 2
    u = (body & ((1 << (8 * nb)) - 1)).astype(np.uint32)
    out = np.stack([(u >> (8 * k)) & 0xFF for k in range(nb)], axis=1).astype(np.uint8).reshape(-1)
    with open(path, "wb") as fh:
        fh.write(bytes(h) + out.tobytes())
    return nrec * spr
