// EDF / EDF+ / BDF reader (European Data Format; BioSemi 24-bit variant). Most research and consumer
// headsets can export one of these (Emotiv, OpenBCI GUI BDF+, g.tec, BioSemi, Neuroelectrics, BrainVision
// converters), so file playback works with any of them. Mirrors emotif/edf.py; checked against MNE.

export type EdfData = {
  fs: number;
  channels: string[];
  rows: number[][];          // samples x channels, µV
  skipped: string[];         // signals left out (annotations, other sampling rates, non-voltage units)
  format: "EDF" | "BDF";
  durationSec: number;
};

const UNIT_TO_UV: Record<string, number> = { uv: 1, "µv": 1, "μv": 1, mv: 1e3, v: 1e6, nv: 1e-3 };

function ascii(b: Uint8Array, off: number, len: number): string {
  let s = "";
  for (let i = 0; i < len; i++) s += String.fromCharCode(b[off + i]);
  return s.trim();
}

function num(b: Uint8Array, off: number, len: number, what: string): number {
  const v = Number(ascii(b, off, len));
  if (!Number.isFinite(v)) throw new Error(`EDF header: ${what} is not a number.`);
  return v;
}

export function parseEdf(buf: ArrayBuffer): EdfData {
  const b = new Uint8Array(buf);
  if (b.length < 256) throw new Error("File is too short to be EDF/BDF.");
  const bdf = b[0] === 255 && ascii(b, 1, 7) === "BIOSEMI";
  if (!bdf && ascii(b, 0, 8) !== "0") throw new Error("Not an EDF or BDF file (bad version field).");
  const bytes = bdf ? 3 : 2;
  const headerBytes = num(b, 184, 8, "header size");
  let nRec = num(b, 236, 8, "number of records");
  const recDur = num(b, 244, 8, "record duration");
  const ns = num(b, 252, 4, "number of signals");
  if (!(ns > 0) || headerBytes !== 256 * (ns + 1)) throw new Error("EDF header: signal count and header size disagree.");
  if (!(recDur > 0)) throw new Error("EDF records with zero duration are not supported.");
  const field = (start: number, width: number, i: number) => 256 + start * ns + i * width;
  const sig = Array.from({ length: ns }, (_, i) => ({
    label: ascii(b, field(0, 16, i), 16),
    unit: ascii(b, field(96, 8, i), 8),
    pmin: num(b, field(104, 8, i), 8, "physical minimum"),
    pmax: num(b, field(112, 8, i), 8, "physical maximum"),
    dmin: num(b, field(120, 8, i), 8, "digital minimum"),
    dmax: num(b, field(128, 8, i), 8, "digital maximum"),
    spr: num(b, field(216, 8, i), 8, "samples per record"),
  }));
  const recSamples = sig.reduce((s, x) => s + x.spr, 0);
  const recBytes = recSamples * bytes;
  const avail = Math.floor((b.length - headerBytes) / recBytes);
  if (nRec < 0 || nRec > avail) nRec = avail;               // -1 = unknown (still recording) or truncated file
  if (nRec < 1) throw new Error("EDF file has no complete data records.");

  // keep voltage signals at the most common sampling rate
  const isAnnot = (l: string) => /^(EDF|BDF) Annotations$/i.test(l) || /^status$/i.test(l);
  const scaleOf = (u: string) => UNIT_TO_UV[u.toLowerCase()];
  const counts = new Map<number, number>();
  sig.forEach(s => { if (!isAnnot(s.label) && scaleOf(s.unit) !== undefined && s.spr > 0) counts.set(s.spr, (counts.get(s.spr) ?? 0) + 1); });
  if (!counts.size) throw new Error("No voltage signals (µV, mV or V) in this file.");
  const spr = Array.from(counts.entries()).sort((x, y) => y[1] - x[1] || y[0] - x[0])[0][0];
  const keep: number[] = [], skippedIdx: number[] = [];
  sig.forEach((s, i) => (!isAnnot(s.label) && scaleOf(s.unit) !== undefined && s.spr === spr ? keep : skippedIdx).push(i));
  const offsets: number[] = [];
  sig.reduce((o, s) => { offsets.push(o); return o + s.spr; }, 0);
  const gains = keep.map(i => {
    const s = sig[i];
    if (s.dmax === s.dmin) throw new Error(`Signal ${s.label}: digital minimum equals maximum.`);
    return { g: (s.pmax - s.pmin) / (s.dmax - s.dmin) * scaleOf(s.unit)!, s };
  });
  const rows: number[][] = new Array(nRec * spr);
  for (let r = 0; r < nRec; r++) {
    const base = headerBytes + r * recBytes;
    for (let k = 0; k < spr; k++) rows[r * spr + k] = new Array(keep.length);
    keep.forEach((si, c) => {
      const { g, s } = gains[c];
      let p = base + offsets[si] * bytes;
      for (let k = 0; k < spr; k++, p += bytes) {
        let d: number;
        if (bytes === 2) { d = b[p] | (b[p + 1] << 8); if (d & 0x8000) d -= 0x10000; }
        else { d = b[p] | (b[p + 1] << 8) | (b[p + 2] << 16); if (d & 0x800000) d -= 0x1000000; }
        rows[r * spr + k][c] = (d - s.dmin) * g + s.pmin * scaleOf(s.unit)!;
      }
    });
  }
  const fs = spr / recDur;
  return { fs, channels: keep.map(i => sig[i].label), rows, skipped: skippedIdx.map(i => sig[i].label).filter(l => !/annotations/i.test(l)),
           format: bdf ? "BDF" : "EDF", durationSec: nRec * recDur };
}
