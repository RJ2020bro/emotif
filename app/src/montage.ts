// Mirror of emotif/montage.py: place any headset's electrodes on the scalp and pool into 17 regions.
import electrodeTable from "../../emotif/data/electrodes_1005.json";

export type Vec3 = [number, number, number];
const POS = new Map<string, Vec3>(
  Object.entries((electrodeTable as { positions: Record<string, number[]> }).positions)
    .map(([k, v]) => [k.toUpperCase(), [v[0], v[1], v[2]] as Vec3]),
);

export const REGIONS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "Pz", "P8", "O1", "O2"];
export const REGION_PAIRS: [string, string][] = [["Fp1", "Fp2"], ["F7", "F8"], ["F3", "F4"], ["T7", "T8"], ["C3", "C4"], ["P7", "P8"], ["O1", "O2"]];
export const SIGMA_DEG = 18, MAX_DEG = 30, N_BANDS = 5;
export const N_FEATURES = REGIONS.length * N_BANDS + REGION_PAIRS.length * N_BANDS + REGIONS.length;
const R_IDX = new Map(REGIONS.map((r, i) => [r, i]));

export const DEVICES: Record<string, { label: string; channels: string[] }> = {
  muse: { label: "Muse 2 / Muse S", channels: ["TP9", "AF7", "AF8", "TP10"] },
  emotiv_insight: { label: "Emotiv Insight", channels: ["AF3", "AF4", "T7", "T8", "Pz"] },
  emotiv_epoc: { label: "Emotiv EPOC X / EPOC+", channels: ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"] },
  openbci_cyton: { label: "OpenBCI Cyton (default cap)", channels: ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"] },
  neurosity_crown: { label: "Neurosity Crown", channels: ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"] },
  unicorn: { label: "g.tec Unicorn", channels: ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"] },
  brainbit: { label: "BrainBit", channels: ["T3", "T4", "O1", "O2"] },
  frontal_2ch: { label: "2-channel forehead band", channels: ["Fp1", "Fp2"] },
};

/** 'EEG Fp1-REF' -> 'FP1'; null when the name is not a 10-05 position. */
export function canonical(name: string): string | null {
  let n = name.trim().replace(/^(EEG|EXG)[\s_:-]*/i, "");
  n = (n.split(/[-/\s]/)[0] || "").toUpperCase();
  return POS.has(n) ? n : null;
}
export function position(name: string): Vec3 | null {
  const k = canonical(name);
  return k === null ? null : POS.get(k)!;
}
export const anchor = (region: string): Vec3 => POS.get(region.toUpperCase())!;

function angleDeg(a: Vec3, b: Vec3): number {
  const d = Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
  return Math.acos(d) * 180 / Math.PI;
}

export type RegionMap = { W: Float64Array[]; mask: Float64Array };
/** W[r][c] rows sum to 1 for observed regions; mask[r] = 1 when some electrode lies within MAX_DEG. */
export function regionWeights(channels: string[]): RegionMap {
  const pos = channels.map(position);
  const W = REGIONS.map(() => new Float64Array(channels.length));
  const mask = new Float64Array(REGIONS.length);
  REGIONS.forEach((reg, r) => {
    const a = anchor(reg);
    let s = 0;
    pos.forEach((p, c) => {
      if (!p) return;
      const d = angleDeg(a, p);
      if (d <= MAX_DEG) { W[r][c] = Math.exp(-d * d / (2 * SIGMA_DEG * SIGMA_DEG)); s += W[r][c]; }
    });
    if (s > 0) { mask[r] = 1; for (let c = 0; c < channels.length; c++) W[r][c] /= s; }
  });
  return { W, mask };
}

/** zde: C x 5 per-session z-scored channel DE (only the channels the RegionMap was built for). */
export function regionFeatures(zde: ArrayLike<number>[], map: RegionMap): Float64Array {
  const R = REGIONS.length, out = new Float64Array(N_FEATURES);
  const reg = REGIONS.map((_, r) => {
    const v = new Float64Array(N_BANDS);
    const w = map.W[r];
    for (let c = 0; c < zde.length; c++) if (w[c] !== 0) for (let b = 0; b < N_BANDS; b++) v[b] += w[c] * zde[c][b];
    return v;
  });
  for (let r = 0; r < R; r++) for (let b = 0; b < N_BANDS; b++) out[r * N_BANDS + b] = reg[r][b];
  let o = R * N_BANDS;
  for (const [l, rr] of REGION_PAIRS) {
    const li = R_IDX.get(l)!, ri = R_IDX.get(rr)!;
    const both = map.mask[li] * map.mask[ri];
    for (let b = 0; b < N_BANDS; b++) out[o++] = (reg[li][b] - reg[ri][b]) * both;
  }
  for (let r = 0; r < R; r++) out[o++] = map.mask[r];
  return out;
}

export function coverage(channels: string[]): { observed: string[]; missing: string[]; unknown: string[] } {
  const { mask } = regionWeights(channels);
  return {
    observed: REGIONS.filter((_, i) => mask[i] > 0),
    missing: REGIONS.filter((_, i) => mask[i] === 0),
    unknown: channels.filter(c => position(c) === null),
  };
}

/** Azimuthal-equidistant projection to a unit head disk (nose up): returns [x, y] in [-1, 1]. */
export function project(v: Vec3): [number, number] {
  const theta = Math.acos(Math.max(-1, Math.min(1, v[2])));
  const r = theta / (Math.PI / 2);
  const phi = Math.atan2(v[1], v[0]);
  return [r * Math.cos(phi), -r * Math.sin(phi)];
}

export const knownNames = (): string[] => Array.from(POS.keys());

/** Mask-aware graph diffusion over the region graph (mirrors emotif/montage.py graph_features).
 * Returns [x, A x, A^2 x, ...] over the region DE block, each hop averaging a region's observed neighbours. */
export function graphFeatures(x: ArrayLike<number>, adjacency: number[][], hops: number): Float64Array {
  const R = REGIONS.length, B = N_BANDS;
  const out = new Float64Array(x.length + hops * R * B);
  for (let i = 0; i < x.length; i++) out[i] = x[i];
  const maskOff = x.length - R;
  const W = adjacency.map(row => {
    const w = row.map((a, j) => a * x[maskOff + j]);
    const s = w.reduce((p, q) => p + q, 0);
    return s > 0 ? w.map(v => v / s) : w.map(() => 0);
  });
  let h = Float64Array.from({ length: R * B }, (_, k) => x[k]);
  for (let k = 0; k < hops; k++) {
    const nh = new Float64Array(R * B);
    for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) {
      const w = W[i][j];
      if (w) for (let b = 0; b < B; b++) nh[i * B + b] += w * h[j * B + b];
    }
    out.set(nh, x.length + k * R * B);
    h = nh;
  }
  return out;
}
