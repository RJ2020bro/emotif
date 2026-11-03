// Mirror of emotif/face.py: facial-activity indices per channel and window, from the filtered signal before
// artifact repair. eye = log peak-to-peak of the 0-3 Hz component (two causal 3 Hz low-pass biquads from a zero
// state); jaw1, jaw2 = log power 20-30 and 30-45 Hz. tests/test_parity.py checks both sides on the same input.
import { bandPowers, type Ranges } from "./dsp";
import { REGIONS, type RegionMap } from "./montage";

export const EYE_LP_HZ = 3;
export const MUSCLE_BANDS: Ranges = [[20, 30], [30, 45]];
export const N_IDX = 3;
export const N_REGION_FACE = REGIONS.length * N_IDX;
export const N_FACE_INPUT = 2 * N_REGION_FACE;
export const FACE_NAMES = ["eye", "jaw 20-30 Hz", "jaw 30-45 Hz"];

export function eyeCoeffs(fs: number): { b: number[]; a: number[] } {
  const w0 = 2 * Math.PI * EYE_LP_HZ / fs, cw = Math.cos(w0), al = Math.sin(w0) / (2 * (1 / Math.sqrt(2)));
  return { b: [(1 - cw) / 2 / (1 + al), (1 - cw) / (1 + al), (1 - cw) / 2 / (1 + al)], a: [1, -2 * cw / (1 + al), (1 - al) / (1 + al)] };
}

/** Two cascaded low-pass biquads per channel, direct form II transposed, zero initial state (= scipy sosfilt). */
export class EyeFilter {
  private b: number[]; private a: number[];
  private z = [0, 1].map(() => ({ z1: new Float64Array(0), z2: new Float64Array(0) }));
  constructor(fs: number, C: number) {
    ({ b: this.b, a: this.a } = eyeCoeffs(fs));
    this.z = [0, 1].map(() => ({ z1: new Float64Array(C), z2: new Float64Array(C) }));
  }
  process(row: ArrayLike<number>): Float64Array {
    const y = Float64Array.from(row), b = this.b, a = this.a;
    for (const s of this.z) {
      for (let c = 0; c < y.length; c++) {
        const x = y[c], out = b[0] * x + s.z1[c];
        s.z1[c] = b[1] * x - a[1] * out + s.z2[c];
        s.z2[c] = b[2] * x - a[2] * out;
        y[c] = out;
      }
    }
    return y;
  }
}

/** C x 3: eye, jaw1, jaw2 for one window of the filtered signal and the matching slow-signal window. */
export function windowIndices(win: ArrayLike<number>[], slow: ArrayLike<number>[], fs: number): Float64Array[] {
  const p = bandPowers(win, fs, MUSCLE_BANDS);
  return win.map((_, c) => {
    let mn = Infinity, mx = -Infinity;
    const s = slow[c];
    for (let i = 0; i < s.length; i++) { if (s[i] < mn) mn = s[i]; if (s[i] > mx) mx = s[i]; }
    return Float64Array.from([Math.log(mx - mn + 1e-9), Math.log(p[c][0] + 1e-12), Math.log(p[c][1] + 1e-12)]);
  });
}

/** Pool relative indices of the channels a RegionMap was built for into regions: R*3, region-major. */
export function pool(face: ArrayLike<number>[], map: RegionMap): Float64Array {
  const out = new Float64Array(N_REGION_FACE);
  map.W.forEach((w, r) => {
    for (let c = 0; c < face.length; c++) if (w[c] !== 0) for (let k = 0; k < N_IDX; k++) out[r * N_IDX + k] += w[c] * face[c][k];
  });
  return out;
}

/** Pooled indices and their squares, standardized with a head's training statistics. */
export function faceInput(region: ArrayLike<number>, mu: ArrayLike<number>, sd: ArrayLike<number>): Float64Array {
  const n = region.length, out = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    out[i] = (region[i] - mu[i]) / sd[i];
    out[n + i] = (region[i] * region[i] - mu[n + i]) / sd[n + i];
  }
  return out;
}
