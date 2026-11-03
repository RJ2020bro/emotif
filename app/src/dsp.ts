// Mirror of emotif/features.py. tests/test_parity.py runs both on the same input.

export const BANDS: ReadonlyArray<readonly [string, number, number]> = [
  ["delta", 1, 4], ["theta", 4, 8], ["alpha", 8, 13], ["beta", 13, 30], ["gamma", 30, 45],
];
export const N_BANDS = 5;
const WIN_SEC = 2.0, SEG_SEC = 1.0, EPS = 1e-12;
const LOG2PIE = Math.log(2 * Math.PI * Math.E);
const HP_HZ = 0.5, NOTCH_HZ = [50, 60], NOTCH_Q = 30;
export const REJECT_UV = 200, BAD_UV = 500, FLAT_UV = 0.1;

/** Python 3 round(): half to even. */
export function pyRound(x: number): number {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
export const windowLen = (fs: number): number => pyRound(WIN_SEC * fs);

export type Biquad = { b: number[]; a: number[] };
export function filterCoeffs(fs: number): Biquad[] {
  const out: Biquad[] = [];
  const norm = (b: number[], a: number[]): Biquad => ({ b: b.map(v => v / a[0]), a: [1, a[1] / a[0], a[2] / a[0]] });
  let w0 = 2 * Math.PI * HP_HZ / fs, cw = Math.cos(w0), al = Math.sin(w0) / (2 * (1 / Math.sqrt(2)));
  out.push(norm([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2], [1 + al, -2 * cw, 1 - al]));
  for (const f0 of NOTCH_HZ) {
    if (f0 < 0.45 * fs) {
      w0 = 2 * Math.PI * f0 / fs; cw = Math.cos(w0); al = Math.sin(w0) / (2 * NOTCH_Q);
      out.push(norm([1, -2 * cw, 1], [1 + al, -2 * cw, 1 - al]));
    }
  }
  return out;
}

/** Causal streaming filter for C channels; same output as features.filter_signal sample for sample. */
export class StreamFilter {
  private st: { b: number[]; a: number[]; z1: Float64Array; z2: Float64Array }[];
  private started = false;
  constructor(fs: number, C: number) {
    this.st = filterCoeffs(fs).map(({ b, a }) => ({ b, a, z1: new Float64Array(C), z2: new Float64Array(C) }));
  }
  process(row: ArrayLike<number>): Float64Array {
    const y = Float64Array.from(row);
    for (const s of this.st) {
      const { b, a } = s;
      for (let c = 0; c < y.length; c++) {
        const x = y[c];
        if (!this.started) {
          const yss = x * ((b[0] + b[1] + b[2]) / (a[0] + a[1] + a[2]));
          s.z2[c] = b[2] * x - a[2] * yss;
          s.z1[c] = b[1] * x - a[1] * yss + s.z2[c];
        }
        const out = b[0] * x + s.z1[c];
        s.z1[c] = b[1] * x - a[1] * out + s.z2[c];
        s.z2[c] = b[2] * x - a[2] * out;
        y[c] = out;
      }
    }
    this.started = true;
    return y;
  }
}

export type Ranges = ReadonlyArray<readonly [number, number]>;
const EEG_RANGES: Ranges = BANDS.map(b => [b[1], b[2]] as const);
type Plan = { w: Float64Array; scale: number; bins: { b: number; c: Float64Array; s: Float64Array }[]; df: number };
const plans = new Map<string, Plan>();
function plan(L: number, fs: number, ranges: Ranges): Plan {
  const key = `${L}|${fs}|${ranges.join(";")}`;
  const hit = plans.get(key);
  if (hit) return hit;
  const w = new Float64Array(L);
  let sw2 = 0;
  for (let n = 0; n < L; n++) { w[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / L); sw2 += w[n] * w[n]; }
  const bins: Plan["bins"] = [];
  for (let k = 0; k <= Math.floor(L / 2); k++) {
    const f = k * fs / L;
    for (let b = 0; b < ranges.length; b++) {
      if (f >= ranges[b][0] && f < ranges[b][1]) {
        const c = new Float64Array(L), s = new Float64Array(L);
        for (let n = 0; n < L; n++) { const ang = 2 * Math.PI * ((k * n) % L) / L; c[n] = Math.cos(ang); s[n] = Math.sin(ang); }
        bins.push({ b, c, s });
      }
    }
  }
  const p = { w, scale: 2 / (fs * sw2), bins, df: fs / L };
  plans.set(key, p);
  return p;
}

/** win: C arrays of length windowLen(fs). Returns C x bands band power (µV²); default the 5 EEG bands. */
export function bandPowers(win: ArrayLike<number>[], fs: number, ranges: Ranges = EEG_RANGES): Float64Array[] {
  const N = win[0].length;
  const L = pyRound(SEG_SEC * fs), step = Math.floor(L / 2);
  const starts: number[] = [];
  for (let s = 0; s <= N - L; s += step) starts.push(s);
  if (!starts.length) throw new Error("window shorter than one Welch segment");
  const P = plan(L, fs, ranges), seg = new Float64Array(L), nb = ranges.length;
  return win.map(x => {
    const acc = new Float64Array(nb);
    for (const s0 of starts) {
      let mean = 0;
      for (let n = 0; n < L; n++) mean += x[s0 + n];
      mean /= L;
      for (let n = 0; n < L; n++) seg[n] = (x[s0 + n] - mean) * P.w[n];
      for (const bin of P.bins) {
        let re = 0, im = 0;
        for (let n = 0; n < L; n++) { re += seg[n] * bin.c[n]; im -= seg[n] * bin.s[n]; }
        acc[bin.b] += (re * re + im * im) * P.scale;
      }
    }
    for (let b = 0; b < nb; b++) acc[b] = acc[b] / starts.length * P.df;
    return acc;
  });
}

export const deFromPower = (p: number): number => 0.5 * (LOG2PIE + Math.log(p + EPS));
export const deMatrix = (win: ArrayLike<number>[], fs: number): Float64Array[] =>
  bandPowers(win, fs).map(r => r.map(deFromPower));

export function detrend(x: ArrayLike<number>): Float64Array {
  const N = x.length, c = (N - 1) / 2;
  let mean = 0, st = 0, tt = 0;
  for (let n = 0; n < N; n++) { mean += x[n]; const t = n - c; st += x[n] * t; tt += t * t; }
  mean /= N;
  const slope = st / tt, d = new Float64Array(N);
  for (let n = 0; n < N; n++) d[n] = x[n] - mean - slope * (n - c);
  return d;
}

export type ChannelStat = { ptp: number; std: number };
export function channelStats(win: ArrayLike<number>[]): ChannelStat[] {
  return win.map(x => {
    const d = detrend(x);
    let mn = Infinity, mx = -Infinity, ss = 0;
    for (let n = 0; n < d.length; n++) { if (d[n] < mn) mn = d[n]; if (d[n] > mx) mx = d[n]; ss += d[n] * d[n]; }
    return { ptp: mx - mn, std: Math.sqrt(ss / d.length) };
  });
}

/** Mirror of features.channel_quality: (good per channel, reject window). */
export function channelQuality(stats: ChannelStat[], rejectUv = REJECT_UV, badUv = BAD_UV): { good: boolean[]; reject: boolean } {
  const flat = stats.map(s => s.std < FLAT_UV);
  if (!(rejectUv > 0)) {
    const good = flat.map(f => !f);
    return { good, reject: !good.some(Boolean) };
  }
  const good = stats.map((s, i) => !flat[i] && s.ptp <= badUv);
  const reject = !good.some(Boolean) || stats.some((s, i) => good[i] && s.ptp > rejectUv);
  return { good, reject };
}
