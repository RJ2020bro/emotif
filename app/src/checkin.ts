// Daily check-in: a two-minute resting recording plus mood and sleep answers, once a day. Mood and sleep are traits
// of days, not of seconds, so they are compared across days within one person, using gain-free resting markers
// (relative band power), which unlike the live decoder's features don't depend on that day's baseline.
// On public data these markers did not separate depressed, anxious or sleepy people from others; whether they
// track one person's changes over days is the open question this measures. It never diagnoses.
import { canonical } from "./montage";
import type { CheckinAnswers } from "./protocol";

export const MARKERS = ["frontal alpha asymmetry", "relative theta", "relative alpha", "relative beta", "posterior alpha"] as const;
export type Marker = typeof MARKERS[number];
export type CheckinEntry = { date: string; answers: CheckinAnswers; markers: Record<Marker, number>; windows: number; channels: string[] };

const LEFT_FRONT = new Set(["FP1", "AF3", "AF7", "F3", "F7", "F5", "F1"]);
const RIGHT_FRONT = new Set(["FP2", "AF4", "AF8", "F4", "F8", "F6", "F2"]);
const POSTERIOR = new Set(["PZ", "P3", "P4", "P7", "P8", "POZ", "PO3", "PO4", "PO7", "PO8", "O1", "O2", "OZ", "TP9", "TP10"]);

/** Accumulates usable windows' relative log band power per channel during the check-in. */
export class CheckinAccumulator {
  private rows: Float64Array[][] = [];
  constructor(readonly channels: string[]) {}
  /** bp: per channel, power in the 5 bands (delta..gamma); masked: channels to skip in this window. */
  add(bp: ArrayLike<number>[], masked: boolean[]) {
    this.rows.push(bp.map((r, c) => {
      if (masked[c]) return new Float64Array(5).fill(NaN);
      let tot = 0;
      for (let b = 0; b < 5; b++) tot += r[b];
      return Float64Array.from({ length: 5 }, (_, b) => (tot > 0 && r[b] > 0 ? Math.log(r[b] / tot) : NaN));
    }));
  }
  get count() { return this.rows.length; }
  /** Median over windows per channel and band, then the markers. Null with fewer than 20 usable windows. */
  summarize(): { markers: Record<Marker, number>; windows: number } | null {
    if (this.rows.length < 20) return null;
    const C = this.channels.length;
    const med = Array.from({ length: C }, (_, c) => Float64Array.from({ length: 5 }, (_, b) => median(this.rows.map(r => r[c][b]))));
    const names = this.channels.map(n => canonical(n) ?? "");
    const avg = (sel: (n: string) => boolean, b: number) => {
      const v = med.filter((_, c) => sel(names[c])).map(r => r[b]).filter(Number.isFinite);
      return v.length ? v.reduce((a, x) => a + x, 0) / v.length : NaN;
    };
    const any = (n: string) => n !== "";
    return {
      windows: this.rows.length,
      markers: {
        "frontal alpha asymmetry": avg(n => RIGHT_FRONT.has(n), 2) - avg(n => LEFT_FRONT.has(n), 2),
        "relative theta": avg(any, 1), "relative alpha": avg(any, 2), "relative beta": avg(any, 3),
        "posterior alpha": avg(n => POSTERIOR.has(n), 2),
      },
    };
  }
}

function median(v: number[]): number {
  const s = v.filter(Number.isFinite).sort((a, b) => a - b);
  if (!s.length) return NaN;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function ranks(v: number[]): number[] {
  const idx = v.map((x, i) => [x, i] as [number, number]).sort((a, b) => a[0] - b[0]);
  const r = new Array(v.length).fill(0);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}
export function spearman(a: number[], b: number[]): number {
  const ra = ranks(a), rb = ranks(b), n = a.length;
  const ma = ra.reduce((s, x) => s + x, 0) / n, mb = rb.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2; }
  return da > 0 && db > 0 ? num / Math.sqrt(da * db) : NaN;
}

export type TrendRow = { marker: Marker; answer: keyof CheckinAnswers; rho: number; p: number; pHolm: number; n: number };
export const MIN_DAYS = 7;

/** Within-person Spearman correlation of each marker with each answer across days, two-sided permutation p
 * (days shuffled), Holm-corrected over the whole table. Empty until MIN_DAYS complete check-ins exist. */
export function trends(entries: CheckinEntry[], perms = 2000, seed = 1): TrendRow[] {
  const ok = entries.filter(e => MARKERS.every(m => Number.isFinite(e.markers[m])));
  if (ok.length < MIN_DAYS) return [];
  let a = seed >>> 0;
  const rnd = () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const rows: TrendRow[] = [];
  for (const marker of MARKERS) for (const answer of ["mood", "sleepHours", "sleepQuality"] as const) {
    const x = ok.map(e => e.markers[marker]), y = ok.map(e => e.answers[answer]);
    const rho = spearman(x, y);
    let ge = 0;
    const yp = y.slice();
    for (let i = 0; i < perms; i++) {
      for (let k = yp.length - 1; k > 0; k--) { const j = Math.floor(rnd() * (k + 1)); [yp[k], yp[j]] = [yp[j], yp[k]]; }
      const r = spearman(x, yp);
      if (Number.isFinite(r) && Number.isFinite(rho) && Math.abs(r) >= Math.abs(rho) - 1e-12) ge++;
    }
    rows.push({ marker, answer, rho, p: Number.isFinite(rho) ? (1 + ge) / (1 + perms) : 1, pHolm: 1, n: ok.length });
  }
  const order = rows.map((r, i) => [r.p, i] as [number, number]).sort((u, v) => u[0] - v[0]);
  let run = 0;
  order.forEach(([p, i], k) => { run = Math.max(run, Math.min(1, (rows.length - k) * p)); rows[i].pHolm = run; });
  return rows;
}
