// Deliberate-control signals for the robot arm (a hybrid EEG/EOG/EMG interface):
//   double blink  -> frontal electrodes pick up the eye's electrical field (EOG)
//   jaw clench    -> temporal electrodes pick up masseter/temporalis muscle activity (EMG)
// Both are calibrated per user. Spontaneous single blinks are ignored; only double blinks count.
import { position } from "./montage";

export type IntentEvent = { kind: "blink" | "double_blink" | "clench_start" | "clench_end"; t: number };
export type IntentCalibration = { blinkUv: number; clenchRatio: number; restRms: number };
export type IntentLevels = { frontal: number; clenchRms: number; clenching: boolean; blinkArmed: boolean };

const isFrontal = (n: string) => { const p = position(n); return !!p && p[1] > 0.55; };            // anterior
const isTemporal = (n: string) => { const p = position(n); return !!p && Math.abs(p[0]) > 0.7 && p[2] < 0.45; }; // lateral, low

export class IntentDetector {
  cal: IntentCalibration = { blinkUv: 70, clenchRatio: 3, restRms: 0 };
  levels: IntentLevels = { frontal: 0, clenchRms: 0, clenching: false, blinkArmed: true };
  listeners = new Set<(e: IntentEvent) => void>();
  readonly frontalIdx: number[];
  readonly temporalIdx: number[];
  private fs: number;
  private n = 0;
  private lp = 0;                       // low-passed frontal mean (~8 Hz)
  private base = 0;                     // slow baseline (~0.5 s)
  private peak = 0; private peakStart = -1;
  private lastBlink = -10;
  private rmsAcc = 0; private prevT = new Float64Array(0);
  private rmsWin: number[] = []; private rmsWinLen: number;
  private clenchOn = 0; private clenchOff = 0;
  private restSamples: number[] = [];

  constructor(fs: number, channels: string[]) {
    this.fs = fs;
    this.frontalIdx = channels.map((c, i) => (isFrontal(c) ? i : -1)).filter(i => i >= 0);
    this.temporalIdx = channels.map((c, i) => (isTemporal(c) ? i : -1)).filter(i => i >= 0);
    if (!this.temporalIdx.length) this.temporalIdx = channels.map((_, i) => i);
    this.rmsWinLen = Math.max(4, Math.round(0.1 * fs));
    this.prevT = new Float64Array(channels.length);
  }

  get t() { return this.n / this.fs; }
  get canBlink() { return this.frontalIdx.length > 0; }

  private emit(kind: IntentEvent["kind"]) {
    const e = { kind, t: this.t };
    for (const f of this.listeners) f(e);
  }

  /** Feed one filtered sample (µV per channel). */
  process(row: ArrayLike<number>) {
    this.n++;
    const dt = 1 / this.fs;
    // ---- blink: positive frontal deflection, 80-500 ms wide, above calibrated amplitude
    if (this.frontalIdx.length) {
      let m = 0;
      for (const i of this.frontalIdx) m += row[i];
      m /= this.frontalIdx.length;
      this.lp += (m - this.lp) * Math.min(1, dt * 2 * Math.PI * 8);
      this.base += (this.lp - this.base) * Math.min(1, dt * 2 * Math.PI * 0.4);
      const d = this.lp - this.base;
      this.levels.frontal = d;
      const thr = this.cal.blinkUv;
      if (d > thr) {
        if (this.peakStart < 0) { this.peakStart = this.t; this.peak = d; }
        this.peak = Math.max(this.peak, d);
      } else if (this.peakStart >= 0 && d < thr * 0.5) {
        const width = this.t - this.peakStart;
        this.peakStart = -1;
        if (width >= 0.04 && width <= 0.5) {
          const now = this.t;
          if (now - this.lastBlink <= 0.7) { this.emit("double_blink"); this.lastBlink = -10; }
          else { this.emit("blink"); this.lastBlink = now; }
        }
      }
      this.levels.blinkArmed = this.peakStart < 0;
    }
    // ---- clench: RMS of the first difference (emphasises > 30 Hz muscle activity) on temporal channels
    let e = 0;
    for (const i of this.temporalIdx) { const dd = row[i] - this.prevT[i]; e += dd * dd; this.prevT[i] = row[i]; }
    e /= this.temporalIdx.length;
    this.rmsWin.push(e); this.rmsAcc += e;
    if (this.rmsWin.length > this.rmsWinLen) this.rmsAcc -= this.rmsWin.shift()!;
    const rms = Math.sqrt(Math.max(0, this.rmsAcc) / this.rmsWin.length);
    this.levels.clenchRms = rms;
    if (this.restSamples.length < this.fs * 60) this.restSamples.push(rms);
    const rest = this.cal.restRms > 0 ? this.cal.restRms : median(this.restSamples);
    const on = rms > rest * this.cal.clenchRatio;
    if (on) { this.clenchOn += dt; this.clenchOff = 0; } else { this.clenchOff += dt; this.clenchOn = 0; }
    if (!this.levels.clenching && this.clenchOn >= 0.15) { this.levels.clenching = true; this.emit("clench_start"); }
    else if (this.levels.clenching && this.clenchOff >= 0.2) { this.levels.clenching = false; this.emit("clench_end"); }
  }

  /** Calibrate from recorded segments: rest RMS, clench RMS and blink peak amplitudes. */
  calibrate(rest: number[], clench: number[], blinkPeaks: number[]) {
    const r = median(rest), c = median(clench);
    if (r > 0 && c > r * 1.5) this.cal.clenchRatio = Math.max(1.8, Math.sqrt(c / r));
    if (r > 0) this.cal.restRms = r;
    if (blinkPeaks.length >= 2) this.cal.blinkUv = Math.max(25, 0.5 * median(blinkPeaks));
  }
}

export function median(a: number[]): number {
  if (!a.length) return 0;
  const s = a.slice().sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
