// Live decoding pipeline
//   raw -> causal filter (0.5 Hz HP, 50/60 Hz notch) ──┬─> intent detector (blinks, jaw) and raw traces
//                                                      └─> ASR artifact repair -> 2 s windows every 0.25 s
//   -> amplitude rules + learned artifact detector (mask bad channels) -> differential entropy
//   -> per-session z-score -> 17-region embedding (montage-agnostic) -> model or neuro-markers -> Kalman smoothing
// In parallel, facial-activity indices (blink size, jaw-muscle power) come from the filtered signal before repair,
// pooled into the same regions; brain-only model heads subtract everything they predict (face.ts, model.ts).
// The resting baseline calibrates ASR, the detector's per-channel reference, the z-score and the face centre.
import { BANDS, StreamFilter, bandPowers, channelQuality, channelStats, deFromPower, windowLen, type ChannelStat } from "./dsp";
import { N_FEATURES, REGIONS, REGION_PAIRS, regionFeatures, regionWeights, type RegionMap } from "./montage";
import { faceOutOfRange, headInput, headOutput, validateModel, type Head, type ModelFile } from "./model";
import { EyeFilter, N_IDX, N_REGION_FACE, pool, windowIndices } from "./face";
import { stateProbs, type PersonalStates } from "./states";
import { AsrStream, calibrate as asrCalibrate, MIN_CAL_SEC, DEFAULT_K, type AsrState } from "./asr";
import { baselineOf, channelFeatures, classify, validateArtifactModel, type ArtifactModel, type ChannelVerdict } from "./artifact";

export const HOP_SEC = 0.25;

class Welford {
  n: Float64Array; m: Float64Array; s: Float64Array;
  constructor(d: number) { this.n = new Float64Array(d); this.m = new Float64Array(d); this.s = new Float64Array(d); }
  push(i: number, x: number) {
    this.n[i]++;
    const d = x - this.m[i];
    this.m[i] += d / this.n[i];
    this.s[i] += d * (x - this.m[i]);
  }
  z(i: number, x: number): number {
    let sd = this.n[i] > 1 ? Math.sqrt(this.s[i] / this.n[i]) : 1;
    if (!(sd >= 1e-6)) sd = 1;
    return this.n[i] > 0 ? (x - this.m[i]) / sd : 0;
  }
}

class Kalman1 {
  x = 0; P = 1;
  constructor(private q: number) {}
  reset() { this.x = 0; this.P = 1; }
  predict() { this.P += this.q; }
  update(z: number, r: number) { this.P += this.q; const K = this.P / (this.P + r); this.x += K * (z - this.x); this.P *= 1 - K; }
}

/** Fixed-capacity multichannel ring buffer. */
class Ring {
  buf: Float64Array[]; w = 0; total = 0;
  constructor(C: number, readonly cap: number) { this.buf = Array.from({ length: C }, () => new Float64Array(cap)); }
  push(row: ArrayLike<number>) { for (let c = 0; c < this.buf.length; c++) this.buf[c][this.w] = row[c]; this.w = (this.w + 1) % this.cap; this.total++; }
  last(n: number): Float64Array[] {
    n = Math.min(n, this.total, this.cap);
    return this.buf.map(b => {
      const out = new Float64Array(n);
      let s = (this.w - n + this.cap) % this.cap;
      for (let i = 0; i < n; i++) { out[i] = b[s]; s = (s + 1) % this.cap; }
      return out;
    });
  }
}

export type Normalization = "auto" | "baseline" | "session";
/** cmp: the comparison (standard) decoder, smoothed the same way, when one is set. */
/** faceHold: a brain-only head saw facial activity beyond its training range, so the output holds its last value. */
/** extra: the model's other heads (e.g. "stress"), each the expected class value in -1..1, smoothed like v and a. */
export type AffectOut = { v: number; a: number; rawV: number; rawA: number; conf: number | null; source: "model" | "markers"; focus: number; faceHold: boolean;
  cmp: { v: number; a: number; rawV: number; rawA: number } | null; extra: Record<string, number>;
  /** Personal states (after calibration): smoothed probability of each state; only states that passed their
   * held-out check are reported. */
  states: Record<string, number> };
export type WindowEvent = {
  t: number;
  reject: boolean;
  stats: ChannelStat[];
  good: boolean[];                       // passes amplitude rules
  masked: boolean[];                     // excluded from decoding (amplitude or detector)
  verdicts: ChannelVerdict[] | null;     // detector output per channel (after baseline)
  bandPower: Float64Array[] | null;      // per channel, µV²
  zde: (Float64Array | null)[] | null;   // per channel z-scored DE (null = channel unusable)
  features: Float64Array | null;         // region features
  face: Float64Array[] | null;           // per channel eye, jaw1, jaw2 relative to the baseline (null before it)
  faceRegion: Float64Array | null;       // the same pooled into regions over the channels used for decoding
  quality: number;                       // share of usable windows over the last 10 s
  repaired: number;                      // share of ASR blocks repaired over the last 10 s
  calibProgress: number;
  calibrated: boolean;
  out: AffectOut | null;
};

export type Cleaning = { asr: boolean; asrK: number; detector: boolean };
export type Settings = { calibSec: number; rejectUv: number; normalization: Normalization; cleaning: Cleaning };
const MAX_CAL_SEC = 300;

export class Engine {
  fs = 0; channels: string[] = []; C = 0;
  total = 0;
  settings: Settings = { calibSec: 30, rejectUv: 200, normalization: "auto", cleaning: { asr: true, asrK: DEFAULT_K, detector: true } };
  model: ModelFile | null = null;
  artifactModel: ArtifactModel | null = null;
  windows = 0; rejected = 0; faceHolds = 0;
  last: AffectOut | null = null;
  lastEvent: WindowEvent | null = null;
  compareModel: ModelFile | null = null;
  history: { t: number; v: number; a: number; q: number; cv?: number; ca?: number }[] = [];
  featureLog: { t: number; x: Float32Array; f: Float32Array }[] = [];
  listeners = new Set<(e: WindowEvent) => void>();
  rowTaps = new Set<(row: Float64Array, raw: ArrayLike<number>) => void>();
  calibError = "";
  asrActive = false;
  detectorActive = false;

  private live = false;
  private raw: Ring | null = null; private filt: Ring | null = null; private clean: Ring | null = null;
  private cal: Float64Array[] = []; private calLen = 0; private calCap = 0;
  private sinceHop = 0; private hopN = 1; private N = 0;
  private filter: StreamFilter | null = null;
  private asr: AsrStream | null = null;
  private asrState: AsrState | null = null;
  private detBase: Float64Array[] | null = null;
  private norm: Welford | null = null;
  private focusNorm = new Welford(1);
  private kv = new Kalman1(0.002); private ka = new Kalman1(0.002);
  private kcv = new Kalman1(0.002); private kca = new Kalman1(0.002);
  private kx = new Map<string, Kalman1>();     // extra heads
  personalStates: PersonalStates | null = null;
  private stateEma: Float64Array | null = null;
  stateTau = 8;                                // seconds: smoothing of personal-state probabilities
  private eye: EyeFilter | null = null; private slow: Ring | null = null;
  private faceCal: Float64Array[][] = []; private faceBase: Float64Array[] | null = null;
  private recent: number[] = [];
  private blockLog: number[] = [];
  private maps = new Map<string, RegionMap>();

  get t(): number { return this.fs ? this.total / this.fs : 0; }
  get calibrated(): boolean { return this.live; }
  get frozen(): boolean {
    const n = this.settings.normalization;
    return n === "baseline" || (n === "auto" && !this.model);
  }
  get delaySec(): number { return this.asr ? this.asr.delay / this.fs : 0; }

  setArtifactModel(m: unknown | null) { this.artifactModel = m ? validateArtifactModel(m) : null; }

  setStream(fs: number, channels: string[]) {
    if (!(fs >= 100)) throw new Error(`Sampling rate ${fs} Hz is too low; Emotif needs at least 100 Hz.`);
    if (!channels.length) throw new Error("The stream has no channels.");
    this.fs = fs; this.channels = channels.slice(); this.C = channels.length;
    this.N = windowLen(fs); this.hopN = Math.max(1, Math.round(HOP_SEC * fs));
    const cap = Math.max(this.N, Math.round(fs * 8));
    this.raw = new Ring(this.C, cap); this.filt = new Ring(this.C, cap); this.slow = new Ring(this.C, cap);
    this.total = 0;
    this.filter = new StreamFilter(fs, this.C);
    this.eye = new EyeFilter(fs, this.C);
    this.maps.clear();
    this.history = []; this.featureLog = [];
    this.resetCalibration();
  }

  /** Rename channels without restarting the stream (montage editor). */
  renameChannels(names: string[]) {
    if (names.length !== this.C) throw new Error("channel count changed");
    this.channels = names.slice();
    this.maps.clear();
    this.resetCalibration();
  }

  /** Start a new resting baseline: decoding pauses until it completes. */
  resetCalibration() {
    this.norm = new Welford(Math.max(1, this.C * 5));
    this.focusNorm = new Welford(1);
    this.windows = 0; this.rejected = 0; this.faceHolds = 0; this.recent = []; this.blockLog = [];
    this.kv.reset(); this.ka.reset(); this.kcv.reset(); this.kca.reset(); this.kx.clear(); this.stateEma = null;
    this.faceCal = []; this.faceBase = null;
    this.last = null;
    this.live = false;
    this.asr = null; this.asrState = null; this.asrActive = false;
    this.detBase = null; this.detectorActive = false;
    this.clean = this.C ? new Ring(this.C, Math.max(this.N, Math.round(this.fs * 8))) : null;
    this.calCap = Math.round(Math.min(this.settings.calibSec, MAX_CAL_SEC) * this.fs);
    this.cal = Array.from({ length: this.C }, () => new Float64Array(Math.max(1, this.calCap)));
    this.calLen = 0;
    this.sinceHop = 0;
  }

  setModel(m: unknown | null) {
    this.model = m ? validateModel(m) : null;
    this.resetCalibration();
  }

  /** A second decoder run on the same windows for comparison (e.g. the standard model next to a brain-only one). */
  setCompareModel(m: unknown | null) {
    this.compareModel = m ? validateModel(m) : null;
    this.kcv.reset(); this.kca.reset();
  }

  push(rows: ArrayLike<number>[]) {
    if (!this.C || !this.filter || !this.raw || !this.filt || !this.clean || !this.eye || !this.slow) return;
    for (const row of rows) {
      if (row.length !== this.C) continue;
      let ok = true;
      for (let c = 0; c < this.C; c++) if (!Number.isFinite(row[c])) { ok = false; break; }
      if (!ok) continue;
      const f = this.filter.process(row);
      this.raw.push(row); this.filt.push(f); this.slow.push(this.eye.process(f)); this.total++;
      for (const tap of this.rowTaps) tap(f, row);
      if (!this.live) {
        if (this.calLen < this.calCap) { for (let c = 0; c < this.C; c++) this.cal[c][this.calLen] = f[c]; this.calLen++; }
        this.addClean(f);
        if (this.calLen >= this.calCap) this.finishCalibration();
      } else if (this.asr) {
        const outs = this.asr.push(f);
        if (outs.length) {                       // one ASR block finished
          this.blockLog.push(this.asr.removed > 0 ? 1 : 0);
          if (this.blockLog.length > 80) this.blockLog.shift();   // last 10 s
        }
        for (const o of outs) this.addClean(o);
      } else this.addClean(f);
    }
  }

  private addClean(row: ArrayLike<number>) {
    this.clean!.push(row);
    this.sinceHop++;
    if (this.clean!.total >= this.N && this.sinceHop >= this.hopN) { this.sinceHop = 0; this.step(); }
  }

  lastSamples(n: number, which: "clean" | "filtered" | "raw" = "clean"): Float64Array[] {
    const r = which === "clean" ? this.clean : which === "filtered" ? this.filt : this.raw;
    return r ? r.last(n) : [];
  }

  private regionMap(good: boolean[]): { map: RegionMap; idx: number[] } {
    const idx = good.map((g, i) => (g ? i : -1)).filter(i => i >= 0);
    const key = idx.join(",");
    let map = this.maps.get(key);
    if (!map) { map = regionWeights(idx.map(i => this.channels[i])); this.maps.set(key, map); }
    return { map, idx };
  }

  /** Calibrate ASR on the baseline, clean it, and learn the detector reference and z-score from it. */
  private finishCalibration() {
    const fs = this.fs, C = this.C;
    const calX = this.cal.map(b => b.subarray(0, this.calLen));
    const cl = this.settings.cleaning;
    let cleaned: Float64Array[] = calX;
    let asr: AsrStream | null = null;
    if (cl.asr && this.calLen >= MIN_CAL_SEC * fs) {
      try {
        this.asrState = asrCalibrate(calX, fs, cl.asrK);
        asr = new AsrStream(this.asrState, fs);
        const out: Float64Array[] = [];
        const row = new Float64Array(C);
        for (let t = 0; t < this.calLen; t++) { for (let c = 0; c < C; c++) row[c] = calX[c][t]; for (const o of asr.push(row)) out.push(o); }
        cleaned = Array.from({ length: C }, (_, c) => Float64Array.from(out, o => o[c]));
      } catch { asr = null; this.asrState = null; }
    }
    // baseline windows
    const rows: { de: Float64Array[]; good: boolean[]; feats: Float64Array[] | null; bp: Float64Array[] }[] = [];
    const n = cleaned[0].length;
    for (let s = 0; s + this.N <= n; s += this.hopN) {
      const win = cleaned.map(x => x.subarray(s, s + this.N));
      const { good, reject } = channelQuality(channelStats(win), this.settings.rejectUv);
      if (reject) continue;
      const bp = bandPowers(win, fs);
      const feats = cl.detector && this.artifactModel ? win.map((x, c) => channelFeatures(x, fs, bp[c])) : null;
      rows.push({ de: bp.map(r => r.map(deFromPower)), good, feats, bp });
    }
    if (rows.length < 8) {
      this.calibError = "The baseline was too noisy to calibrate. Check electrode contact, sit still, and try again.";
      this.resetCalibration();
      return;
    }
    this.calibError = "";
    this.detBase = cl.detector && this.artifactModel ? baselineOf(rows.map(r => r.feats!)) : null;
    const norm = this.norm!;
    for (const r of rows) {
      const mask = r.good.slice();
      if (this.detBase) r.feats!.forEach((f, c) => { if (classify(this.artifactModel!, f, this.detBase![c]).bad) mask[c] = false; });
      for (let c = 0; c < C; c++) if (mask[c]) for (let b = 0; b < 5; b++) norm.push(c * 5 + b, r.de[c][b]);
      let eng = 0, ne = 0;
      r.bp.forEach((p, c) => { if (mask[c]) { eng += Math.log((p[3] + 1e-9) / (p[2] + p[1] + 1e-9)); ne++; } });
      if (ne) this.focusNorm.push(0, eng / ne);
    }
    // switch to live: the ASR stream continues from the end of the baseline without a gap
    this.asr = asr;
    this.asrActive = !!asr;
    this.detectorActive = !!this.detBase;
    this.clean = new Ring(C, Math.max(this.N, Math.round(fs * 8)));
    for (let t = Math.max(0, n - this.clean.cap); t < n; t++) this.clean.push(cleaned.map(x => x[t]));
    // face centre: per channel median of the baseline windows' face indices
    this.faceBase = Array.from({ length: C }, (_, c) => {
      const out = new Float64Array(N_IDX);
      for (let k = 0; k < N_IDX; k++) {
        const v = this.faceCal.map(f => f[c][k]).sort((a, b) => a - b);
        out[k] = v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : 0;
      }
      return out;
    });
    this.sinceHop = 0;
    this.kv.reset(); this.ka.reset(); this.kcv.reset(); this.kca.reset(); this.kx.clear();
    this.live = true;
  }

  /** Face indices for the window the cleaned signal is showing: the filtered and slow signals are not delayed by
   * ASR, so read them that many samples back. */
  private faceWindow(): Float64Array[] | null {
    const D = this.live && this.asr ? this.asr.delay : 0;
    if (!this.filt || !this.slow || this.filt.total < this.N + D) return null;
    const f = this.filt.last(this.N + D).map(x => x.subarray(0, this.N));
    const s = this.slow.last(this.N + D).map(x => x.subarray(0, this.N));
    return windowIndices(f, s, this.fs);
  }

  private headValue(m: ModelFile, head: Head | undefined, features: Float64Array, faceRegion: Float64Array | null) {
    return head ? headOutput(head, headInput(m, head, features, faceRegion)) : null;
  }

  private step() {
    const win = this.clean!.last(this.N);
    const stats = channelStats(win);
    const { good, reject } = channelQuality(stats, this.settings.rejectUv);
    const faceAbs = this.faceWindow();
    let face: Float64Array[] | null = null, faceRegion: Float64Array | null = null;
    if (!this.live && faceAbs && !reject) this.faceCal.push(faceAbs);
    if (this.live && faceAbs && this.faceBase) face = faceAbs.map((r, c) => r.map((v, k) => v - this.faceBase![c][k]));
    let masked = good.map(g => !g);
    let verdicts: ChannelVerdict[] | null = null;
    let bandPower: Float64Array[] | null = null, zde: (Float64Array | null)[] | null = null, features: Float64Array | null = null;
    let usable = !reject;
    if (!reject) {
      bandPower = bandPowers(win, this.fs);
      if (this.live && this.detBase && this.artifactModel) {
        verdicts = win.map((x, c) => classify(this.artifactModel!, channelFeatures(x, this.fs, bandPower![c]), this.detBase![c]));
        masked = masked.map((m, c) => m || verdicts![c].bad);
        if (masked.every(Boolean)) usable = false;
      }
    }
    this.recent.push(usable ? 1 : 0);
    if (this.recent.length > 40) this.recent.shift();
    const quality = this.recent.reduce((a, b) => a + b, 0) / this.recent.length;
    const repaired = this.blockLog.length ? this.blockLog.reduce((a, b) => a + b, 0) / this.blockLog.length : 0;

    if (!usable) {
      this.rejected++;
      this.kv.predict(); this.ka.predict(); this.kcv.predict(); this.kca.predict(); for (const k of this.kx.values()) k.predict();
    } else {
      this.windows++;
      if (this.live) {
        const de = bandPower!.map(r => r.map(deFromPower));
        const norm = this.norm!;
        const update = !this.frozen;
        if (update) for (let c = 0; c < this.C; c++) if (!masked[c]) for (let b = 0; b < 5; b++) norm.push(c * 5 + b, de[c][b]);
        zde = de.map((r, c) => (!masked[c] && norm.n[c * 5] > 0 ? r.map((v, b) => norm.z(c * 5 + b, v)) : null));
        const { map, idx } = this.regionMap(zde.map(z => z !== null));
        features = idx.length ? regionFeatures(idx.map(i => zde![i]!), map) : new Float64Array(N_FEATURES);
        faceRegion = face && idx.length ? pool(idx.map(i => face![i]), map) : null;
        this.featureLog.push({ t: this.t, x: Float32Array.from(features), f: faceRegion ? Float32Array.from(faceRegion) : new Float32Array(N_REGION_FACE) });
        if (this.featureLog.length > 4 * 3600 * 2) this.featureLog.shift();

        // engagement index beta / (alpha + theta) over usable channels (Pope et al. 1995)
        let eng = 0, ne = 0;
        bandPower!.forEach((r, c) => { if (!masked[c]) { eng += Math.log((r[3] + 1e-9) / (r[2] + r[1] + 1e-9)); ne++; } });
        eng = ne ? eng / ne : 0;
        if (update) this.focusNorm.push(0, eng);
        const focus = 1 / (1 + Math.exp(-this.focusNorm.z(0, eng)));

        const mk = markers(features);
        let v = Math.tanh(mk.v / 2), a = Math.tanh(mk.a / 2), conf: number | null = null, source: AffectOut["source"] = "markers";
        let faceHold = false;
        if (this.model) {
          const confs: number[] = [];
          faceHold = Object.values(this.model.heads).some(h => !!h && faceOutOfRange(h, faceRegion));
          const ov = this.headValue(this.model, this.model.heads.valence, features, faceRegion);
          const oa = this.headValue(this.model, this.model.heads.arousal, features, faceRegion);
          if (ov) { v = ov.value; confs.push(ov.conf); }
          if (oa) { a = oa.value; confs.push(oa.conf); }
          conf = confs.reduce((s, x) => s + x, 0) / confs.length;
          source = "model";
        }
        const r = conf === null ? 0.05 : 0.03 / Math.max(0.2, conf);
        if (faceHold) { this.faceHolds++; this.kv.predict(); this.ka.predict(); }
        else { this.kv.update(v, r); this.ka.update(a, r); }
        let cmp: AffectOut["cmp"] = null;
        if (this.compareModel) {
          const m = this.compareModel;
          const ov = this.headValue(m, m.heads.valence, features, faceRegion), oa = this.headValue(m, m.heads.arousal, features, faceRegion);
          const cc = [ov, oa].filter(Boolean).map(o => o!.conf);
          const rc = cc.length ? 0.03 / Math.max(0.2, cc.reduce((s, x) => s + x, 0) / cc.length) : 0.05;
          const cv = ov ? ov.value : Math.tanh(mk.v / 2), ca = oa ? oa.value : Math.tanh(mk.a / 2);
          this.kcv.update(cv, rc); this.kca.update(ca, rc);
          cmp = { v: this.kcv.x, a: this.kca.x, rawV: cv, rawA: ca };
        }
        const extra: Record<string, number> = {};
        if (this.model) for (const [name, head] of Object.entries(this.model.heads)) {
          if (name === "valence" || name === "arousal" || !head) continue;
          let k = this.kx.get(name);
          if (!k) { k = new Kalman1(0.002); this.kx.set(name, k); }
          const o = faceHold ? null : this.headValue(this.model, head, features, faceRegion);
          if (o) k.update(o.value, 0.03 / Math.max(0.2, o.conf)); else k.predict();
          extra[name] = k.x;
        }
        const states: Record<string, number> = {};
        const ps = this.personalStates;
        if (ps) {
          const held = faceOutOfRange(ps.head, faceRegion);
          if (!held) {
            const p = stateProbs(ps, features, faceRegion ?? new Float64Array(N_REGION_FACE));
            const k = 1 - Math.exp(-HOP_SEC / this.stateTau);
            if (!this.stateEma || this.stateEma.length !== p.length) this.stateEma = Float64Array.from(p);
            else for (let i = 0; i < p.length; i++) this.stateEma[i] += k * (p[i] - this.stateEma[i]);
          }
          if (this.stateEma) ps.states.forEach((n, i) => { if (ps.checks[n]?.on) states[n] = this.stateEma![i]; });
        }
        this.last = { v: this.kv.x, a: this.ka.x, rawV: v, rawA: a, conf, source, focus, faceHold, cmp, extra, states };
        this.history.push({ t: this.t, v: this.kv.x, a: this.ka.x, q: quality, ...(cmp ? { cv: cmp.v, ca: cmp.a } : {}) });
        if (this.history.length > 4 * 3600 * 4) this.history.shift();
      }
    }
    const ev: WindowEvent = {
      t: this.t, reject: !usable, stats, good, masked, verdicts, bandPower, zde, features, face, faceRegion, quality, repaired,
      calibProgress: this.live ? 1 : Math.min(1, this.calLen / Math.max(1, this.calCap)), calibrated: this.live, out: this.last,
    };
    this.lastEvent = ev;
    for (const f of this.listeners) f(ev);
  }
}

const FRONTAL_PAIRS = ["Fp1", "F7", "F3"];
/** Neuro-markers from region features (relative to the baseline):
 * valence = right-minus-left frontal alpha (more right alpha = more left-frontal activity = approach / positive),
 * arousal = beta minus alpha across observed regions. */
export function markers(x: Float64Array): { v: number; a: number } {
  const R = REGIONS.length, off = R * 5, maskOff = off + REGION_PAIRS.length * 5;
  let v = 0, nv = 0;
  REGION_PAIRS.forEach(([l, r], p) => {
    const li = REGIONS.indexOf(l), ri = REGIONS.indexOf(r);
    if (FRONTAL_PAIRS.includes(l) && x[maskOff + li] > 0 && x[maskOff + ri] > 0) { v += -x[off + p * 5 + 2]; nv++; }
  });
  let a = 0, na = 0;
  for (let r = 0; r < R; r++) if (x[maskOff + r] > 0) { a += x[r * 5 + 3] - x[r * 5 + 2]; na++; }
  return { v: nv ? v / nv : 0, a: na ? a / na : 0 };
}

export const BAND_LABELS = BANDS.map(b => b[0]);

export function emotionWord(v: number, a: number): string {
  if (Math.hypot(v, a) < 0.25) return "Neutral";
  if (v >= 0) return a >= 0 ? "Happy" : "Calm";
  return a >= 0 ? "Stressed" : "Sad";
}

/** Plain-language contact check from raw (unfiltered) and filtered windows, for the fitting screen.
 * Works before any baseline exists. */
export type FitStatus = "good" | "noisy" | "muscle" | "mains" | "drift" | "no contact";
export function fitCheck(raw: Float64Array[], filtered: Float64Array[], fs: number): { status: FitStatus; ptp: number; mainsRatio: number }[] {
  if (!raw.length || raw[0].length < fs) return [];
  const bp = bandPowers(filtered.map(x => x.subarray(x.length - windowLen(fs))), fs);
  const st = channelStats(filtered.map(x => x.subarray(x.length - windowLen(fs))));
  return raw.map((x, c) => {
    const n = Math.min(x.length, Math.round(2 * fs)), seg = x.subarray(x.length - n);
    // mains: power at 50 or 60 Hz vs the 1-45 Hz EEG band (single-bin DFT on the raw signal)
    let mean = 0;
    for (let i = 0; i < n; i++) mean += seg[i];
    mean /= n;
    const tone = (f: number) => {
      if (f >= fs / 2) return 0;
      let re = 0, im = 0;
      for (let i = 0; i < n; i++) { const a = 2 * Math.PI * f * i / fs; re += (seg[i] - mean) * Math.cos(a); im -= (seg[i] - mean) * Math.sin(a); }
      return (re * re + im * im) / (n * n) * 2;
    };
    let eeg = 0;
    for (let b = 0; b < 5; b++) eeg += bp[c][b];
    const mainsRatio = Math.max(tone(50), tone(60)) / (eeg + 1e-9);
    let lo = 0;
    for (let i = 0; i < n; i++) lo += (seg[i] - mean) * (seg[i] - mean);
    const rawStd = Math.sqrt(lo / n);
    const s = st[c];
    let status: FitStatus = "good";
    if (s.std < 0.1 || s.ptp > 500) status = "no contact";
    else if (mainsRatio > 2) status = "mains";
    else if (rawStd > 6 * Math.max(s.std, 1) && rawStd > 100) status = "drift";
    else if (bp[c][4] / (eeg + 1e-9) > 0.25) status = "muscle";
    else if (s.ptp > 200) status = "noisy";
    return { status, ptp: s.ptp, mainsRatio };
  });
}
