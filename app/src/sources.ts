// Signal sources. Each calls onMeta(fs, channels, label) once, then onRows(rows) with raw µV samples.
import { DEVICES, position } from "./montage";
import { parseEdf } from "./edf";
import { STATES, type StateName } from "./states";
import { mulberry32, type VA } from "./director";
import SYNTH_JSON from "../../emotif/data/synthetic_states.json";

export type SourceSink = {
  onMeta: (fs: number, channels: string[], label: string) => void;
  onRows: (rows: number[][]) => void;
  onStatus: (text: string, kind?: "ok" | "warn" | "error") => void;
  onEnd: () => void;
};
export interface Source { readonly kind: string; start(sink: SourceSink): Promise<void> | void; stop(): void }

// ------------------------------------------------------------------ simulator / virtual participant
// SYNTHETIC EEG: nothing the simulator produces is a measurement of a brain. Its parameters live in
// emotif/data/synthetic_states.json, shared with scripts/synth_insight.py (the offline generator).
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

export type SimMode = "tour" | "manual" | "participant" | "story";
export const SIM_MONTAGES: Record<string, { label: string; channels: string[] }> = {
  lab8: { label: "8-channel cap (Fp1 Fp2 F7 F3 F4 F8 O1 O2)", channels: ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "O1", "O2"] },
  muse: { label: DEVICES.muse.label, channels: DEVICES.muse.channels },
  emotiv_insight: { label: DEVICES.emotiv_insight.label, channels: DEVICES.emotiv_insight.channels },
  emotiv_epoc: { label: DEVICES.emotiv_epoc.label, channels: DEVICES.emotiv_epoc.channels },
  openbci_cyton: { label: DEVICES.openbci_cyton.label, channels: DEVICES.openbci_cyton.channels },
};
/** Sampling rate the simulator uses for a layout: the Emotiv Insight streams at 128 Hz, the others at 256 Hz. */
export const simRate = (montage: string): number => (montage === "emotiv_insight" ? 128 : 256);
const TOUR: VA[] = [{ v: 0.7, a: 0.6 }, { v: 0.65, a: -0.6 }, { v: -0.65, a: -0.55 }, { v: -0.7, a: 0.65 }];

const SIG_KEYS = ["delta", "theta", "thetaF", "alpha", "alphaP", "alphaF", "faa", "beta", "betaF", "betaR", "hibeta", "gamma",
  "alphaHz", "blinkRate", "blinkSize", "emg", "roll"] as const;
type SigKey = typeof SIG_KEYS[number];
/** A brain signature: log-amplitude gains per band and region, alpha peak shift (Hz), blink rate and size
 * (multipliers), jaw tension (0-1) and slow rolling eye movements (0-1). */
export type Sig = Record<SigKey, number>;
type Spec = {
  bands: { hz: number[]; uv: number[] };
  background: { slow_sd: number; slow_tau: number; fast_sd: number; fast_tau: number; white_sd: number };
  envelope: { sd: number; tau: number; lo: number; hi: number };
  jitter: { sd: number; tau: number };
  posterior_alpha_boost: number;
  blink: { uv: number; width_s: number; interval_s: [number, number]; task_interval_s: [number, number] };
  emg: { hp_hz: number; clench_uv: number; tension_uv: number; burst_s: [number, number]; gap_s: [number, number] };
  roll: { hz: number; uv: number; envelope_hz: number };
  focus: { alpha: number; alpha_global_share: number; beta: number; thetaF: number; tau: number };
  eyes_closed: { alpha: number; alphaP: number };
  va: { alpha: number; beta: number; hibeta: number; gamma: number; theta: number; faa: number };
  transition_tau: number;
  states: Record<StateName, Partial<Sig> & { va: [number, number] }>;
};
export const SYNTH = SYNTH_JSON as unknown as Spec;
const NEUTRAL: Sig = Object.fromEntries(SIG_KEYS.map(k => [k, k === "blinkRate" || k === "blinkSize" ? 1 : 0])) as Sig;
/** The six personal states' signatures (on top of the neutral resting signal). */
export const STATE_SIG = Object.fromEntries(STATES.map(s => [s, Object.fromEntries(SIG_KEYS.map(k => [k, SYNTH.states[s][k] ?? NEUTRAL[k]])) as Sig])) as Record<StateName, Sig>;
/** Where each state sits on the affect map (shown as the simulator's dot). */
export const STATE_VA = Object.fromEntries(STATES.map(s => [s, { v: SYNTH.states[s].va[0], a: SYNTH.states[s].va[1] }])) as Record<StateName, VA>;
/** Valence -> frontal alpha asymmetry; arousal -> alpha down, beta and gamma up, theta slightly down. */
export function vaSig(v: number, a: number): Sig {
  const k = SYNTH.va;
  return { ...NEUTRAL, alpha: k.alpha * a, beta: k.beta * a, hibeta: k.hibeta * a, gamma: k.gamma * a, theta: k.theta * a, faa: k.faa * v };
}

/** "story" mode: a scripted journey that loops after the baseline, exercising the world and its brain-controlled objects. */
export type StorySegment = { label: string; seconds: number; state?: StateName; va?: VA; focus?: number };
export const STORY: StorySegment[] = [
  { label: "Calm seas", seconds: 24, state: "calm" },
  { label: "Focus: steer clear", seconds: 10, va: { v: 0.3, a: 0.3 }, focus: 1 },
  { label: "Relax: calm the sea", seconds: 12, va: { v: 0.4, a: -0.3 }, focus: -1 },
  { label: "Stress: a storm rises", seconds: 24, state: "stress" },
  { label: "Anxiety: fog closes in", seconds: 24, state: "anxiety" },
  { label: "Anger: the forge erupts", seconds: 24, state: "anger" },
  { label: "Slow breaths: calm returns", seconds: 16, state: "calm" },
  { label: "Focus: steer clear", seconds: 10, va: { v: 0.5, a: 0.3 }, focus: 1 },
  { label: "Relax: calm the sea", seconds: 12, va: { v: 0.5, a: -0.2 }, focus: -1 },
  { label: "Low mood: the grey sea", seconds: 24, state: "low_mood" },
  { label: "Drowsy: the isle of sleep", seconds: 24, state: "drowsy" },
  { label: "Waking, content", seconds: 6, va: { v: 0.5, a: 0 } },
];
export const STORY_SECONDS = STORY.reduce((s, x) => s + x.seconds, 0);
/** Index of the story segment at `t` seconds after the baseline (the story loops). */
export function storyIndex(t: number): number {
  let u = ((t % STORY_SECONDS) + STORY_SECONDS) % STORY_SECONDS;
  for (let i = 0; i < STORY.length; i++) { if (u < STORY[i].seconds) return i; u -= STORY[i].seconds; }
  return STORY.length - 1;
}

/** What the virtual participant does during a protocol block (participant mode). */
export type SimBlock = StateName | "happy" | "eyes_open" | "eyes_closed" | "blinks" | "clench";
const TASK_BLOCK: Record<string, SimBlock> = {
  breathing: "calm", arithmetic: "stress", anticipation: "anxiety", recall_anger: "anger", recall_negative: "low_mood", drowsy: "drowsy",
  recall_positive: "happy", rest: "eyes_open", eyes_closed: "eyes_closed", blinks: "blinks", clench: "clench",
};
/** The simulator block for a protocol block_start value ({ task, state }); null when the person should simply
 * respond to the world (e.g. the closed-loop "watch" blocks). */
export function simBlockFor(info: unknown): SimBlock | null {
  const o = (info ?? {}) as { task?: unknown; state?: unknown };
  if (typeof o.state === "string" && (STATES as readonly string[]).includes(o.state)) return o.state as StateName;
  return typeof o.task === "string" && Object.prototype.hasOwnProperty.call(TASK_BLOCK, o.task) ? TASK_BLOCK[o.task] : null;
}

type ChState = {
  ar1: number; ar2: number; ph: number[]; env: number[]; fj: number; hp: Float64Array;
  front: number; post: number; side: number; temporal: number; blinkW: number; rollW: number;
};
type Biquad = { b0: number; b1: number; b2: number; a1: number; a2: number };
/** RBJ high-pass biquad. */
function highpass(fs: number, f0: number, q: number): Biquad {
  const w = 2 * Math.PI * f0 / fs, cw = Math.cos(w), al = Math.sin(w) / (2 * q), a0 = 1 + al;
  return { b0: (1 + cw) / 2 / a0, b1: -(1 + cw) / a0, b2: (1 + cw) / 2 / a0, a1: -2 * cw / a0, a2: (1 - al) / a0 };
}

/** Synthetic EEG with a 1/f background, six band oscillators whose amplitudes follow a brain signature, mains hum,
 * spontaneous blinks, and on-demand double blinks and jaw clenches (jaw muscle activity is high-passed above
 * 25 Hz and appears only on temporal electrodes).
 * The signature comes from, in order of precedence: the protocol block (participant mode), `state`, the story
 * segment (story mode), or the valence/arousal target. `focus` adds posterior alpha suppression (+1) or
 * enhancement (-1) on top, which is what interact.ts AlphaControl reads.
 * Modes: tour (four affect quadrants), manual (target set from outside), participant (affect drifts toward what
 * the world shows; protocol blocks produce their state, so in-app calibrations train on it), story (scripted loop). */
export class Simulator implements Source {
  readonly kind = "sim";
  fs = 256;
  mode: SimMode = "tour";
  target: VA = { v: 0, a: 0 };
  cur: VA = { v: 0, a: 0 };
  envState: VA = { v: 0, a: 0 };  // set by the environment director (participant mode)
  responsiveness = 0.08;          // per second
  calibSec = 30;
  montage = "emotiv_insight";   // the study headset
  /** Personal state to produce; overrides valence/arousal targeting (and the story) when set. */
  state: StateName | null = null;
  /** Brain control in [-1, 1]: +1 focused (posterior alpha well below baseline), -1 relaxed (well above). Smoothed over ~1 s. */
  focus = 0;
  /** Seed for a reproducible signal; null = Math.random. Read by start(). */
  seed: number | null = null;
  private blk: SimBlock | null = null; private blkT0 = 0;
  private t = 0; private timer: number | null = null; private t0 = 0; private emitted = 0;
  private st: ChState[] = []; private channels: string[] = [];
  private nextBlink = 5; private nextTaskBlink = 0; private blinkTimes: { t: number; amp: number }[] = []; private clenchUntil = -1;
  private burstUntil = -1; private nextBurst = 0;
  private rnd: () => number = Math.random; private spare: number | null = null;
  private sig: Sig = { ...NEUTRAL }; private focusCur = 0; private ecCur = 0; private active: StateName | null = null;
  private hum = 2; private rollPh = 0;
  private c = { a1: 0, s1: 0, a2: 0, s2: 0, envA: 0, envS: 0, fjA: 0, fjS: 0, hp: [] as Biquad[] };

  /** The protocol block now running (main.ts sets it on block_start / block_end); used in participant mode. */
  get block(): SimBlock | null { return this.blk; }
  set block(b: SimBlock | null) {
    if (b === this.blk) return;
    this.blk = b; this.blkT0 = this.t;
    if (b === "blinks") this.nextTaskBlink = this.t + 0.5;
  }
  /** Seconds of signal produced since start(). */
  get time(): number { return this.t; }
  /** The personal state being produced now (null = valence/arousal driven). */
  get activeState(): StateName | null { return this.active; }
  /** Smoothed focus level actually applied (-1 relaxed .. +1 focused). */
  get focusLevel(): number { return this.focusCur; }
  /** Story mode: what the simulated mind is doing now ("" in other modes). */
  get storySegment(): string {
    if (this.mode !== "story") return "";
    return this.t < this.calibSec ? "Resting baseline" : STORY[storyIndex(this.t - this.calibSec)].label;
  }

  private gauss(): number {
    if (this.spare !== null) { const g = this.spare; this.spare = null; return g; }
    let u = 0, v = 0;
    while (u === 0) u = this.rnd();
    while (v === 0) v = this.rnd();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
  private uni(lo: number, hi: number): number { return lo + (hi - lo) * this.rnd(); }

  start(sink: SourceSink) {
    this.stop();
    if (!SIM_MONTAGES[this.montage]) this.montage = "lab8";
    this.channels = SIM_MONTAGES[this.montage].channels;
    this.fs = simRate(this.montage);
    this.hum = this.montage === "emotiv_insight" ? 0.3 : 2;    // the Insight filters mains in the headset
    this.rnd = this.seed === null ? Math.random : mulberry32(this.seed);
    this.spare = null;
    const dt = 1 / this.fs, S = SYNTH, ar = (tau: number, sd: number) => { const a = Math.exp(-dt / tau); return [a, sd * Math.sqrt(1 - a * a)]; };
    [this.c.a1, this.c.s1] = ar(S.background.slow_tau, S.background.slow_sd);
    [this.c.a2, this.c.s2] = ar(S.background.fast_tau, S.background.fast_sd);
    [this.c.envA, this.c.envS] = ar(S.envelope.tau, S.envelope.sd);
    [this.c.fjA, this.c.fjS] = ar(S.jitter.tau, S.jitter.sd);
    this.c.hp = [highpass(this.fs, S.emg.hp_hz, 0.5411961), highpass(this.fs, S.emg.hp_hz, 1.3065630)];   // 4th-order Butterworth
    this.t = 0; this.blkT0 = 0;
    this.cur = { v: 0, a: 0 }; this.target = { v: 0, a: 0 }; this.sig = { ...NEUTRAL }; this.focusCur = 0; this.ecCur = 0; this.active = null;
    this.blinkTimes = []; this.clenchUntil = -1; this.burstUntil = -1; this.nextBurst = 0; this.nextTaskBlink = 0;
    this.nextBlink = this.uni(4, 10);
    this.rollPh = this.uni(0, 2 * Math.PI);
    this.st = this.channels.map(name => {
      const p = position(name) ?? [0, 0, 1];
      const front = clamp((p[1] - 0.2) / 0.5, 0, 1);
      return {
        ar1: 0, ar2: 0, ph: S.bands.hz.map(() => this.uni(0, 2 * Math.PI)), env: S.bands.hz.map(() => 1), fj: 0, hp: new Float64Array(4),
        front, post: clamp((-p[1] - 0.2) / 0.5, 0, 1), side: Math.abs(p[0]) > 0.1 ? Math.sign(p[0]) : 0,
        temporal: Math.abs(p[0]) > 0.7 && p[2] < 0.45 ? 1 : 0, blinkW: Math.max(0, (front - 0.3) / 0.7) ** 2,   // the eye field falls off quickly behind the brow
        rollW: p[0] * clamp((p[1] + 0.3) / 0.8, 0, 1),            // horizontal eye movements: opposite sign left and right
      };
    });
    sink.onMeta(this.fs, this.channels, `Simulator · ${SIM_MONTAGES[this.montage].label}`);
    sink.onStatus("Simulated signal", "ok");
    this.t0 = performance.now(); this.emitted = 0;
    this.timer = window.setInterval(() => {
      const due = Math.floor((performance.now() - this.t0) / 1000 * this.fs);
      let n = due - this.emitted;
      if (n <= 0) return;
      if (n > this.fs * 2) { n = this.fs * 2; }
      const rows: number[][] = new Array(n);
      for (let i = 0; i < n; i++) rows[i] = this.sample();
      this.emitted = due;
      sink.onRows(rows);
    }, 40);
  }
  stop() { if (this.timer !== null) window.clearInterval(this.timer); this.timer = null; }

  doubleBlink() { this.blinkTimes.push({ t: this.t + 0.05, amp: 1 }, { t: this.t + 0.4, amp: 1 }); }
  clench(seconds = 1.2) { this.clenchUntil = this.t + seconds; }

  /** The next sample (µV per channel). start() must have run; the timer calls this, tests may call it directly. */
  sample(): number[] {
    const S = SYNTH, dt = 1 / this.fs;
    this.t += dt;
    const t = this.t, calib = t < this.calibSec;
    let st: StateName | null = null, va: VA | null = null, focusDrive = this.focus, ec = 0, blinkTask = false, clenchTask = false;
    if (this.mode === "tour") va = calib ? { v: 0, a: 0 } : TOUR[Math.floor((t - this.calibSec) / 25) % TOUR.length];
    else if (this.mode === "story") {
      if (calib) va = { v: 0, a: 0 };
      else { const s = STORY[storyIndex(t - this.calibSec)]; st = s.state ?? null; va = s.va ?? null; focusDrive += s.focus ?? 0; }
    } else if (this.mode === "participant") {
      const b = this.blk;
      if (b === null) {
        if (!calib) {
          const g = this.responsiveness * dt;
          this.target.v = clamp(this.target.v + g * (this.envState.v - this.target.v) + 0.02 * Math.sqrt(dt) * this.gauss(), -1, 1);
          this.target.a = clamp(this.target.a + g * (this.envState.a - this.target.a) + 0.02 * Math.sqrt(dt) * this.gauss(), -1, 1);
        }
      } else if (b === "happy") va = { v: 0.7, a: 0.6 };
      else if (b === "eyes_closed") { va = { v: 0, a: -0.2 }; ec = 1; }
      else if (b === "eyes_open" || b === "blinks" || b === "clench") { va = { v: 0, a: 0 }; blinkTask = b === "blinks"; clenchTask = b === "clench"; }
      else st = b;
    }
    const fromBlock = st !== null && this.mode === "participant";
    if (!fromBlock && this.state) st = this.state;
    if (st) va = STATE_VA[st];
    if (va) { this.target.v = va.v; this.target.a = va.a; }
    this.active = st;

    const k = dt / S.transition_tau;
    this.cur.v += (this.target.v - this.cur.v) * k;
    this.cur.a += (this.target.a - this.cur.a) * k;
    const tgt = st ? STATE_SIG[st] : vaSig(this.target.v, this.target.a), sig = this.sig;
    for (const key of SIG_KEYS) sig[key] += (tgt[key] - sig[key]) * k;
    this.focusCur += (clamp(focusDrive, -1, 1) - this.focusCur) * Math.min(1, dt / S.focus.tau);
    this.ecCur += (ec - this.ecCur) * Math.min(1, dt / 0.5);

    // eyes: spontaneous blinks (none with the eyes closed), the blink task, slow rolling movements when drowsy
    const B = S.blink;
    if (this.ecCur < 0.5) {
      if (blinkTask) {
        if (t >= this.nextTaskBlink) { this.blinkTimes.push({ t, amp: 1 }); this.nextTaskBlink = t + this.uni(B.task_interval_s[0], B.task_interval_s[1]); }
      } else if (t >= this.nextBlink) {
        this.blinkTimes.push({ t, amp: sig.blinkSize });
        this.nextBlink = t + this.uni(B.interval_s[0], B.interval_s[1]) / Math.max(0.05, sig.blinkRate);
      }
    }
    let blink = 0;
    for (const bt of this.blinkTimes) {
      const d = t - bt.t - 0.15;
      if (Math.abs(d) < 0.4) blink += B.uv * bt.amp * Math.exp(-(d * d) / (2 * B.width_s * B.width_s));
    }
    while (this.blinkTimes.length && t - this.blinkTimes[0].t > 1) this.blinkTimes.shift();
    this.rollPh += 2 * Math.PI * S.roll.hz * dt;
    if (this.rollPh > 2 * Math.PI) this.rollPh -= 2 * Math.PI;
    const roll = S.roll.uv * Math.max(0, sig.roll) * (0.5 + 0.5 * Math.sin(2 * Math.PI * S.roll.envelope_hz * t)) * Math.sin(this.rollPh);

    // jaw: deliberate clenches, the clench task, and brief tension bursts (anger)
    const E = S.emg, tension = sig.emg;
    if (tension > 0.05 && t >= this.nextBurst) {
      this.burstUntil = t + this.uni(E.burst_s[0], E.burst_s[1]);
      this.nextBurst = this.burstUntil + this.uni(E.gap_s[0], E.gap_s[1]) / tension;
    }
    const clenching = t < this.clenchUntil || (clenchTask && (t - this.blkT0) % 4 < 2);
    const emgUv = clenching ? E.clench_uv : t < this.burstUntil ? E.tension_uv * clamp(tension, 0, 1) : 0;

    const F = S.focus, f = this.focusCur, EC = S.eyes_closed, ecl = this.ecCur, hz = S.bands.hz, uv = S.bands.uv, c = this.c;
    const twoPiDt = 2 * Math.PI * dt;
    return this.st.map(s => {
      s.ar1 = c.a1 * s.ar1 + c.s1 * this.gauss();
      s.ar2 = c.a2 * s.ar2 + c.s2 * this.gauss();
      s.fj = c.fjA * s.fj + c.fjS * this.gauss();
      const fr = s.front, po = s.post, sf = s.side * fr;
      const gBeta = sig.beta + sig.betaF * fr + sig.betaR * sf + F.beta * f;
      const gains = [
        sig.delta,
        sig.theta + sig.thetaF * fr + F.thetaF * f * fr,
        sig.alpha + sig.alphaP * po + sig.alphaF * fr + sig.faa * sf + S.posterior_alpha_boost * po
          + F.alpha * f * (F.alpha_global_share + (1 - F.alpha_global_share) * po) + ecl * (EC.alpha + EC.alphaP * po),
        gBeta,
        gBeta + sig.hibeta,
        sig.gamma,
      ];
      let x = s.ar1 + s.ar2 + S.background.white_sd * this.gauss();
      for (let b = 0; b < gains.length; b++) {
        s.env[b] = clamp(1 + c.envA * (s.env[b] - 1) + c.envS * this.gauss(), S.envelope.lo, S.envelope.hi);
        s.ph[b] += twoPiDt * (b === 2 ? hz[b] + sig.alphaHz : hz[b]) * (1 + s.fj);
        if (s.ph[b] > 6.283185307179586) s.ph[b] -= 6.283185307179586;
        x += uv[b] * Math.exp(gains[b]) * s.env[b] * Math.sin(s.ph[b]);
      }
      x += this.hum * Math.sin(2 * Math.PI * 60 * t);
      x += blink * s.blinkW + roll * s.rollW;
      if (s.temporal) {
        let y = this.gauss();
        for (let q = 0; q < 2; q++) {                   // direct form II transposed
          const h = c.hp[q], z = s.hp, o = h.b0 * y + z[2 * q];
          z[2 * q] = h.b1 * y - h.a1 * o + z[2 * q + 1];
          z[2 * q + 1] = h.b2 * y - h.a2 * o;
          y = o;
        }
        x += emgUv * y;
      }
      return x;
    });
  }
}

// ------------------------------------------------------------------ CSV playback
export class CsvPlayback implements Source {
  readonly kind = "csv";
  data: { rows: number[][]; fs: number; channels: string[]; name: string } | null = null;
  private timer: number | null = null; private pos = 0;

  load(text: string, fallbackFs: number, name: string) {
    const lines = text.split(/\r?\n/).filter(l => l.trim() !== "");
    if (lines.length < 3) throw new Error("The file has fewer than two data rows.");
    const header = lines[0].split(",").map(h => h.trim());
    const low = header.map(h => h.toLowerCase());
    const tcol = low.findIndex(h => ["time", "time_s", "timestamp", "timestamps", "t"].includes(h));
    const skip = (h: string) => ["valence_label", "arousal_label", "right aux", "aux"].includes(h) || /^(pred_|event|marker)/.test(h);
    const chIdx = header.map((_, i) => i).filter(i => i !== tcol && !skip(low[i]));
    if (!chIdx.length) throw new Error("No channel columns found.");
    const rows: number[][] = [], times: number[] = [];
    for (let li = 1; li < lines.length; li++) {
      const p = lines[li].split(",");
      if (p.length !== header.length) continue;
      const r = chIdx.map(i => Number(p[i]));
      if (r.some(v => !Number.isFinite(v))) throw new Error(`Row ${li + 1} has a non-numeric value in a channel column.`);
      rows.push(r);
      if (tcol >= 0) times.push(Number(p[tcol]));
    }
    let fs = fallbackFs;
    if (tcol >= 0) {
      const d: number[] = [];
      for (let i = 1; i < times.length; i++) d.push(times[i] - times[i - 1]);
      d.sort((x, y) => x - y);
      const med = d[d.length >> 1];
      if (!(med > 0)) throw new Error("The time column isn't increasing.");
      fs = 1 / med;
      if (Math.abs(fs - Math.round(fs)) < 0.02 * fs) fs = Math.round(fs);
    }
    if (!(fs >= 100)) throw new Error(`Sampling rate ${fs.toFixed(1)} Hz is too low; Emotif needs at least 100 Hz.`);
    if (rows.length < 2 * fs) throw new Error("The recording is shorter than one 2-second window.");
    this.data = { rows, fs, channels: chIdx.map(i => header[i]), name };
  }

  /** EDF, EDF+ or BDF (any headset's export). Voltage signals at the most common rate are kept. */
  loadEdf(buf: ArrayBuffer, name: string): string[] {
    const e = parseEdf(buf);
    if (!(e.fs >= 100)) throw new Error(`Sampling rate ${e.fs.toFixed(1)} Hz is too low; Emotif needs at least 100 Hz.`);
    if (e.rows.length < 2 * e.fs) throw new Error("The recording is shorter than one 2-second window.");
    this.data = { rows: e.rows, fs: e.fs, channels: e.channels, name };
    return e.skipped;
  }

  start(sink: SourceSink) {
    this.stop();
    const d = this.data;
    if (!d) { sink.onStatus("Choose a recording file first", "warn"); return; }
    sink.onMeta(d.fs, d.channels, `File · ${d.name}`);
    sink.onStatus(`Playing ${d.name}`, "ok");
    this.pos = 0;
    const t0 = performance.now();
    let emitted = 0;
    this.timer = window.setInterval(() => {
      const due = Math.floor((performance.now() - t0) / 1000 * d.fs);
      let n = Math.min(due - emitted, Math.round(d.fs * 2));
      if (n <= 0) return;
      emitted = due;
      const out: number[][] = [];
      while (n-- > 0) { out.push(d.rows[this.pos]); this.pos = (this.pos + 1) % d.rows.length; }
      sink.onRows(out);
    }, 40);
  }
  stop() { if (this.timer !== null) window.clearInterval(this.timer); this.timer = null; }
}

// ------------------------------------------------------------------ OpenBCI Cyton over Web Serial (Chrome / Edge)
/** Cyton (8 channels, 250 Hz) binary stream: 33-byte packets, 0xA0 | sample no. | 8 x 24-bit big-endian
 * signed | 6 aux bytes | 0xC0-0xCF. Default gain 24, so 1 count = 4.5 V / 24 / (2^23 - 1). */
export const CYTON_UV_PER_COUNT = 4.5 / 24 / (2 ** 23 - 1) * 1e6;
export class CytonParser {
  private buf = new Uint8Array(0);
  dropped = 0;           // bytes skipped while resynchronising
  lost = 0;              // missing sample numbers (radio drop-outs)
  private lastSample = -1;
  push(chunk: Uint8Array): number[][] {
    const b = new Uint8Array(this.buf.length + chunk.length);
    b.set(this.buf); b.set(chunk, this.buf.length);
    const rows: number[][] = [];
    let i = 0;
    while (i + 33 <= b.length) {
      if (b[i] !== 0xa0 || (b[i + 32] & 0xf0) !== 0xc0) { i++; this.dropped++; continue; }
      const sn = b[i + 1];
      if (this.lastSample >= 0) this.lost += (sn - this.lastSample - 1 + 256) % 256;
      this.lastSample = sn;
      const r = new Array(8);
      for (let c = 0; c < 8; c++) {
        const p = i + 2 + 3 * c;
        let v = (b[p] << 16) | (b[p + 1] << 8) | b[p + 2];
        if (v & 0x800000) v -= 0x1000000;
        r[c] = v * CYTON_UV_PER_COUNT;
      }
      rows.push(r);
      i += 33;
    }
    this.buf = b.slice(i);
    return rows;
  }
}

type SerialPortLike = {
  open(o: { baudRate: number }): Promise<void>;
  close(): Promise<void>;
  readable: ReadableStream<Uint8Array> | null;
  writable: WritableStream<Uint8Array> | null;
};

export class OpenBciSerial implements Source {
  readonly kind = "cyton";
  channels: string[] = DEVICES.openbci_cyton.channels.slice();
  private port: SerialPortLike | null = null;
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  private running = false;

  static supported(): boolean { return typeof navigator !== "undefined" && "serial" in navigator; }

  private async send(cmd: string) {
    const w = this.port?.writable?.getWriter();
    if (!w) return;
    try { await w.write(new TextEncoder().encode(cmd)); } finally { w.releaseLock(); }
  }

  async start(sink: SourceSink) {
    if (!OpenBciSerial.supported()) {
      sink.onStatus("This browser has no Web Serial. Use Chrome or Edge on a desktop, or connect through the Emotif server.", "error");
      return;
    }
    const serial = (navigator as unknown as { serial: { requestPort(): Promise<SerialPortLike> } }).serial;
    sink.onStatus("Choose the OpenBCI dongle in the port dialog…", "warn");
    try {
      this.port = await serial.requestPort();
      await this.port.open({ baudRate: 115200 });
      await this.send("v");                                // soft reset: default gain 24, all channels on
      await new Promise(r => setTimeout(r, 1500));
      await this.send("b");                                // start streaming
    } catch (e) {
      sink.onStatus(`Couldn't open the OpenBCI port: ${(e as Error).message}`, "error");
      await this.port?.close().catch(() => undefined);
      this.port = null;
      return;
    }
    sink.onMeta(250, this.channels, "OpenBCI Cyton · Web Serial");
    sink.onStatus("Streaming from OpenBCI Cyton", "ok");
    const parser = new CytonParser();
    this.running = true;
    this.reader = this.port.readable!.getReader();
    (async () => {
      try {
        while (this.running) {
          const { value, done } = await this.reader!.read();
          if (done) break;
          const rows = value ? parser.push(value) : [];
          if (rows.length) sink.onRows(rows);
        }
      } catch (e) {
        if (this.running) sink.onStatus(`OpenBCI stream stopped: ${(e as Error).message}`, "error");
      }
      if (this.running) sink.onEnd();
    })();
  }

  stop() {
    this.running = false;
    const port = this.port, reader = this.reader;
    this.port = null; this.reader = null;
    if (!port) return;
    (async () => {
      try { await reader?.cancel(); reader?.releaseLock(); } catch { /* already released */ }
      const w = port.writable?.getWriter();
      if (w) { try { await w.write(new TextEncoder().encode("s")); } catch { /* closing */ } w.releaseLock(); }
      await port.close().catch(() => undefined);
    })();
  }
}

// ------------------------------------------------------------------ Muse over Web Bluetooth (Chrome / Edge)
export class MuseBluetooth implements Source {
  readonly kind = "muse";
  private client: import("muse-js").MuseClient | null = null;
  private sub: { unsubscribe(): void } | null = null;

  static supported(): boolean { return typeof navigator !== "undefined" && "bluetooth" in navigator; }

  async start(sink: SourceSink) {
    if (!MuseBluetooth.supported()) {
      sink.onStatus("This browser has no Web Bluetooth. Use Chrome or Edge on desktop or Android, or connect through the Emotif server.", "error");
      return;
    }
    const { MuseClient, zipSamples } = await import("muse-js");
    const client = new MuseClient();
    this.client = client;
    sink.onStatus("Choose your Muse in the Bluetooth dialog…", "warn");
    try {
      await client.connect();
      await client.start();
    } catch (e) {
      this.client = null;
      sink.onStatus(`Couldn't connect to the Muse: ${(e as Error).message}`, "error");
      return;
    }
    sink.onMeta(256, ["TP9", "AF7", "AF8", "TP10"], `Muse · ${client.deviceName ?? "Bluetooth"}`);
    sink.onStatus(`Connected to ${client.deviceName ?? "Muse"}`, "ok");
    let batch: number[][] = [];
    let last = performance.now();
    this.sub = zipSamples(client.eegReadings).subscribe(s => {
      batch.push(s.data.slice(0, 4));
      const now = performance.now();
      if (batch.length >= 12 || now - last > 50) { sink.onRows(batch); batch = []; last = now; }
    });
    client.connectionStatus.subscribe(up => { if (!up) { sink.onStatus("Muse disconnected", "error"); sink.onEnd(); } });
  }
  stop() {
    this.sub?.unsubscribe(); this.sub = null;
    try { this.client?.disconnect(); } catch { /* already closed */ }
    this.client = null;
  }
}
