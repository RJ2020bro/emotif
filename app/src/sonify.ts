// Sound display of the decoded state, for blind and low-vision users and for eyes-closed sessions.
// Valence -> pitch (+-3 semitones around G3) and the interval above it (minor third when negative, major
// when positive); arousal -> pulse rate (0.4-2.4 Hz) and brightness (low-pass 500-2500 Hz). Every parameter
// glides (time constant 0.8 s), so the sound never jumps. Starts only after a click, as browsers require.

export function soundParams(v: number, a: number) {
  const cv = Math.max(-1, Math.min(1, v)), ca = Math.max(-1, Math.min(1, a));
  const root = 196 * 2 ** (3 * cv / 12);
  const third = 3.5 + 0.5 * Math.tanh(3 * cv);             // 3 (minor) .. 4 (major) semitones
  const ar = (ca + 1) / 2;
  return { root, upper: root * 2 ** (third / 12), pulseHz: 0.4 + 2.0 * ar, cutoff: 500 + 2000 * ar };
}

export class Sonifier {
  private ctx: AudioContext | null = null;
  private oscs: OscillatorNode[] = [];
  private lfo: OscillatorNode | null = null;
  private filter: BiquadFilterNode | null = null;
  private master: GainNode | null = null;
  volume = 0.06;
  get running() { return this.ctx !== null; }

  start() {
    if (this.ctx) return;
    const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) throw new Error("This browser has no Web Audio.");
    const ctx = new Ctx();
    const master = ctx.createGain();
    master.gain.value = 0;
    master.gain.setTargetAtTime(this.volume, ctx.currentTime, 0.5);
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.Q.value = 0.7;
    const amp = ctx.createGain();
    amp.gain.value = 0.75;
    const lfo = ctx.createOscillator(), depth = ctx.createGain();
    lfo.frequency.value = 1;
    depth.gain.value = 0.25;                                 // gentle pulse: gain 0.5-1.0
    lfo.connect(depth).connect(amp.gain);
    const p = soundParams(0, 0);
    this.oscs = [p.root, p.upper].map(f => {
      const o = ctx.createOscillator();
      o.type = "triangle";
      o.frequency.value = f;
      o.connect(filter);
      o.start();
      return o;
    });
    filter.connect(amp).connect(master).connect(ctx.destination);
    lfo.start();
    Object.assign(this, { ctx, master, filter, lfo });
  }

  update(v: number, a: number) {
    const ctx = this.ctx;
    if (!ctx || !this.filter || !this.lfo) return;
    const p = soundParams(v, a), t = ctx.currentTime, tc = 0.8;
    this.oscs[0].frequency.setTargetAtTime(p.root, t, tc);
    this.oscs[1].frequency.setTargetAtTime(p.upper, t, tc);
    this.lfo.frequency.setTargetAtTime(p.pulseHz, t, tc);
    this.filter.frequency.setTargetAtTime(p.cutoff, t, tc);
  }

  stop() {
    const ctx = this.ctx;
    if (!ctx) return;
    this.master?.gain.setTargetAtTime(0, ctx.currentTime, 0.2);
    this.ctx = null;
    window.setTimeout(() => { ctx.close().catch(() => undefined); }, 800);
  }
}

/** Polite screen-reader announcements, rate-limited so a live stream doesn't flood the reader. */
export class Announcer {
  private el: HTMLElement;
  private last = new Map<string, number>();
  constructor(el: HTMLElement) { this.el = el; }
  say(key: string, text: string, minGapSec: number, now = performance.now() / 1000) {
    const prev = this.last.get(key);
    if (prev !== undefined && now - prev < minGapSec) return false;
    this.last.set(key, now);
    this.el.textContent = "";
    window.setTimeout(() => { this.el.textContent = text; }, 30);
    return true;
  }
}
