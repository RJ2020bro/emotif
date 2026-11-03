// Environment director: decides which affective state the world renders.
//   mirror  the world shows the decoded state (biofeedback)
//   guide   the world shows the decoded state pulled toward a target state (closed-loop regulation)
//   sham    the world follows a pre-generated trajectory unrelated to the brain (control condition)
export type DirectorMode = "mirror" | "guide" | "sham";
export type VA = { v: number; a: number };

export const TARGETS: Record<string, { label: string; va: VA }> = {
  calm: { label: "Calm", va: { v: 0.6, a: -0.6 } },
  happy: { label: "Happy", va: { v: 0.7, a: 0.5 } },
  focused: { label: "Focused", va: { v: 0.3, a: 0.4 } },
  neutral: { label: "Neutral", va: { v: 0, a: 0 } },
};

/** Seeded PRNG so a sham trajectory can be reproduced from its seed (recorded with the session). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Director {
  mode: DirectorMode = "mirror";
  target: VA = { ...TARGETS.calm.va };
  gain = 0.5;
  maxRate = 0.25;             // state units per second
  env: VA = { v: 0, a: 0 };
  shamSeed = 1;
  private sham: VA = { v: 0, a: 0 };
  private rand = mulberry32(1);
  private gaussSpare: number | null = null;

  setSham(seed: number) { this.shamSeed = seed; this.rand = mulberry32(seed); this.sham = { v: 0, a: 0 }; this.gaussSpare = null; }

  private gauss(): number {
    if (this.gaussSpare !== null) { const g = this.gaussSpare; this.gaussSpare = null; return g; }
    const u = Math.max(1e-12, this.rand()), w = this.rand();
    const r = Math.sqrt(-2 * Math.log(u));
    this.gaussSpare = r * Math.sin(2 * Math.PI * w);
    return r * Math.cos(2 * Math.PI * w);
  }

  update(dt: number, decoded: VA | null): VA {
    // Ornstein-Uhlenbeck sham process advances regardless of mode so switching is seamless
    const th = 0.05, sg = 0.35;
    this.sham.v = Math.max(-1, Math.min(1, this.sham.v - th * this.sham.v * dt + sg * Math.sqrt(dt) * this.gauss()));
    this.sham.a = Math.max(-1, Math.min(1, this.sham.a - th * this.sham.a * dt + sg * Math.sqrt(dt) * this.gauss()));
    let want: VA;
    if (this.mode === "sham") want = this.sham;
    else if (!decoded) want = { v: 0, a: 0 };
    else if (this.mode === "guide") want = { v: decoded.v + this.gain * (this.target.v - decoded.v), a: decoded.a + this.gain * (this.target.a - decoded.a) };
    else want = decoded;
    const step = this.maxRate * dt;
    this.env = {
      v: this.env.v + Math.max(-step, Math.min(step, want.v - this.env.v)),
      a: this.env.a + Math.max(-step, Math.min(step, want.a - this.env.a)),
    };
    return this.env;
  }
}
