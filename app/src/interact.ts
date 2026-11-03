// Brain-only object control for the inner world. The command signal is alpha (8-13 Hz) power relative to the
// wearer's own resting baseline: alpha drops when attention is engaged ("focus") and rises when they let go
// ("relax"). Alpha sits below most jaw-muscle power and is not a blink band, and the control holds whenever the
// brain-only decoder holds for facial activity, so the face can't drive it. With the Insight, Pz carries the
// posterior alpha; other headsets use every observed region.
import { N_BANDS, REGIONS } from "./montage";

const R = REGIONS.length, ALPHA = 2, MASK0 = R * N_BANDS + 7 * N_BANDS;
const POSTERIOR = ["Pz", "P7", "P8", "O1", "O2"].map(r => REGIONS.indexOf(r));

export type Command = "idle" | "focus" | "relax";
export type ControlState = { level: number; command: Command; holdFor: number; held: boolean };

/** Mean baseline-relative alpha over posterior regions (all observed regions if none are posterior), or null. */
export function alphaZ(features: ArrayLike<number>): number | null {
  const obs = (r: number) => features[MASK0 + r] > 0.5;
  let rs = POSTERIOR.filter(obs);
  if (!rs.length) rs = Array.from({ length: R }, (_, r) => r).filter(obs);
  if (!rs.length) return null;
  return rs.reduce((a, r) => a + features[r * N_BANDS + ALPHA], 0) / rs.length;
}

export class AlphaControl {
  level = 0;                  // smoothed control level: + focus (alpha below baseline), - relax (alpha above)
  command: Command = "idle";
  private since = 0;          // seconds the level has stayed beyond the threshold in its direction
  private pending: Command = "idle";
  held = false;
  constructor(public threshold = 0.7, public dwell = 1.0, public tau = 1.5) {}

  /** One decoded window (every `dt` seconds). held: facial activity out of range or the window was rejected. */
  update(features: ArrayLike<number> | null, held: boolean, dt: number): ControlState {
    this.held = held;
    const z = features && !held ? alphaZ(features) : null;
    if (z !== null && Number.isFinite(z)) this.level += (1 - Math.exp(-dt / this.tau)) * (-z - this.level);
    // a command needs the level beyond the threshold for `dwell` seconds; it ends when the level comes back inside
    // half the threshold (hysteresis), so noise around the threshold doesn't flicker it
    const want: Command = held ? this.command : this.level > this.threshold ? "focus" : this.level < -this.threshold ? "relax" : "idle";
    if (want !== this.pending) { this.pending = want; this.since = 0; } else this.since += dt;
    if (this.command !== "idle" && !held) {
      const inside = this.command === "focus" ? this.level < this.threshold / 2 : this.level > -this.threshold / 2;
      if (inside) this.command = "idle";
    }
    if (this.command === "idle" && want !== "idle" && this.since >= this.dwell) this.command = want;
    return { level: this.level, command: this.command, holdFor: this.since, held };
  }

  reset() { this.level = 0; this.command = "idle"; this.since = 0; this.pending = "idle"; }
}
