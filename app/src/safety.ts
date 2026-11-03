// Photosensitive safety for the 3-D world. The scene's colours and lights follow the decoded state (v, a),
// so limiting how fast the state the scene shows may change bounds how fast its brightness can change.
//
// WCAG 2.3.1 / ITU-R BT.1702 "general flash": a pair of opposing relative-luminance changes of >= 10 % of the
// maximum where the darker image is below 0.80; more than three flashes in any one second fails.
// With every visual parameter linear (or slower) in v and a, a slew limit of MAX_RATE per second on each,
// and v, a in [-1, 1], relative luminance changes by at most MAX_RATE / 2 per second per axis. A 10 % change
// then takes >= 0.1 / MAX_RATE s (both axes moving together), so at MAX_RATE = 0.5 one flash (two changes)
// takes >= 0.4 s: at most 2.5 flashes per second even under adversarial input.

export const MAX_RATE = 0.5;           // state units per second, per axis
export const TAU = 1.2;                // first-order smoothing time constant (s)

/** One frame of smoothing followed by the slew limit. Returns the new displayed value. */
export function slewToward(cur: number, target: number, dt: number, maxRate = MAX_RATE, tau = TAU): number {
  const smoothed = cur + (target - cur) * (1 - Math.exp(-dt / tau));
  const step = Math.max(-maxRate * dt, Math.min(maxRate * dt, smoothed - cur));
  return cur + step;
}

/** Number of WCAG general flashes in a relative-luminance series sampled at fps, in the worst 1 s window.
 * A transition is a monotone run whose total change is >= 0.1; a flash is two opposing transitions
 * where the darker end is < 0.8. */
export function maxFlashesPerSecond(luma: number[], fps: number): number {
  type Tr = { t: number; dir: number; lo: number };
  const trs: Tr[] = [];
  let start = 0, dir = 0;
  const close = (end: number) => {
    const d = luma[end] - luma[start];
    if (Math.abs(d) >= 0.1) trs.push({ t: end / fps, dir: Math.sign(d), lo: Math.min(luma[start], luma[end]) });
  };
  for (let i = 1; i < luma.length; i++) {
    const s = Math.sign(luma[i] - luma[i - 1]);
    if (s === 0) continue;
    if (dir !== 0 && s !== dir) { close(i - 1); start = i - 1; }
    dir = s;
  }
  close(luma.length - 1);
  const flashes: number[] = [];
  for (let k = 1; k < trs.length; k++) {
    if (trs[k].dir !== trs[k - 1].dir && Math.min(trs[k].lo, trs[k - 1].lo) < 0.8) { flashes.push(trs[k].t); k++; }
  }
  let best = 0;
  for (let i = 0; i < flashes.length; i++) {
    let j = i;
    while (j < flashes.length && flashes[j] - flashes[i] < 1) j++;
    best = Math.max(best, j - i);
  }
  return best;
}
