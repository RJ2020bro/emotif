// Procedural ground for the inner world: a seeded height function and a chunk mesher for each art style.
// Pure TypeScript (no three.js), so the same code runs in the page and in the Node tests. Every vertex
// carries a linear-light colour and two extras (material kind, height above its own ground) that the
// world shader uses for wind sway, glow and blooming.
import type { Feature, WorldSpec } from "./worldspec";

export const CHUNK = 24;
export const KIND = { ground: 0, sway: 1, glow: 2, bloom: 3, plain: 4, grass: 5, leaf: 6, bark: 7, stone: 8 } as const;
export const WATER_Y = { voxel: -0.12, field: 0 };

export type ChunkGeometry = {
  position: Float32Array; normal: Float32Array; color: Float32Array; extra: Float32Array; uv: Float32Array;
  index: Uint32Array | Uint16Array; features: number;
};
export type RGB = [number, number, number];

export function hexRgb(h: string): RGB {
  const n = parseInt(h.slice(1, 7), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
const toLin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const mul = (a: RGB, k: number): RGB => [a[0] * k, a[1] * k, a[2] * k];
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
const smooth = (e0: number, e1: number, x: number) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };

// ---------------------------------------------------------------- seeded noise
export function ihash(x: number, y: number, s: number): number {
  let h = Math.imul(s ^ 0x5bd1e995, 0x27d4eb2d) ^ Math.imul(x | 0, 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d) ^ Math.imul(y | 0, 0xc2b2ae35);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}
function vnoise(x: number, y: number, s: number): number {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), w = yf * yf * (3 - 2 * yf);
  const a = ihash(xi, yi, s), b = ihash(xi + 1, yi, s), c = ihash(xi, yi + 1, s), d = ihash(xi + 1, yi + 1, s);
  return (a + (b - a) * u + (c - a) * w + (a - b - c + d) * u * w) * 2 - 1;
}
function fbm(x: number, y: number, s: number, oct: number, gain: number): number {
  let sum = 0, amp = 1, f = 1, norm = 0;
  for (let i = 0; i < oct; i++) { sum += amp * vnoise(x * f, y * f, s + i * 1013); norm += amp; amp *= gain; f *= 2.03; }
  return sum / norm;
}

const SAND: RGB = [0.89, 0.83, 0.62], ROCK: RGB = [0.5, 0.49, 0.47], DIRT: RGB = [0.45, 0.33, 0.22];
const TRUNK: RGB = [0.42, 0.29, 0.18], PALMWOOD: RGB = [0.62, 0.5, 0.33], STONE: RGB = [0.62, 0.61, 0.57];
const WILT: RGB = [0.5, 0.42, 0.27];
const STEM: RGB = [0.92, 0.88, 0.8], DARK: RGB = [0.16, 0.15, 0.15], LIGHT: RGB = [1, 0.84, 0.5], CACTUS: RGB = [0.3, 0.56, 0.3];
const BASE_RATE: Record<Feature, number> = { trees: 0.32, pines: 0.32, palms: 0.18, crystals: 0.1, ruins: 0.04, mushrooms: 0.12, flowers: 0.5, lanterns: 0.05, rocks: 0.12, cacti: 0.1 };
const HIGH_OK: Feature[] = ["rocks", "crystals", "pines"];

// ---------------------------------------------------------------- terrain
export class Terrain {
  readonly spec: WorldSpec;
  readonly voxel: boolean;
  readonly waterY: number;
  private s: number;
  private lo: number;
  private hi: number;
  private pal: { low: RGB; high: RGB; peak: RGB; water: RGB; accent: RGB; foliage: RGB };

  constructor(spec: WorldSpec) {
    this.spec = spec;
    this.voxel = spec.style === "voxel";
    this.waterY = this.voxel ? WATER_Y.voxel : WATER_Y.field;
    this.s = (spec.seed * 7919 + 13) | 0;
    const p = spec.palette;
    this.pal = { low: hexRgb(p.low), high: hexRgb(p.high), peak: hexRgb(p.peak), water: hexRgb(p.water), accent: hexRgb(p.accent), foliage: hexRgb(p.foliage) };
    // sea level and the high end come from this seed's own height distribution, so "water" is the share of sea
    const vals: number[] = [];
    const sc = spec.terrain.scale;
    for (let i = 0; i < 48; i++) for (let j = 0; j < 48; j++) vals.push(this.raw((i - 24) * sc * 0.21 + 3.1, (j - 24) * sc * 0.21 - 1.7));
    vals.sort((a, b) => a - b);
    const q = (f: number) => vals[clamp(Math.round(f * (vals.length - 1)), 0, vals.length - 1)];
    this.lo = q(0.04 + 0.8 * spec.terrain.water);
    this.hi = Math.max(this.lo + 0.05, q(0.985));
  }

  private raw(x: number, z: number): number {
    const t = this.spec.terrain, sc = t.scale, s = this.s;
    let n = fbm(x / sc, z / sc, s, 5, 0.4 + 0.2 * t.roughness);
    const ridge = 1 - Math.abs(fbm(x / (sc * 0.45) + 17.3, z / (sc * 0.45) - 9.1, s + 7, 4, 0.5));
    n += t.roughness * 0.45 * (ridge - 0.6);
    if (t.islands) n = 0.45 * n + 0.8 * fbm(x / (sc * 2.4) - 40.7, z / (sc * 2.4) + 23.9, s + 13, 3, 0.5);
    return n;
  }

  /** Ground height in world units; sea level is 0. */
  height(x: number, z: number): number {
    const t = this.spec.terrain;
    const e = (this.raw(x, z) - this.lo) / (this.hi - this.lo);
    let h = e > 0 ? t.height * Math.pow(e, 1.3) : Math.max(-8, e * 12);
    if (t.terrace > 0 && h > 0) {
      const step = Math.max(2, t.height / 5), k = h / step, f = Math.floor(k);
      h += ((f + smooth(0.7, 1, k - f)) * step - h) * t.terrace;
    }
    return h;
  }

  /** Top of the voxel column whose block spans [ix, ix+1] x [iz, iz+1]. */
  column(ix: number, iz: number): number { return Math.round(this.height(ix + 0.5, iz + 0.5)); }

  /** Height the explorer and camera stand on (ground or water surface). */
  surface(x: number, z: number): number {
    const g = this.voxel ? this.column(Math.floor(x), Math.floor(z)) : this.height(x, z);
    return Math.max(g, this.waterY);
  }

  /** Nearest dry spot to (x, z), searched in rings out to 240 units. */
  spawn(x = 0, z = 0): { x: number; z: number } {
    for (let r = 0; r <= 240; r += 6) {
      const n = Math.max(1, Math.round(r / 3));
      for (let k = 0; k < n; k++) {
        const a = (k / n) * Math.PI * 2, px = x + Math.cos(a) * r, pz = z + Math.sin(a) * r;
        if (this.surface(px, pz) >= 1.5) return { x: px, z: pz };
      }
    }
    return { x, z };
  }

  groundColor(h: number, slope: number, jitter: number): RGB {
    const p = this.pal, t = this.spec.terrain, style = this.spec.style;
    let c: RGB;
    if (h < -0.3) c = mul(mix(p.low, p.water, 0.35), 0.75);
    else if (h < 1 && t.water > 0.02) c = mix(p.low, SAND, 0.55);
    else {
      const u = h / Math.max(2, t.height);
      c = u < 0.5 ? mix(p.low, p.high, u / 0.5) : mix(p.high, p.peak, smooth(0.62, 0.92, u));
    }
    c = mix(c, mix(p.high, ROCK, 0.6), smooth(0.9, 1.6, slope) * 0.8);
    c = mul(c, 0.95 + 0.1 * jitter);
    if (style === "dream") c = mix(c, [1, 1, 1], 0.12);
    if (style === "neon") c = mix(mul(c, 0.12), p.accent, 0.05);
    return c;
  }

  colors(mood: ChunkMood = NEUTRAL) {
    const p = this.pal, neon = this.spec.style === "neon";
    const wilt = 0.55 * Math.max(0, -clamp(mood.v, -1, 1)), lush = 0.15 * Math.max(0, clamp(mood.v, -1, 1));
    let foliage = this.spec.style === "dream" ? mix(p.foliage, [1, 1, 1], 0.12) : p.foliage;
    foliage = mix(mix(foliage, WILT, wilt), [0.2, 0.5, 0.16], lush);
    const k = neon ? 0.35 : 1;
    return {
      foliage: mul(foliage, k), pine: mul(mix(foliage, [0.08, 0.22, 0.13], 0.4), k), palmLeaf: mul(mix([0.35, 0.6, 0.25], foliage, 0.2), k),
      crystal: mix(p.accent, [1, 1, 1], 0.2), cap: p.accent, flowers: [p.accent, [1, 0.85, 0.3], [1, 0.6, 0.75], [0.96, 0.96, 0.96]] as RGB[],
    };
  }
}

// ---------------------------------------------------------------- mesh builder
type V3 = [number, number, number];
class MB {
  p: number[] = []; n: number[] = []; c: number[] = []; e: number[] = []; i: number[] = []; u: number[] = [];
  private vert(x: number, y: number, z: number, nx: number, ny: number, nz: number, lin: RGB, kind: number, ground: number, u = 0, v = 0) {
    this.p.push(x, y, z); this.n.push(nx, ny, nz); this.c.push(lin[0], lin[1], lin[2]); this.e.push(kind, Math.max(0, y - ground)); this.u.push(u, v);
  }
  /** A textured, double-sided square card (leaf clusters) centred on c, spanned by unit vectors a and b; its normal
   * points away from `from` (the crown centre) so light wraps around the crown. */
  card(c: V3, a: V3, b: V3, half: number, from: V3, col: RGB, kind: number, ground: number) {
    const L: RGB = [toLin(col[0]), toLin(col[1]), toLin(col[2])], base = this.p.length / 3;
    const nx = c[0] - from[0], ny = c[1] - from[1] + 0.35, nz = c[2] - from[2], nl = Math.hypot(nx, ny, nz) || 1;
    const corners: [number, number][] = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    for (const [s, t] of corners) {
      this.vert(c[0] + (a[0] * s + b[0] * t) * half, c[1] + (a[1] * s + b[1] * t) * half, c[2] + (a[2] * s + b[2] * t) * half,
        nx / nl, ny / nl, nz / nl, L, kind, ground, (s + 1) / 2, (t + 1) / 2);
    }
    this.i.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  quad(v: V3[], nrm: V3, col: RGB, kind: number, ground: number) {
    const L: RGB = [toLin(col[0]), toLin(col[1]), toLin(col[2])], b = this.p.length / 3;
    for (const q of v) this.vert(q[0], q[1], q[2], nrm[0], nrm[1], nrm[2], L, kind, ground);
    this.i.push(b, b + 1, b + 2, b, b + 2, b + 3);
  }
  /** A triangle with a normal per vertex, or the face normal (oriented away from `out`) when `nrm` is null. */
  tri(a: V3, b: V3, c: V3, nrm: [V3, V3, V3] | null, col: RGB, kind: number, ground: number, out: V3 | null) {
    const L: RGB = [toLin(col[0]), toLin(col[1]), toLin(col[2])], base = this.p.length / 3;
    let ns = nrm;
    if (!ns) {
      const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], wx = c[0] - a[0], wy = c[1] - a[1], wz = c[2] - a[2];
      let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
      const len = Math.hypot(nx, ny, nz) || 1;
      nx /= len; ny /= len; nz /= len;
      const cx = (a[0] + b[0] + c[0]) / 3, cy = (a[1] + b[1] + c[1]) / 3, cz = (a[2] + b[2] + c[2]) / 3;
      const flip = out ? (cx - out[0]) * nx + (cy - out[1]) * ny + (cz - out[2]) * nz < 0 : ny < 0;
      if (flip) { nx = -nx; ny = -ny; nz = -nz; }
      ns = [[nx, ny, nz], [nx, ny, nz], [nx, ny, nz]];
    }
    [a, b, c].forEach((q, k) => this.vert(q[0], q[1], q[2], ns![k][0], ns![k][1], ns![k][2], L, kind, ground));
    this.i.push(base, base + 1, base + 2);
  }
  box(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, col: RGB, kind: number, ground: number) {
    this.quad([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], [0, 1, 0], col, kind, ground);
    this.quad([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], [1, 0, 0], col, kind, ground);
    this.quad([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]], [-1, 0, 0], col, kind, ground);
    this.quad([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0, 0, 1], col, kind, ground);
    this.quad([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], [0, 0, -1], col, kind, ground);
  }
  /** Frustum (r1 > 0) or cone (r1 = 0) standing on (x, y, z). */
  cone(x: number, y: number, z: number, r0: number, r1: number, h: number, sides: number, rot: number, col: RGB, kind: number, ground: number, smoothN: boolean) {
    const slope = (r0 - r1) / h;
    const nAt = (a: number): V3 => { const l = Math.hypot(1, slope); return [Math.cos(a) / l, slope / l, Math.sin(a) / l]; };
    for (let k = 0; k < sides; k++) {
      const a0 = rot + (k / sides) * Math.PI * 2, a1 = rot + ((k + 1) / sides) * Math.PI * 2, am = (a0 + a1) / 2;
      const b0: V3 = [x + Math.cos(a0) * r0, y, z + Math.sin(a0) * r0], b1: V3 = [x + Math.cos(a1) * r0, y, z + Math.sin(a1) * r0];
      const n0 = smoothN ? nAt(a0) : nAt(am), n1 = smoothN ? nAt(a1) : nAt(am);
      if (r1 <= 1e-3) {
        this.tri(b0, b1, [x, y + h, z], [n0, n1, smoothN ? [0, 1, 0] : nAt(am)], col, kind, ground, null);
      } else {
        const t0: V3 = [x + Math.cos(a0) * r1, y + h, z + Math.sin(a0) * r1], t1: V3 = [x + Math.cos(a1) * r1, y + h, z + Math.sin(a1) * r1];
        this.tri(b0, b1, t1, [n0, n1, n1], col, kind, ground, null);
        this.tri(b0, t1, t0, [n0, n1, n0], col, kind, ground, null);
        this.tri(t0, t1, [x, y + h, z], [[0, 1, 0], [0, 1, 0], [0, 1, 0]], col, kind, ground, null);
      }
    }
  }
  /** Ellipsoid from a once-subdivided octahedron with a seeded wobble; faceted unless smoothN. */
  blob(x: number, y: number, z: number, rx: number, ry: number, rz: number, col: RGB, kind: number, ground: number, smoothN: boolean, seed: number, fine = false) {
    for (const [a, b, c] of fine ? BLOB2 : BLOB) {
      const P = (d: V3): V3 => {
        const w = 1 + 0.16 * (ihash(Math.round(d[0] * 1000), Math.round(d[1] * 1000) + Math.round(d[2] * 1000) * 7, seed) - 0.5);
        return [x + d[0] * rx * w, y + d[1] * ry * w, z + d[2] * rz * w];
      };
      this.tri(P(a), P(b), P(c), smoothN ? [a, b, c] : null, col, kind, ground, [x, y, z]);
    }
  }
  finish(features: number): ChunkGeometry {
    const nv = this.p.length / 3;
    return {
      position: Float32Array.from(this.p), normal: Float32Array.from(this.n), color: Float32Array.from(this.c), extra: Float32Array.from(this.e),
      uv: Float32Array.from(this.u), index: nv > 65535 ? Uint32Array.from(this.i) : Uint16Array.from(this.i), features,
    };
  }
}

const BLOB: [V3, V3, V3][] = (() => {
  const O: V3[] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const F = [[0, 2, 4], [4, 2, 1], [1, 2, 5], [5, 2, 0], [4, 3, 0], [1, 3, 4], [5, 3, 1], [0, 3, 5]];
  const nm = (a: V3, b: V3): V3 => { const m: V3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]; const l = Math.hypot(...m); return [m[0] / l, m[1] / l, m[2] / l]; };
  const out: [V3, V3, V3][] = [];
  for (const [i, j, k] of F) {
    const a = O[i], b = O[j], c = O[k], ab = nm(a, b), bc = nm(b, c), ca = nm(c, a);
    out.push([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]);
  }
  return out;
})();

const BLOB2: [V3, V3, V3][] = (() => {     // one more subdivision (128 faces) for rounder crowns
  const nm = (a: V3, b: V3): V3 => { const m: V3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]; const l = Math.hypot(...m); return [m[0] / l, m[1] / l, m[2] / l]; };
  const out: [V3, V3, V3][] = [];
  for (const [a, b, c] of BLOB) { const ab = nm(a, b), bc = nm(b, c), ca = nm(c, a); out.push([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]); }
  return out;
})();

// ---------------------------------------------------------------- chunks
/** Decoded state when a chunk is first generated: valence and arousal in [-1, 1]. The land itself is the same for
 * any state (so chunk borders always meet); what grows on it is not. */
export type ChunkMood = { v: number; a: number };
export const NEUTRAL: ChunkMood = { v: 0, a: 0 };

/** How the state at generation time scales each feature's frequency. Pleasant: flowers, trees, lanterns; unpleasant:
 * bare rock and ruins; high arousal: crystals, pines and rock. */
export function moodRates(kind: Feature, m: ChunkMood): number {
  const v = clamp(m.v, -1, 1), a = clamp(m.a, -1, 1), pos = Math.max(0, v), neg = Math.max(0, -v), up = Math.max(0, a);
  switch (kind) {
    case "flowers": return 1 + 1.6 * pos - 0.6 * neg;
    case "trees": case "palms": case "mushrooms": return 1 + 0.5 * v;
    case "pines": return 1 + 0.4 * up;
    case "rocks": return 1 + 1.5 * neg + 0.6 * up;
    case "crystals": return 1 + 0.9 * up;
    case "ruins": return 1 + 0.9 * neg;
    case "lanterns": return 1 + pos;
    default: return 1;
  }
}

export function buildChunk(T: Terrain, cx: number, cz: number, mood: ChunkMood = NEUTRAL): ChunkGeometry {
  const mb = new MB();
  if (T.voxel) voxelGround(mb, T, cx, cz);
  else fieldGround(mb, T, cx, cz, T.spec.style === "lowpoly" ? 2 : 1, T.spec.style !== "dream" && T.spec.style !== "natural");
  let features = placeFeatures(mb, T, cx, cz, mood);
  if (T.spec.style === "natural") features += grass(mb, T, cx, cz, mood);
  return mb.finish(features);
}

const WITHERED: RGB = [0.52, 0.45, 0.3];
/** Grass blades (one tapered triangle each) on gentle, dry, not-too-high ground. Denser and greener in calm or
 * pleasant states, sparse and straw-coloured in unpleasant ones. Returns the number of tufts (25 blades each). */
function grass(mb: MB, T: Terrain, cx: number, cz: number, m: ChunkMood): number {
  const s = T.spec.seed + 77, H = Math.max(2, T.spec.terrain.height), v = clamp(m.v, -1, 1), a = clamp(m.a, -1, 1);
  const per = 2.4 * (0.6 + 0.4 * (v + 1) / 2);                       // blades per square unit
  const n = Math.round(CHUNK * CHUNK * per), x0 = cx * CHUNK, z0 = cz * CHUNK;
  const leaf = T.colors().foliage;
  const green = mix(leaf, [0.36, 0.55, 0.2], 0.45), dry = mix(green, WITHERED, clamp(0.25 + 0.6 * Math.max(0, -v), 0, 0.85));
  const tall = 0.55 + 0.35 * (1 - Math.abs(a));
  let made = 0;
  for (let k = 0; k < n; k++) {
    const gx = x0 + ihash(cx * 7919 + k, cz, s) * CHUNK, gz = z0 + ihash(cx, cz * 7919 + k, s + 1) * CHUNK;
    const y = T.height(gx, gz);
    if (y < 0.9 || y / H > 0.62) continue;
    const e = 0.5, sl = Math.hypot(T.height(gx + e, gz) - T.height(gx - e, gz), T.height(gx, gz + e) - T.height(gx, gz - e)) / (2 * e);
    if (sl > 0.75) continue;
    const r = ihash(k, cx - cz, s + 2), th = r * Math.PI, h = tall * (0.55 + 0.8 * ihash(k, cz + cx, s + 3)), w = 0.07 + 0.05 * r;
    const lean = (ihash(k, cx, s + 4) - 0.5) * 0.5 * h, dx = Math.cos(th) * w, dz = Math.sin(th) * w;
    const col = mix(green, dry, ihash(k, cz, s + 5) * 0.8);
    mb.tri([gx - dx, y - 0.05, gz - dz], [gx + dx, y - 0.05, gz + dz], [gx + lean * Math.sin(th), y + h, gz - lean * Math.cos(th)],
      [[0, 1, 0], [0, 1, 0], [0, 1, 0]], col, KIND.grass, y - 0.05, null);
    made++;
  }
  return Math.round(made / 25);
}

function voxelGround(mb: MB, T: Terrain, cx: number, cz: number) {
  const W = CHUNK + 2, x0 = cx * CHUNK, z0 = cz * CHUNK, s = T.spec.seed;
  const H = new Int32Array(W * W);
  for (let j = 0; j < W; j++) for (let i = 0; i < W; i++) H[j * W + i] = T.column(x0 + i - 1, z0 + j - 1);
  for (let j = 1; j <= CHUNK; j++) for (let i = 1; i <= CHUNK; i++) {
    const h = H[j * W + i], x = x0 + i - 1, z = z0 + j - 1;
    const nb: [number, V3][] = [[H[j * W + i + 1], [1, 0, 0]], [H[j * W + i - 1], [-1, 0, 0]], [H[(j + 1) * W + i], [0, 0, 1]], [H[(j - 1) * W + i], [0, 0, -1]]];
    let slope = 0;
    for (const [hn] of nb) slope = Math.max(slope, Math.abs(h - hn));
    const top = T.groundColor(h, slope / 2, ihash(x, z, s));
    mb.quad([[x, h, z], [x + 1, h, z], [x + 1, h, z + 1], [x, h, z + 1]], [0, 1, 0], top, KIND.ground, h);
    const sandy = h < 1 || h / Math.max(2, T.spec.terrain.height) > 0.8;
    for (const [hn, d] of nb) {
      for (let y = hn; y < h; y++) {
        const depth = h - 1 - y;
        const col = sandy ? mul(top, 0.92) : depth === 0 ? mix(top, DIRT, 0.45) : depth < 4 ? mix(DIRT, top, 0.1) : mix(ROCK, top, 0.15);
        const xa = d[0] === 1 ? x + 1 : x, za = d[2] === 1 ? z + 1 : z;
        const v: V3[] = d[0] !== 0
          ? [[xa, y, z], [xa, y + 1, z], [xa, y + 1, z + 1], [xa, y, z + 1]]
          : [[x, y, za], [x + 1, y, za], [x + 1, y + 1, za], [x, y + 1, za]];
        mb.quad(v, d, col, KIND.ground, h);
      }
    }
  }
}

function fieldGround(mb: MB, T: Terrain, cx: number, cz: number, step: number, flat: boolean) {
  const n = CHUNK / step, x0 = cx * CHUNK, z0 = cz * CHUNK, s = T.spec.seed, lowpoly = step > 1;
  const P: V3[] = [];
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) {
    const gx = x0 / step + i, gz = z0 / step + j;     // global vertex index: jitter agrees across chunk borders
    const jx = lowpoly ? (ihash(gx, gz, s + 5) - 0.5) * 0.7 * step : 0, jz = lowpoly ? (ihash(gx, gz, s + 6) - 0.5) * 0.7 * step : 0;
    const x = gx * step + jx, z = gz * step + jz;
    P.push([x, Math.max(-8, T.height(x, z)), z]);
  }
  const at = (i: number, j: number) => P[j * (n + 1) + i];
  if (!flat) {
    const base = mb.p.length / 3;
    for (const q of P) {
      const e = 0.5, hx = T.height(q[0] + e, q[2]) - T.height(q[0] - e, q[2]), hz = T.height(q[0], q[2] + e) - T.height(q[0], q[2] - e);
      const l = Math.hypot(hx, 1, hz);
      const col = T.groundColor(q[1], Math.hypot(hx, hz), ihash(Math.round(q[0]), Math.round(q[2]), s));
      const L: RGB = [toLin(col[0]), toLin(col[1]), toLin(col[2])];
      mb.p.push(q[0], q[1], q[2]); mb.n.push(-hx / l, 1 / l, -hz / l); mb.c.push(L[0], L[1], L[2]); mb.e.push(KIND.ground, 0); mb.u.push(0, 0);
    }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const a = base + j * (n + 1) + i, b = a + 1, d = a + n + 1, c = d + 1;
      mb.i.push(a, d, c, a, c, b);
    }
    return;
  }
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
    const flipDiag = ihash(x0 + i, z0 + j, s + 9) < 0.5;
    const tris: [V3, V3, V3][] = flipDiag ? [[a, d, c], [a, c, b]] : [[a, d, b], [b, d, c]];
    tris.forEach(([p, q, r], k) => {
      const ux = q[0] - p[0], uy = q[1] - p[1], uz = q[2] - p[2], wx = r[0] - p[0], wy = r[1] - p[1], wz = r[2] - p[2];
      const ny = Math.abs(uz * wx - ux * wz), len = Math.hypot(uy * wz - uz * wy, ny, ux * wy - uy * wx) || 1;
      const slope = Math.sqrt(Math.max(0, 1 - (ny / len) ** 2)) / Math.max(0.05, ny / len);
      const col = T.groundColor((p[1] + q[1] + r[1]) / 3, slope, ihash(x0 + i * 2 + k, z0 + j, s + 3));
      mb.tri(p, q, r, null, col, KIND.ground, Math.min(p[1], q[1], r[1]), null);
    });
  }
}

const G = 3;   // feature cell size; CHUNK is a multiple of it
function placeFeatures(mb: MB, T: Terrain, cx: number, cz: number, mood: ChunkMood): number {
  const spec = T.spec, s = spec.seed, feats = spec.features.filter(f => f.density > 0);
  if (!feats.length) return 0;
  const rates = feats.map(f => BASE_RATE[f.kind] * f.density * Math.max(0, moodRates(f.kind, mood)));
  const total = rates.reduce((x, y) => x + y, 0), k = total > 0.85 ? 0.85 / total : 1;
  const C = T.colors(mood), n = CHUNK / G;
  let count = 0;
  for (let gj = 0; gj < n; gj++) for (let gi = 0; gi < n; gi++) {
    const gx = cx * n + gi, gz = cz * n + gj;
    let r = ihash(gx, gz, s + 101);
    let kind: Feature | null = null;
    for (let f = 0; f < feats.length; f++) { const w = rates[f] * k; if (r < w) { kind = feats[f].kind; break; } r -= w; }
    if (!kind) continue;
    const u1 = ihash(gx, gz, s + 202), u2 = ihash(gx, gz, s + 303), size = ihash(gx, gz, s + 404), r2 = ihash(gx, gz, s + 505);
    let x: number, z: number, y: number;
    if (T.voxel) { x = gx * G + Math.floor(u1 * G); z = gz * G + Math.floor(u2 * G); y = T.column(x, z); }
    else { x = gx * G + u1 * G; z = gz * G + u2 * G; y = T.height(x, z); }
    if (y < (T.voxel ? 1 : 0.6)) continue;
    const u = y / Math.max(2, spec.terrain.height);
    if (u > 0.8 && !HIGH_OK.includes(kind)) continue;
    if (kind === "palms" && y > 4 + 0.15 * spec.terrain.height) continue;
    if (!T.voxel) {
      const e = 0.6, sl = Math.hypot(T.height(x + e, z) - T.height(x - e, z), T.height(x, z + e) - T.height(x, z - e)) / (2 * e);
      if (sl > 1.1 && kind !== "rocks" && kind !== "crystals") continue;
    }
    if (T.voxel) voxelFeature(mb, kind, x, y, z, size, r2, C, s + gx * 31 + gz);
    else if (spec.style === "natural") naturalFeature(mb, kind, x, y, z, size, r2, C, s + gx * 31 + gz);
    else smoothFeature(mb, kind, x, y, z, size, r2, C, spec.style === "dream", s + gx * 31 + gz);
    count++;
  }
  return count;
}

type Cols = ReturnType<Terrain["colors"]>;
function voxelFeature(mb: MB, kind: Feature, x: number, y: number, z: number, s: number, r2: number, C: Cols, seed: number) {
  const K = KIND;
  switch (kind) {
    case "trees": {
      const th = 3 + Math.floor(s * 3);
      mb.box(x, y, z, x + 1, y + th, z + 1, TRUNK, K.ground, y);
      mb.box(x - 1, y + th - 1, z - 1, x + 2, y + th + 2, z + 2, C.foliage, K.sway, y);
      mb.box(x, y + th + 2, z, x + 1, y + th + 3, z + 1, C.foliage, K.sway, y);
      break;
    }
    case "pines": {
      const th = 2 + Math.floor(s * 2);
      mb.box(x, y, z, x + 1, y + th, z + 1, TRUNK, K.ground, y);
      mb.box(x - 1, y + th, z - 1, x + 2, y + th + 2, z + 2, C.pine, K.sway, y);
      mb.box(x, y + th + 2, z, x + 1, y + th + 4, z + 1, C.pine, K.sway, y);
      break;
    }
    case "palms": {
      const th = 4 + Math.floor(s * 2);
      mb.box(x, y, z, x + 1, y + th, z + 1, PALMWOOD, K.ground, y);
      mb.box(x - 2, y + th, z, x + 3, y + th + 0.5, z + 1, C.palmLeaf, K.sway, y);
      mb.box(x, y + th + 0.02, z - 2, x + 1, y + th + 0.52, z + 3, C.palmLeaf, K.sway, y);
      break;
    }
    case "crystals":
      mb.box(x + 0.2, y, z + 0.2, x + 0.8, y + 2 + 3 * s, z + 0.8, C.crystal, K.glow, y);
      mb.box(x - 0.3, y, z + 0.5, x + 0.1, y + 1.2 + s, z + 0.9, C.crystal, K.glow, y);
      break;
    case "ruins":
      if (r2 > 0.5) {
        mb.box(x, y - 3, z, x + 1, y + 3, z + 1, STONE, K.ground, y);
        mb.box(x + 3, y - 3, z, x + 4, y + 3, z + 1, STONE, K.ground, y);
        mb.box(x, y + 3, z, x + 4, y + 4, z + 1, STONE, K.ground, y);
      } else mb.box(x, y - 1, z, x + 1, y + 2 + Math.floor(s * 3), z + 1, STONE, K.ground, y);
      break;
    case "mushrooms": {
      const sh = 1 + s;
      mb.box(x + 0.35, y, z + 0.35, x + 0.65, y + sh, z + 0.65, STEM, K.plain, y);
      mb.box(x - 0.4, y + sh, z - 0.4, x + 1.4, y + sh + 0.6, z + 1.4, r2 > 0.5 ? C.cap : [0.85, 0.2, 0.2], K.bloom, y);
      break;
    }
    case "flowers":
      for (let k = 0; k < 3; k++) {
        const fx = x + 0.15 + ihash(seed, k, 1) * 0.6, fz = z + 0.15 + ihash(seed, k, 2) * 0.6, col = C.flowers[Math.floor(ihash(seed, k, 3) * 4)];
        mb.box(fx - 0.04, y, fz - 0.04, fx + 0.04, y + 0.45, fz + 0.04, [0.3, 0.55, 0.25], K.bloom, y);
        mb.box(fx - 0.13, y + 0.45, fz - 0.13, fx + 0.13, y + 0.71, fz + 0.13, col, K.bloom, y);
      }
      break;
    case "lanterns":
      mb.box(x + 0.42, y, z + 0.42, x + 0.58, y + 2.2, z + 0.58, DARK, K.plain, y);
      mb.box(x + 0.25, y + 2.2, z + 0.25, x + 0.75, y + 2.8, z + 0.75, LIGHT, K.glow, y);
      mb.box(x + 0.2, y + 2.8, z + 0.2, x + 0.8, y + 2.9, z + 0.8, DARK, K.plain, y);
      break;
    case "rocks":
      mb.box(x, y, z, x + 1, y + 1, z + 1, STONE, K.ground, y);
      if (r2 > 0.6) mb.box(x + 1, y - 1, z, x + 2, y + 1, z + 1, STONE, K.ground, y);
      break;
    case "cacti": {
      const th = 3 + Math.floor(s * 2);
      mb.box(x, y, z, x + 1, y + th, z + 1, CACTUS, K.ground, y);
      if (r2 > 0.4) {
        mb.box(x + 1, y + 1.5, z + 0.25, x + 1.6, y + 2, z + 0.75, CACTUS, K.plain, y);
        mb.box(x + 1.1, y + 2, z + 0.25, x + 1.6, y + 3.2, z + 0.75, CACTUS, K.plain, y);
      }
      break;
    }
  }
}

function smoothFeature(mb: MB, kind: Feature, x: number, y: number, z: number, s: number, r2: number, C: Cols, sm: boolean, seed: number) {
  const K = KIND, sc = 0.8 + 0.5 * s, rot = r2 * Math.PI * 2;
  switch (kind) {
    case "trees":
      mb.cone(x, y - 0.5, z, 0.28 * sc, 0.2 * sc, 3.1 * sc, 6, rot, TRUNK, K.plain, y, sm);
      mb.blob(x, y + 3 * sc, z, 1.5 * sc, 1.3 * sc, 1.5 * sc, C.foliage, K.sway, y, sm, seed);
      break;
    case "pines":
      mb.cone(x, y - 0.5, z, 0.22 * sc, 0.18 * sc, 1.5 * sc, 5, rot, TRUNK, K.plain, y, sm);
      for (let k = 0; k < 3; k++) mb.cone(x, y + (0.8 + 1.1 * k) * sc, z, (1.6 - 0.4 * k) * sc, 0, 1.8 * sc, 7, rot + k, C.pine, K.sway, y, sm);
      break;
    case "palms":
      mb.cone(x, y - 0.5, z, 0.25 * sc, 0.17 * sc, 5 * sc, 6, rot, PALMWOOD, K.plain, y, sm);
      mb.cone(x, y + 4.3 * sc, z, 2.4 * sc, 0, 0.7 * sc, 7, rot, C.palmLeaf, K.sway, y, sm);
      break;
    case "crystals":
      for (let k = 0; k < 3; k++) {
        const a = rot + k * 2.1, d = k === 0 ? 0 : 0.45 * sc;
        mb.cone(x + Math.cos(a) * d, y - 0.4, z + Math.sin(a) * d, (0.38 - 0.08 * k) * sc, 0, (2.6 - 0.6 * k) * sc, 5, a, C.crystal, K.glow, y, false);
      }
      break;
    case "ruins":
      if (r2 > 0.5) {
        mb.box(x - 0.5, y - 3, z - 0.5, x + 0.5, y + 3, z + 0.5, STONE, K.plain, y);
        mb.box(x + 2.5, y - 3, z - 0.5, x + 3.5, y + 3, z + 0.5, STONE, K.plain, y);
        mb.box(x - 0.6, y + 3, z - 0.6, x + 3.6, y + 3.8, z + 0.6, STONE, K.plain, y);
      } else mb.box(x - 0.5, y - 1.5, z - 0.5, x + 0.5, y + 1.5 + 2.5 * s, z + 0.5, STONE, K.plain, y);
      break;
    case "mushrooms":
      mb.cone(x, y - 0.3, z, 0.22 * sc, 0.18 * sc, 1.6 * sc, 6, rot, STEM, K.plain, y, sm);
      mb.blob(x, y + 1.3 * sc, z, 0.9 * sc, 0.45 * sc, 0.9 * sc, r2 > 0.5 ? C.cap : [0.85, 0.2, 0.2], K.bloom, y, sm, seed);
      break;
    case "flowers":
      for (let k = 0; k < 3; k++) {
        const fx = x + (ihash(seed, k, 1) - 0.5) * 1.6, fz = z + (ihash(seed, k, 2) - 0.5) * 1.6, col = C.flowers[Math.floor(ihash(seed, k, 3) * 4)];
        mb.box(fx - 0.04, y - 0.3, fz - 0.04, fx + 0.04, y + 0.5, fz + 0.04, [0.3, 0.55, 0.25], K.bloom, y);
        mb.blob(fx, y + 0.58, fz, 0.17, 0.12, 0.17, col, K.bloom, y, sm, seed + k);
      }
      break;
    case "lanterns":
      mb.box(x - 0.08, y - 0.4, z - 0.08, x + 0.08, y + 2.2, z + 0.08, DARK, K.plain, y);
      mb.blob(x, y + 2.45, z, 0.32, 0.38, 0.32, LIGHT, K.glow, y, sm, seed);
      break;
    case "rocks":
      mb.blob(x, y + 0.1, z, 0.9 * sc, 0.6 * sc, 0.75 * sc, STONE, K.plain, y, sm, seed);
      break;
    case "cacti":
      mb.cone(x, y - 0.4, z, 0.32, 0.28, 2.8 * sc, 7, rot, CACTUS, K.plain, y, sm);
      if (r2 > 0.4) {
        mb.box(x + 0.2, y + 1.3, z - 0.15, x + 0.8, y + 1.6, z + 0.15, CACTUS, K.plain, y);
        mb.box(x + 0.55, y + 1.3, z - 0.15, x + 0.85, y + 2.4, z + 0.15, CACTUS, K.plain, y);
      }
      break;
  }
}

/** Natural style: bark-textured trunks, crowns of leaf cards around a dark inner volume, textured stones. */
function naturalFeature(mb: MB, kind: Feature, x: number, y: number, z: number, s: number, r2: number, C: Cols, seed: number) {
  const K = KIND, sc = 0.85 + 0.6 * s, rot = r2 * Math.PI * 2;
  const vary = (c: RGB, k: number): RGB => mul(c, 0.82 + 0.3 * ihash(seed, k, 9));
  const crown = (cx: number, cy: number, cz: number, r: number, col: RGB, n: number, k0: number) => {
    mb.blob(cx, cy, cz, r * 0.72, r * 0.6, r * 0.72, mul(col, 0.7), K.sway, y, true, seed + k0, true);
    for (let k = 0; k < n; k++) {
      const u = ihash(seed, k0 * 31 + k, 11) * 2 - 1, ph = ihash(seed, k0 * 31 + k, 12) * Math.PI * 2, q = Math.sqrt(1 - u * u);
      const dir: V3 = [q * Math.cos(ph), u * 0.8, q * Math.sin(ph)];
      const at: V3 = [cx + dir[0] * r * 0.62, cy + dir[1] * r * 0.62, cz + dir[2] * r * 0.62];
      const t = ihash(seed, k0 * 31 + k, 13) * Math.PI;
      const a: V3 = [Math.cos(t), 0.25 * Math.sin(t * 2), Math.sin(t)];
      const al = Math.hypot(...a);
      const A: V3 = [a[0] / al, a[1] / al, a[2] / al];
      const B0: V3 = [dir[1] * A[2] - dir[2] * A[1], dir[2] * A[0] - dir[0] * A[2], dir[0] * A[1] - dir[1] * A[0]];
      const bl = Math.hypot(...B0) || 1;
      mb.card(at, A, [B0[0] / bl, B0[1] / bl, B0[2] / bl], r * (0.55 + 0.25 * ihash(seed, k, 14)), [cx, cy, cz], vary(col, k), K.leaf, y);
    }
  };
  switch (kind) {
    case "trees": {
      const th = 3.4 * sc;
      mb.cone(x, y - 0.6, z, 0.34 * sc, 0.14 * sc, th + 0.6, 9, rot, vary(TRUNK, 0), K.bark, y, true);
      for (let k = 0; k < 4; k++) {
        const a = rot + k * 2.4, d = k === 0 ? 0 : (0.75 + 0.45 * ihash(seed, k, 1)) * sc, hy = th + (k === 0 ? 1.0 : 0.25 + 0.8 * ihash(seed, k, 2)) * sc;
        crown(x + Math.cos(a) * d, y + hy, z + Math.sin(a) * d, (k === 0 ? 1.9 : 1.3 + 0.4 * ihash(seed, k, 3)) * sc, vary(C.foliage, k + 1), k === 0 ? 16 : 10, k);
      }
      break;
    }
    case "pines": {
      mb.cone(x, y - 0.6, z, 0.26 * sc, 0.1 * sc, 2.4 * sc, 8, rot, vary(TRUNK, 0), K.bark, y, true);
      for (let k = 0; k < 6; k++) mb.cone(x, y + (0.9 + 0.85 * k) * sc, z, (2.0 - 0.3 * k) * sc, 0, 1.6 * sc, 12, rot + k * 0.7, vary(C.pine, k), K.sway, y, true);
      break;
    }
    case "palms": {
      // a leaning trunk in segments, then drooping fronds made of leaf cards
      let px = x, pz = z, py = y - 0.5;
      const lean = 0.18 + 0.12 * r2, seg = 5, len = 5.2 * sc / seg;
      for (let k = 0; k < seg; k++) {
        mb.cone(px, py, pz, (0.26 - 0.02 * k) * sc, (0.24 - 0.02 * k) * sc, len + 0.05, 8, rot + k, vary(PALMWOOD, k), K.bark, y, true);
        px += Math.cos(rot) * lean * len * (k / seg); pz += Math.sin(rot) * lean * len * (k / seg); py += len;
      }
      const top: V3 = [px, py, pz];
      for (let f = 0; f < 9; f++) {
        const a = rot + f * (2 * Math.PI / 9) + 0.2 * ihash(seed, f, 21);
        for (let k = 1; k <= 3; k++) {
          const d = k * 0.75 * sc, drop = 0.12 * k * k * sc;
          const at: V3 = [px + Math.cos(a) * d, py - drop + 0.2, pz + Math.sin(a) * d];
          const A: V3 = [Math.cos(a), -0.35 * k / 3, Math.sin(a)], al = Math.hypot(...A);
          mb.card(at, [A[0] / al, A[1] / al, A[2] / al], [-Math.sin(a), 0, Math.cos(a)], 0.62 * sc, top, vary(C.palmLeaf, f), K.leaf, y);
        }
      }
      break;
    }
    case "rocks":
      mb.blob(x, y - 0.1, z, 1.1 * sc, 0.7 * sc, 0.9 * sc, vary(STONE, 0), K.stone, y, true, seed, true);
      if (r2 > 0.45) mb.blob(x + 0.9 * sc, y - 0.2, z + 0.4 * sc, 0.6 * sc, 0.45 * sc, 0.55 * sc, vary(STONE, 1), K.stone, y, true, seed + 1, true);
      break;
    default:
      smoothFeature(mb, kind, x, y, z, s, r2, C, true, seed);
  }
}
