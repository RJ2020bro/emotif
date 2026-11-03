// Learned artifact detector. Mirror of emotif/artifact.py: features, baseline, classification.
import { detrend } from "./dsp";
import { forward, type Layer } from "./model";

export const CLASSES = ["clean", "ocular", "muscle"] as const;
export const N_ART_FEATURES = 11;
const EPS = 1e-12;

export type ArtifactModel = { format: string; classes: string[]; layers: Layer[]; mask_threshold: number; metrics?: Record<string, unknown> };

export function validateArtifactModel(m: unknown): ArtifactModel {
  const a = m as ArtifactModel;
  if (!a || a.format !== "emotif-artifact-v2") throw new Error("Not an Emotif v2 artifact model.");
  if (a.layers[0].W.length !== N_ART_FEATURES || a.layers[a.layers.length - 1].b.length !== 3) throw new Error("Artifact model has the wrong shape.");
  return a;
}

/** One channel's 11 features from a filtered window and its 5 band powers. */
export function channelFeatures(x: ArrayLike<number>, fs: number, bp: ArrayLike<number>): Float64Array {
  const f = new Float64Array(N_ART_FEATURES);
  let tot = 0;
  for (let b = 0; b < 5; b++) tot += bp[b];
  for (let b = 0; b < 5; b++) f[b] = Math.log(bp[b] / (tot + EPS) + EPS);
  const d = detrend(x), N = d.length;
  let m2 = 0, m3 = 0, m4 = 0, mn = Infinity, mx = -Infinity;
  for (let i = 0; i < N; i++) { const v = d[i], v2 = v * v; m2 += v2; m3 += v2 * v; m4 += v2 * v2; if (v < mn) mn = v; if (v > mx) mx = v; }
  m2 = m2 / N + EPS; m3 /= N; m4 /= N;
  const kurt = m4 / (m2 * m2) - 3, skew = m3 / Math.pow(m2, 1.5), std = Math.sqrt(m2);
  let sx = 0, sxx = 0;
  for (let i = 0; i < N - 1; i++) { const a = d[i + 1] - d[i]; sx += a * a; }
  for (let i = 0; i < N - 2; i++) { const a = d[i + 2] - 2 * d[i + 1] + d[i]; sxx += a * a; }
  sx = Math.sqrt(sx / (N - 1) + EPS); sxx = Math.sqrt(sxx / (N - 2) + EPS);
  const mob = sx / std, comp = (sxx / sx) / mob;
  f[5] = Math.sign(kurt) * Math.log1p(Math.abs(kurt));
  f[6] = skew;
  f[7] = Math.log((mx - mn) / std + EPS);
  f[8] = Math.log(mob * fs / (2 * Math.PI) + EPS);
  f[9] = Math.log(comp + EPS);
  f[10] = Math.log(std + EPS);
  return f;
}

/** Per-channel median of features over baseline windows: feats[w][c] -> base[c]. */
export function baselineOf(feats: Float64Array[][]): Float64Array[] {
  const C = feats[0].length;
  return Array.from({ length: C }, (_, c) => {
    const out = new Float64Array(N_ART_FEATURES);
    for (let k = 0; k < N_ART_FEATURES; k++) {
      const v = feats.map(w => w[c][k]).sort((a, b) => a - b), m = v.length >> 1;
      out[k] = v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
    }
    return out;
  });
}

export type ChannelVerdict = { probs: Float64Array; bad: boolean; label: "clean" | "ocular" | "muscle" };
export function classify(model: ArtifactModel, feats: Float64Array, base: Float64Array): ChannelVerdict {
  const rel = feats.map((v, k) => v - base[k]);
  const probs = forward(model.layers, rel);
  const bad = 1 - probs[0] >= model.mask_threshold;
  const label = !bad ? "clean" : probs[1] >= probs[2] ? "ocular" : "muscle";
  return { probs, bad, label };
}
