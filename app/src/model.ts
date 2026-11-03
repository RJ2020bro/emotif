// Runtime for models exported by emotif/models.py (format emotif-model-v2), plus in-app personalization.
import { N_BANDS, N_FEATURES, REGIONS, graphFeatures } from "./montage";
import { N_FACE_INPUT, N_REGION_FACE, faceInput } from "./face";

export type Layer = { W: number[][]; b: number[]; act: "relu" | "softmax" };
/** temperature: divides the final logits (calibrated in train.py); omitted = 1.
 * face: present on brain-only heads; the head then also reads the standardized face input (face.ts) and has the
 * removal of everything the face predicts folded into its first layer. */
export type Head = { classes: number[]; layers: Layer[]; temperature?: number; face?: { mu: number[]; sd: number[]; hold_z?: number } };
export type FaceInputSpec = { kind: "face-indices-v1"; eye_lowpass_hz: number; muscle_bands: number[][]; n: number };
export type Transform = { kind: "graph-diffusion"; hops: number; adjacency: number[][]; sigma_deg?: number };
export type ModelFile = {
  format: string;
  name: string;
  notes?: string;
  created?: string;
  feature_space: { kind: string; regions: string[]; n_features: number };
  transform?: Transform | null;
  face_input?: FaceInputSpec | null;
  model: string;
  datasets: string[];
  n_subjects?: number;
  n_windows?: number;
  heads: Partial<Record<"valence" | "arousal", Head>> & Record<string, Head | undefined>;   // extra heads, e.g. "stress"
  cv?: Record<string, CvResult>;
  personalized?: { from: string; windows: number; date: string };
};
export type Score = { acc: number; bacc: number; f1: number; n: number };
export type Calibration = { temperature: number; ece_before: number; ece_after: number; ece_after_heldout?: number };
export type CvResult = { within?: Record<string, Score>; loso?: Record<string, Score>; chance: number; chance_bacc?: number; calibration?: Calibration };

/** Length of the vector the heads read: region features, plus graph hops when the model has a transform. */
export function inputDim(m: Pick<ModelFile, "transform">): number {
  return N_FEATURES + (m.transform ? m.transform.hops * REGIONS.length * N_BANDS : 0);
}

/** Region features -> shared head input (applies the model's graph transform, if any). */
export function modelInput(m: Pick<ModelFile, "transform">, x: ArrayLike<number>): Float64Array {
  return m.transform ? graphFeatures(x, m.transform.adjacency, m.transform.hops) : Float64Array.from(x);
}

/** What one head reads: the shared input, plus the standardized face input on brain-only heads.
 * faceRegion: pooled face indices (face.pool); zeros (= the person's median) if not available. */
export function headInput(m: Pick<ModelFile, "transform">, head: Head, x: ArrayLike<number>, faceRegion: ArrayLike<number> | null): Float64Array {
  const base = modelInput(m, x);
  if (!head.face) return base;
  const q = faceInput(faceRegion ?? new Float64Array(N_REGION_FACE), head.face.mu, head.face.sd);
  const out = new Float64Array(base.length + q.length);
  out.set(base); out.set(q, base.length);
  return out;
}

export const isBrainOnly = (m: ModelFile | null): boolean => !!m && Object.values(m.heads).some(h => !!h?.face);

/** True when a brain-only head sees facial activity beyond what its correction was learned on (largest standardized
 * pooled face index above the head's hold_z, the 99.5th percentile in training). */
export function faceOutOfRange(head: Head, faceRegion: ArrayLike<number> | null): boolean {
  const f = head.face;
  if (!f || f.hold_z === undefined || !faceRegion) return false;
  for (let i = 0; i < N_REGION_FACE; i++) if (Math.abs((faceRegion[i] - f.mu[i]) / f.sd[i]) > f.hold_z) return true;
  return false;
}

export function validateModel(m: unknown): ModelFile {
  const f = m as ModelFile;
  if (!f || f.format !== "emotif-model-v2") throw new Error("This file isn't an Emotif v2 model. Retrain it with train.py.");
  if (!f.feature_space || f.feature_space.kind !== "region-v1") throw new Error("Unknown feature space in model.");
  if (f.feature_space.n_features !== N_FEATURES || f.feature_space.regions.join() !== REGIONS.join())
    throw new Error("Model was trained on a different region layout.");
  if (f.transform) {
    const t = f.transform, R = REGIONS.length;
    if (t.kind !== "graph-diffusion") throw new Error(`Unknown transform "${t.kind}".`);
    if (!Number.isInteger(t.hops) || t.hops < 1 || t.hops > 4) throw new Error("Graph transform: hops must be 1-4.");
    if (t.adjacency?.length !== R || t.adjacency.some(r => r.length !== R || r.some(v => !(v >= 0))))
      throw new Error("Graph transform: adjacency must be a non-negative 17 x 17 matrix.");
  }
  const heads = Object.entries(f.heads || {});
  if (!heads.length) throw new Error("Model has no output heads.");
  for (const [name, h] of heads) {
    if (!/^[a-z][a-z_]{0,23}$/.test(name)) throw new Error(`Bad head name "${name}".`);
    if (h!.temperature !== undefined && !(h!.temperature > 0)) throw new Error(`Head ${name}: temperature must be positive.`);
    if (h!.face) {
      if (!f.face_input || f.face_input.kind !== "face-indices-v1") throw new Error(`Head ${name} needs face input, but the model doesn't define it.`);
      const { mu, sd } = h!.face;
      if (mu?.length !== N_FACE_INPUT || sd?.length !== N_FACE_INPUT || sd.some(v => !(v > 0)) || mu.some(v => !Number.isFinite(v)))
        throw new Error(`Head ${name}: face statistics must be ${N_FACE_INPUT} finite means and positive deviations.`);
      if (h!.face.hold_z !== undefined && !(h!.face.hold_z > 0)) throw new Error(`Head ${name}: hold_z must be positive.`);
    }
    let dim = inputDim(f) + (h!.face ? N_FACE_INPUT : 0);
    h!.layers.forEach((L, i) => {
      if (L.W.length !== dim) throw new Error(`Head ${name}, layer ${i + 1}: expects ${L.W.length} inputs, got ${dim}.`);
      if (L.W.some(r => r.length !== L.b.length)) throw new Error(`Head ${name}, layer ${i + 1}: ragged weights.`);
      if (L.act !== "relu" && L.act !== "softmax") throw new Error(`Head ${name}: unknown activation ${L.act}.`);
      dim = L.b.length;
    });
    if (dim !== h!.classes.length) throw new Error(`Head ${name}: ${dim} outputs for ${h!.classes.length} classes.`);
    if (h!.layers[h!.layers.length - 1].act !== "softmax") throw new Error(`Head ${name}: last layer must be softmax.`);
  }
  return f;
}

export function forward(layers: Layer[], x: ArrayLike<number>, temperature = 1): Float64Array {
  let h = Float64Array.from(x);
  for (const L of layers) {
    const out = new Float64Array(L.b.length);
    for (let j = 0; j < out.length; j++) {
      let s = L.b[j];
      for (let i = 0; i < h.length; i++) s += h[i] * L.W[i][j];
      out[j] = s;
    }
    if (L.act === "relu") { for (let j = 0; j < out.length; j++) if (out[j] < 0) out[j] = 0; }
    else {
      for (let j = 0; j < out.length; j++) out[j] /= temperature;
      let mx = -Infinity;
      for (const v of out) mx = Math.max(mx, v);
      let sum = 0;
      for (let j = 0; j < out.length; j++) { out[j] = Math.exp(out[j] - mx); sum += out[j]; }
      for (let j = 0; j < out.length; j++) out[j] /= sum;
    }
    h = out;
  }
  return h;
}

/** Expected value of the class labels (each in -1..1) and the top (calibrated) probability. x: head input. */
export function headOutput(head: Head, x: ArrayLike<number>): { value: number; conf: number; probs: Float64Array } {
  const p = forward(head.layers, x, head.temperature ?? 1);
  let value = 0, conf = 0;
  head.classes.forEach((c, k) => { value += p[k] * Math.max(-1, Math.min(1, c)); conf = Math.max(conf, p[k]); });
  return { value, conf, probs: p };
}

/** Per-feature contribution for a single-layer head: w(high class) - w(low class) times x. */
export function contributions(head: Head, x: ArrayLike<number>): Float64Array | null {
  if (head.layers.length !== 1 || head.layers[0].W.length !== x.length) return null;
  const W = head.layers[0].W, hi = head.classes.length - 1;
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (W[i][hi] - W[i][0]) * x[i];
  return out;
}

export type LabeledWindow = { x: Float64Array; y: number; block: number; f?: Float64Array };

/** Multinomial logistic regression fit by gradient descent with an L2 pull toward prior weights
 * (MAP estimate with a Gaussian prior centred on the pretrained head). */
export function fitLogistic(data: LabeledWindow[], classes: number[], prior: Layer | null,
                            opts: { lambda?: number; iters?: number; lr?: number } = {}): Layer {
  const lambda = opts.lambda ?? 0.05, iters = opts.iters ?? 300, lr = opts.lr ?? 0.5;
  const D = data.length ? data[0].x.length : N_FEATURES, K = classes.length, n = data.length;
  const usePrior = !!prior && prior.W.length === D && prior.b.length === K;
  const W0 = usePrior ? prior!.W.map(r => r.slice()) : Array.from({ length: D }, () => new Array(K).fill(0));
  const b0 = usePrior ? prior!.b.slice() : new Array(K).fill(0);
  const W = W0.map(r => r.slice()), b = b0.slice();
  const yi = data.map(d => classes.indexOf(d.y));
  if (yi.some(i => i < 0)) throw new Error("label outside the model's classes");
  const p = new Float64Array(K);
  for (let it = 0; it < iters; it++) {
    const gW = Array.from({ length: D }, () => new Float64Array(K)), gb = new Float64Array(K);
    for (let s = 0; s < n; s++) {
      const x = data[s].x;
      let mx = -Infinity;
      for (let k = 0; k < K; k++) {
        let z = b[k];
        for (let i = 0; i < D; i++) z += x[i] * W[i][k];
        p[k] = z; mx = Math.max(mx, z);
      }
      let sum = 0;
      for (let k = 0; k < K; k++) { p[k] = Math.exp(p[k] - mx); sum += p[k]; }
      for (let k = 0; k < K; k++) {
        const g = p[k] / sum - (k === yi[s] ? 1 : 0);
        gb[k] += g;
        for (let i = 0; i < D; i++) gW[i][k] += g * x[i];
      }
    }
    for (let k = 0; k < K; k++) {
      b[k] -= lr * (gb[k] / n);
      for (let i = 0; i < D; i++) W[i][k] -= lr * (gW[i][k] / n + lambda * (W[i][k] - W0[i][k]));
    }
  }
  return { W, b, act: "softmax" };
}

/** Leave-one-block-out balanced accuracy for prior vs personalized heads. */
/** NaN when some class has fewer than two rated blocks: holding one out would leave that class
 * unseen in training, so the estimate would be meaningless. */
export function personalizationCv(data: LabeledWindow[], classes: number[], prior: Layer | null): { prior: number; personal: number; blocks: number } {
  const blocks = Array.from(new Set(data.map(d => d.block)));
  const blocksPerClass = new Map<number, Set<number>>();
  for (const d of data) { if (!blocksPerClass.has(d.y)) blocksPerClass.set(d.y, new Set()); blocksPerClass.get(d.y)!.add(d.block); }
  if (blocksPerClass.size < 2 || Array.from(blocksPerClass.values()).some(s => s.size < 2)) return { prior: NaN, personal: NaN, blocks: blocks.length };
  const pred = (L: Layer, x: Float64Array) => {
    const p = forward([L], x);
    let k = 0;
    for (let j = 1; j < p.length; j++) if (p[j] > p[k]) k = j;
    return classes[k];
  };
  const bacc = (pairs: [number, number][]) => {
    const per = classes.map(c => {
      const sel = pairs.filter(([t]) => t === c);
      return sel.length ? sel.filter(([t, p]) => t === p).length / sel.length : NaN;
    }).filter(v => !Number.isNaN(v));
    return per.length ? per.reduce((a, b) => a + b, 0) / per.length : NaN;
  };
  const pp: [number, number][] = [], pq: [number, number][] = [];
  for (const blk of blocks) {
    const train = data.filter(d => d.block !== blk), test = data.filter(d => d.block === blk);
    if (new Set(train.map(d => d.y)).size < 2) continue;
    const L = fitLogistic(train, classes, prior);
    for (const d of test) {
      pq.push([d.y, pred(L, d.x)]);
      if (prior) pp.push([d.y, pred(prior, d.x)]);
    }
  }
  return { prior: prior ? bacc(pp) : NaN, personal: bacc(pq), blocks: blocks.length };
}
