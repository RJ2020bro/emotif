// Personal state decoder. A short calibration session (protocol "states") records the person in six states; this
// module learns that person's own brain signature of each, with facial activity removed first, checks every state on
// blocks it never saw, and switches on only the states that pass a block-permutation test.
//
// Why personal: on public data the same features separate states *within* a person (stress, 13 of 17 people) but do
// not transfer *between* people for trait labels (depression, anxiety, sleepiness at chance). So each person is
// calibrated, and the app shows a state only after it has been demonstrated on that person's held-out blocks.
//
// Model: brain-only residual (features minus their ridge prediction from the face indices and their squares, fitted on
// the calibration windows) -> standardize -> shrinkage LDA. Everything is affine before the softmax, so it folds into
// one layer over [region features, standardized face input], the same input a brain-only head reads (model.headInput).
import { N_FEATURES, REGIONS } from "./montage";
import { N_REGION_FACE, N_FACE_INPUT } from "./face";
import { forward, type Head, type Layer } from "./model";

export const STATES = ["calm", "stress", "anxiety", "anger", "low_mood", "drowsy"] as const;
export type StateName = typeof STATES[number];
export const STATE_LABEL: Record<StateName, string> = {
  calm: "Calm", stress: "Stress", anxiety: "Anxiety", anger: "Anger", low_mood: "Low mood", drowsy: "Drowsiness",
};

export type StateWindow = { x: ArrayLike<number>; f: ArrayLike<number>; y: number; block: number };
export type StateCheck = { auc: number; p: number; on: boolean; blocks: number };
export type PersonalStates = {
  format: "emotif-states-v1";
  states: string[];                 // class order of the head's outputs
  head: Head;                       // classes 0..K-1, one softmax layer over [features, standardized face input]
  checks: Record<string, StateCheck>;
  faceOnly: { bacc: number; p: number };   // how well the face alone separates this person's states (audit)
  bacc: number; baccP: number;      // multi-state balanced accuracy on held-out blocks and its permutation p
  windows: number; blocks: number; created: string;
};

const SHRINK = 0.5, RIDGE = 1.0, ALPHA = 0.05;

// ---------------------------------------------------------------- small dense linear algebra
/** Solve A X = B for symmetric positive definite A (n x n), B (n x m), by Cholesky. Returns X (n x m). */
export function cholSolve(A: number[][], B: number[][]): number[][] {
  const n = A.length, L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (!(s > 1e-12)) throw new Error("matrix is not positive definite");
        L[i][i] = Math.sqrt(s);
      } else L[i][j] = s / L[j][j];
    }
  }
  const m = B[0]?.length ?? 0, X = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let c = 0; c < m; c++) {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = B[i][c]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * X[k][c]; X[i][c] = s / L[i][i]; }
  }
  return X;
}

const meanSd = (rows: ArrayLike<number>[], d: number) => {
  const mu = new Float64Array(d), sd = new Float64Array(d), n = rows.length;
  for (const r of rows) for (let j = 0; j < d; j++) mu[j] += r[j] / n;
  for (const r of rows) for (let j = 0; j < d; j++) sd[j] += (r[j] - mu[j]) ** 2 / n;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]);
  return { mu, sd };
};

/** Face design: pooled indices and their squares (102 values), as in face.faceInput. */
function faceDesign(f: ArrayLike<number>): Float64Array {
  const out = new Float64Array(N_FACE_INPUT);
  for (let i = 0; i < N_REGION_FACE; i++) { out[i] = f[i]; out[N_REGION_FACE + i] = f[i] * f[i]; }
  return out;
}

// ---------------------------------------------------------------- fit
/** Cholesky factor of an SPD matrix (lower triangle). */
function chol(A: number[][]): Float64Array[] {
  const n = A.length, L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j];
    for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    if (i === j) { if (!(s > 1e-12)) throw new Error("matrix is not positive definite"); L[i][i] = Math.sqrt(s); }
    else L[i][j] = s / L[j][j];
  }
  return L;
}
function cholApply(L: Float64Array[], v: Float64Array): Float64Array {
  const n = L.length, y = new Float64Array(n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = v[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
  return x;
}

/** Label-free part of the model, learned on training windows: face standardization, the ridge map from face to
 * features, the residual standardization, the kept columns, and the Cholesky factor of the shrunk total covariance.
 * It doesn't depend on the state labels, so permutation tests reuse it and only refit the class means. */
type Prep = {
  qmu: Float64Array; qsd: Float64Array; B: number[][]; rmu: Float64Array; rsd: Float64Array; idx: number[];
  L: Float64Array[]; S: Float64Array[]; hold_z: number;
};
type Rows = { x: ArrayLike<number>; f: ArrayLike<number> }[];

function zFace(f: ArrayLike<number>, qmu: Float64Array, qsd: Float64Array): Float64Array {
  const q = faceDesign(f), z = new Float64Array(N_FACE_INPUT);
  for (let i = 0; i < N_FACE_INPUT; i++) z[i] = (q[i] - qmu[i]) / qsd[i];
  return z;
}
function residual(x: ArrayLike<number>, z: Float64Array, B: number[][]): Float64Array {
  const P = N_FACE_INPUT + 1, r = new Float64Array(N_FEATURES);
  for (let j = 0; j < N_FEATURES; j++) {
    let v = x[j] - B[P - 1][j];
    for (let a = 0; a < N_FACE_INPUT; a++) if (z[a] !== 0) v -= z[a] * B[a][j];
    r[j] = v;
  }
  return r;
}
function standardizeRow(p: Prep, w: { x: ArrayLike<number>; f: ArrayLike<number> }): Float64Array {
  const r = residual(w.x, zFace(w.f, p.qmu, p.qsd), p.B);
  return Float64Array.from(p.idx, j => (r[j] - p.rmu[j]) / p.rsd[j]);
}

function prepare(rows: Rows, removeFace: boolean): Prep {
  const n = rows.length, P = N_FACE_INPUT + 1;
  const Q = rows.map(d => faceDesign(d.f));
  const qs = meanSd(Q, N_FACE_INPUT);
  const qsd = Float64Array.from(qs.sd, v => (v < 1e-9 ? 1 : v));
  const Z = Q.map(q => { const z = new Float64Array(N_FACE_INPUT); for (let i = 0; i < N_FACE_INPUT; i++) z[i] = (q[i] - qs.mu[i]) / qsd[i]; return z; });
  let B: number[][];
  if (removeFace) {
    const ZtZ = Array.from({ length: P }, () => new Array(P).fill(0)), ZtX = Array.from({ length: P }, () => new Array(N_FEATURES).fill(0));
    for (let s = 0; s < n; s++) {
      const z = Z[s], x = rows[s].x;
      for (let a = 0; a < P; a++) {
        const za = a < N_FACE_INPUT ? z[a] : 1;
        if (za === 0) continue;
        for (let c = 0; c < P; c++) ZtZ[a][c] += za * (c < N_FACE_INPUT ? z[c] : 1);
        for (let j = 0; j < N_FEATURES; j++) ZtX[a][j] += za * x[j];
      }
    }
    for (let a = 0; a < N_FACE_INPUT; a++) ZtZ[a][a] += RIDGE;
    ZtZ[P - 1][P - 1] += 1e-9;
    B = cholSolve(ZtZ, ZtX);
  } else {                               // reference model: only centre the features
    B = Array.from({ length: P }, () => new Array(N_FEATURES).fill(0));
    const mu = meanSd(rows.map(r => r.x), N_FEATURES).mu;
    for (let j = 0; j < N_FEATURES; j++) B[P - 1][j] = mu[j];
  }
  const R = rows.map((d, s) => residual(d.x, Z[s], B));
  const rs = meanSd(R, N_FEATURES);
  const idx: number[] = [];
  for (let j = 0; j < N_FEATURES; j++) if (rs.sd[j] > 1e-9) idx.push(j);
  if (!idx.length) throw new Error("no usable features");
  const S = R.map(r => Float64Array.from(idx, j => (r[j] - rs.mu[j]) / rs.sd[j]));
  const D = idx.length, C = Array.from({ length: D }, () => new Array(D).fill(0));
  for (const v of S) for (let a = 0; a < D; a++) { const va = v[a]; for (let c = 0; c <= a; c++) C[a][c] += va * v[c]; }
  for (let a = 0; a < D; a++) for (let c = 0; c <= a; c++) { C[a][c] = (1 - SHRINK) * C[a][c] / n + (a === c ? SHRINK : 0); C[c][a] = C[a][c]; }
  const mx = Z.map(z => { let m = 0; for (let i = 0; i < N_REGION_FACE; i++) m = Math.max(m, Math.abs(z[i])); return m; }).sort((a, c) => a - c);
  const hold_z = Math.max(3, mx[Math.min(mx.length - 1, Math.floor(0.995 * mx.length))]);
  return { qmu: qs.mu, qsd, B, rmu: rs.mu, rsd: Float64Array.from(rs.sd), idx, L: chol(C), S, hold_z };
}

/** Shrinkage LDA with the (label-free) total covariance: W_k = Sigma^-1 m_k, b_k = -m_k' Sigma^-1 m_k / 2. */
function lda(p: Prep, y: number[], K: number): { W: Float64Array[]; b: number[] } {
  const D = p.idx.length, means = Array.from({ length: K }, () => new Float64Array(D)), cnt = new Array(K).fill(0);
  p.S.forEach((v, s) => { cnt[y[s]]++; const m = means[y[s]]; for (let j = 0; j < D; j++) m[j] += v[j]; });
  if (cnt.some(c => c === 0)) throw new Error("a state has no windows");
  const W = means.map((m, k) => { for (let j = 0; j < D; j++) m[j] /= cnt[k]; return cholApply(p.L, m); });
  const b = means.map((m, k) => { let s = 0; for (let j = 0; j < D; j++) s += m[j] * W[k][j]; return -0.5 * s; });
  return { W, b };
}
function softmaxRow(v: Float64Array, W: Float64Array[], b: number[]): Float64Array {
  const K = W.length, z = new Float64Array(K);
  let mx = -Infinity;
  for (let k = 0; k < K; k++) { let s = b[k]; for (let j = 0; j < v.length; j++) s += v[j] * W[k][j]; z[k] = s; mx = Math.max(mx, s); }
  let sum = 0;
  for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - mx); sum += z[k]; }
  for (let k = 0; k < K; k++) z[k] /= sum;
  return z;
}

/** Fit on all windows and fold everything into one softmax layer over [features, standardized face input]. */
export function fitLayer(data: StateWindow[], K: number): { layer: Layer; face: { mu: number[]; sd: number[]; hold_z: number } } {
  if (data.length < 2 * K) throw new Error("too few windows");
  const p = prepare(data, true), { W, b: b0 } = lda(p, data.map(d => d.y), K);
  const P = N_FACE_INPUT + 1;
  const Wx = Array.from({ length: N_FEATURES }, () => new Array(K).fill(0));
  const Wq = Array.from({ length: N_FACE_INPUT }, () => new Array(K).fill(0));
  const b = b0.slice();
  p.idx.forEach((j, t) => {
    for (let k = 0; k < K; k++) {
      const w = W[k][t] / p.rsd[j];
      Wx[j][k] = w;
      b[k] -= (p.B[P - 1][j] + p.rmu[j]) * w;
      for (let a = 0; a < N_FACE_INPUT; a++) Wq[a][k] -= p.B[a][j] * w;
    }
  });
  return { layer: { W: [...Wx, ...Wq], b, act: "softmax" }, face: { mu: Array.from(p.qmu), sd: Array.from(p.qsd), hold_z: p.hold_z } };
}

/** The input a personal head reads for one window (same layout as model.headInput for a brain-only head). */
export function stateInput(x: ArrayLike<number>, f: ArrayLike<number>, face: { mu: ArrayLike<number>; sd: ArrayLike<number> }): Float64Array {
  const out = new Float64Array(N_FEATURES + N_FACE_INPUT);
  for (let j = 0; j < N_FEATURES; j++) out[j] = x[j];
  const q = faceDesign(f);
  for (let i = 0; i < N_FACE_INPUT; i++) out[N_FEATURES + i] = (q[i] - face.mu[i]) / face.sd[i];
  return out;
}

// ---------------------------------------------------------------- evaluation on held-out blocks
export function aucScore(pos: number[], neg: number[]): number {
  if (!pos.length || !neg.length) return NaN;
  let s = 0;
  for (const p of pos) for (const q of neg) s += p > q ? 1 : p === q ? 0.5 : 0;
  return s / (pos.length * neg.length);
}

type Fold = { block: number; prep: Prep; test: Float64Array[]; train: number[] };
/** Leave-one-block-out folds. Each held-out block is scored by a model fitted without it; within a fold the
 * label-free preparation is fixed, so a permutation only refits class means. Brain: the brain-only features;
 * face: the face indices alone (reference). */
function folds(data: StateWindow[], kind: "brain" | "face"): Fold[] {
  const blocks = Array.from(new Set(data.map(d => d.block)));
  const rows: Rows = kind === "brain" ? data : data.map(d => {
    const x = new Float64Array(N_FEATURES); x.set(faceDesign(d.f).subarray(0, Math.min(N_FACE_INPUT, N_FEATURES)));
    return { x, f: new Float64Array(N_REGION_FACE) };
  });
  return blocks.map(blk => {
    const train: number[] = [], test: number[] = [];
    data.forEach((d, i) => (d.block === blk ? test : train).push(i));
    const prep = prepare(train.map(i => rows[i]), kind === "brain");
    return { block: blk, prep, train, test: test.map(i => standardizeRow(prep, rows[i])) };
  });
}

function evaluate(fs: Fold[], blockLabel: Map<number, number>, data: StateWindow[], K: number) {
  const rows: { y: number; p: Float64Array }[] = [];
  for (const f of fs) {
    const y = f.train.map(i => blockLabel.get(data[i].block)!);
    if (new Set(y).size < K) continue;
    const { W, b } = lda(f.prep, y, K);
    const acc = new Float64Array(K);
    for (const v of f.test) { const p = softmaxRow(v, W, b); for (let k = 0; k < K; k++) acc[k] += p[k] / f.test.length; }
    rows.push({ y: blockLabel.get(f.block)!, p: acc });
  }
  const auc = Array.from({ length: K }, (_, k) => aucScore(rows.filter(r => r.y === k).map(r => r.p[k]), rows.filter(r => r.y !== k).map(r => r.p[k])));
  const rec = Array.from({ length: K }, (_, k) => {
    const sel = rows.filter(r => r.y === k);
    return sel.length ? sel.filter(r => { let m = 0; for (let j = 1; j < K; j++) if (r.p[j] > r.p[m]) m = j; return m === k; }).length / sel.length : NaN;
  });
  const ok = rec.filter(v => !Number.isNaN(v));
  return { auc, bacc: ok.length ? ok.reduce((a, c) => a + c, 0) / ok.length : NaN };
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** Calibrate: fit on everything, check each state on held-out blocks against permutations of the state labels
 * between blocks (the unit of the design), and switch on the states whose held-out AUC beats them (p <= 0.05).
 * Needs at least two blocks of every state. */
export function calibrateStates(data: StateWindow[], states: readonly string[], opts: { perms?: number; seed?: number } = {}): PersonalStates {
  const K = states.length, perms = opts.perms ?? 199, rnd = mulberry(opts.seed ?? 1);
  const per = states.map((_, k) => new Set(data.filter(d => d.y === k).map(d => d.block)).size);
  if (per.some(c => c < 2)) throw new Error(`Each state needs at least two calibration blocks (got ${states.map((s, k) => `${s} ${per[k]}`).join(", ")}).`);
  const blocks = Array.from(new Set(data.map(d => d.block)));
  const truth = new Map(blocks.map(b => [b, data.find(d => d.block === b)!.y]));
  const fb = folds(data, "brain"), ff = folds(data, "face");
  const real = evaluate(fb, truth, data, K), faceReal = evaluate(ff, truth, data, K);
  const geAuc = new Array(K).fill(0);
  let geB = 0, geF = 0;
  const lab = blocks.map(b => truth.get(b)!);
  for (let i = 0; i < perms; i++) {
    for (let a = lab.length - 1; a > 0; a--) { const j = Math.floor(rnd() * (a + 1)); [lab[a], lab[j]] = [lab[j], lab[a]]; }
    const m = new Map(blocks.map((b, t) => [b, lab[t]]));
    const s = evaluate(fb, m, data, K);
    s.auc.forEach((a, k) => { if (!(a < real.auc[k])) geAuc[k]++; });
    if (!(s.bacc < real.bacc)) geB++;
    if (!(evaluate(ff, m, data, K).bacc < faceReal.bacc)) geF++;
  }
  const { layer, face } = fitLayer(data, K);
  const checks: Record<string, StateCheck> = {};
  states.forEach((s, k) => {
    const p = (1 + geAuc[k]) / (1 + perms);
    checks[s] = { auc: real.auc[k], p, on: real.auc[k] > 0.5 && p <= ALPHA, blocks: per[k] };
  });
  return {
    format: "emotif-states-v1", states: states.slice(),
    head: { classes: states.map((_, k) => k), layers: [layer], face },
    checks, faceOnly: { bacc: faceReal.bacc, p: (1 + geF) / (1 + perms) },
    bacc: real.bacc, baccP: (1 + geB) / (1 + perms),
    windows: data.length, blocks: blocks.length, created: new Date().toISOString(),
  };
}

/** Validate a stored personal state model before use (it may come from a file or another version). */
export function validateStates(m: unknown): PersonalStates {
  const s = m as PersonalStates;
  if (!s || s.format !== "emotif-states-v1" || !Array.isArray(s.states) || !s.states.length) throw new Error("Not an Emotif personal state model.");
  const L = s.head?.layers?.[0];
  if (!L || L.W.length !== N_FEATURES + N_FACE_INPUT || L.b.length !== s.states.length || L.W.some(r => r.length !== s.states.length))
    throw new Error("Personal state model has the wrong shape.");
  if (!s.head.face || s.head.face.mu.length !== N_FACE_INPUT || s.head.face.sd.some(v => !(v > 0))) throw new Error("Personal state model has no face statistics.");
  if (!s.checks || s.states.some(n => !s.checks[n])) throw new Error("Personal state model has no checks.");
  return s;
}

/** Probabilities of each state for one window. */
export function stateProbs(m: PersonalStates, x: ArrayLike<number>, f: ArrayLike<number>): Float64Array {
  return forward(m.head.layers, stateInput(x, f, m.head.face!), m.head.temperature ?? 1);
}

export const STATE_REGIONS = REGIONS;   // re-exported for reports
