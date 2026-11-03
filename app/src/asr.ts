// Artifact Subspace Reconstruction. Mirror of emotif/asr.py (see that file for the method).
export const WIN_SEC = 0.5, STEP_SEC = 0.125, CAL_STEP_SEC = 0.25, MAX_DROP = 0.66, DEFAULT_K = 20, MIN_CAL_SEC = 10, RIDGE = 1e-3, MIN_RATIO = 2.5;

type Mat = Float64Array[];   // row-major list of rows
export type AsrState = { C: Mat; T: Mat; k: number; fs: number; n: number };

const pyRound = (x: number): number => {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
};
export function sizes(fs: number): { W: number; S: number; L: number } {
  const W = Math.max(4, pyRound(WIN_SEC * fs)), S = Math.max(1, pyRound(STEP_SEC * fs));
  return { W, S, L: Math.floor((W - S) / 2) };
}
export const minKeep = (C: number): number => Math.max(1, C - Math.floor(MAX_DROP * C));

const zeros = (n: number, m: number): Mat => Array.from({ length: n }, () => new Float64Array(m));
function eye(n: number): Mat { const I = zeros(n, n); for (let i = 0; i < n; i++) I[i][i] = 1; return I; }

/** Symmetric eigendecomposition by cyclic Jacobi. Returns ascending eigenvalues and eigenvectors as columns of V. */
export function eigSym(A: Mat): { d: Float64Array; V: Mat } {
  const n = A.length;
  const a = A.map(r => Float64Array.from(r));
  const V = eye(n);
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, tot = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = a[i][j] * a[i][j]; tot += v; if (i !== j) off += v; }
    if (off <= 1e-30 * Math.max(tot, 1e-300)) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  const d = Float64Array.from(idx.map(i => a[i][i]));
  const Vs = zeros(n, n);
  for (let r = 0; r < n; r++) idx.forEach((src, c) => { Vs[r][c] = V[r][src]; });
  return { d, V: Vs };
}

/** Solve B X = Y for SPD/general square B (Gaussian elimination with partial pivoting). */
function solve(B: Mat, Y: Mat): Mat {
  const n = B.length, m = Y[0].length;
  const A = B.map(r => Float64Array.from(r)), X = Y.map(r => Float64Array.from(r));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]]; [X[c], X[piv]] = [X[piv], X[c]];
    const d = A[c][c] || 1e-300;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c] / d;
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      for (let k = 0; k < m; k++) X[r][k] -= f * X[c][k];
    }
  }
  for (let r = 0; r < n; r++) { const d = A[r][r] || 1e-300; for (let k = 0; k < m; k++) X[r][k] /= d; }
  return X;
}

function median(a: number[]): number {
  const s = a.slice().sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Weiszfeld geometric median (Frobenius). Mirror of asr.geometric_median. */
export function geometricMedian(mats: Mat[], iters = 100, tol = 1e-12): Mat {
  const n = mats.length, C = mats[0].length;
  let y = zeros(C, C);
  for (const A of mats) for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) y[i][j] += A[i][j] / n;
  const fro = (M: Mat) => { let s = 0; for (const r of M) for (const v of r) s += v * v; return Math.sqrt(s); };
  for (let it = 0; it < iters; it++) {
    const scale = Math.max(1, fro(y));
    const w = mats.map(A => { let s = 0; for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) { const d = A[i][j] - y[i][j]; s += d * d; } return 1 / Math.max(Math.sqrt(s), 1e-12 * scale); });
    const ws = w.reduce((a, b) => a + b, 0);
    const yn = zeros(C, C);
    mats.forEach((A, k) => { for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) yn[i][j] += w[k] * A[i][j]; });
    for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) yn[i][j] /= ws;
    let ch = 0;
    for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) { const d = yn[i][j] - y[i][j]; ch += d * d; }
    const done = Math.sqrt(ch) <= tol * Math.max(1e-300, fro(y));
    y = yn;
    if (done) break;
  }
  return y;
}

/** X: C channel arrays of filtered calibration data. */
export function calibrate(X: ArrayLike<number>[], fs: number, k = DEFAULT_K): AsrState {
  const C = X.length, N = X[0].length, { W } = sizes(fs);
  if (N < MIN_CAL_SEC * fs) throw new Error(`ASR calibration needs at least ${MIN_CAL_SEC} s of data`);
  const nwin = Math.floor(N / W);
  const covs: Mat[] = [];
  for (let w = 0; w < nwin; w++) {
    const cv = zeros(C, C);
    for (let i = 0; i < C; i++) for (let j = i; j < C; j++) {
      let s = 0;
      for (let t = w * W; t < (w + 1) * W; t++) s += X[i][t] * X[j][t];
      cv[i][j] = cv[j][i] = s / W;
    }
    covs.push(cv);
  }
  const Ccal = geometricMedian(covs);
  for (let i = 0; i < C; i++) for (let j = i + 1; j < C; j++) { const v = (Ccal[i][j] + Ccal[j][i]) / 2; Ccal[i][j] = Ccal[j][i] = v; }
  let tr = 0;
  for (let i = 0; i < C; i++) tr += Ccal[i][i];
  for (let i = 0; i < C; i++) Ccal[i][i] += RIDGE * tr / C;
  const { V } = eigSym(Ccal);
  const step = Math.max(1, pyRound(CAL_STEP_SEC * fs));
  // component signals Y = V^T X
  const Y = zeros(C, N);
  for (let c = 0; c < C; c++) for (let t = 0; t < N; t++) { let s = 0; for (let i = 0; i < C; i++) s += V[i][c] * X[i][t]; Y[c][t] = s; }
  const thr = new Float64Array(C);
  for (let c = 0; c < C; c++) {
    const rms: number[] = [];
    for (let s0 = 0; s0 <= N - W; s0 += step) { let s = 0; for (let t = s0; t < s0 + W; t++) s += Y[c][t] * Y[c][t]; rms.push(Math.sqrt(s / W)); }
    const mu = median(rms);
    let sig = 1.4826 * median(rms.map(r => Math.abs(r - mu)));
    sig = Math.max(sig, 1e-6 * Math.max(mu, 1e-12));
    thr[c] = Math.max(mu + k * sig, MIN_RATIO * mu);
  }
  const T = zeros(C, C);
  for (let r = 0; r < C; r++) for (let c = 0; c < C; c++) T[r][c] = thr[r] * V[c][r];
  return { C: Ccal, T, k, fs, n: C };
}

/** Reconstruction matrix for one block covariance; also the number of removed components. */
export function reconstruction(st: AsrState, cov: Mat): { R: Mat; removed: number } {
  const C = st.n;
  const sym = cov.map((r, i) => r.map((v, j) => (v + cov[j][i]) / 2));
  const { d, V: U } = eigSym(sym);
  const keep: boolean[] = [];
  for (let j = 0; j < C; j++) {
    let s = 0;
    for (let r = 0; r < C; r++) { let tu = 0; for (let c = 0; c < C; c++) tu += st.T[r][c] * U[c][j]; s += tu * tu; }
    keep.push(d[j] < s || j < minKeep(C));
  }
  const kidx = keep.map((k, j) => (k ? j : -1)).filter(j => j >= 0);
  if (kidx.length === C) return { R: eye(C), removed: 0 };
  const K = kidx.length;
  const Uk = zeros(C, K);
  for (let r = 0; r < C; r++) kidx.forEach((j, c) => { Uk[r][c] = U[r][j]; });
  const CU = zeros(C, K);   // Ccal Uk
  for (let r = 0; r < C; r++) for (let c = 0; c < K; c++) { let s = 0; for (let i = 0; i < C; i++) s += st.C[r][i] * Uk[i][c]; CU[r][c] = s; }
  const B = zeros(K, K);    // Uk^T Ccal Uk
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) { let s = 0; for (let i = 0; i < C; i++) s += Uk[i][a] * CU[i][b]; B[a][b] = s; }
  const UkT = zeros(K, C);
  for (let a = 0; a < K; a++) for (let c = 0; c < C; c++) UkT[a][c] = Uk[c][a];
  const Z = solve(B, UkT);  // B^-1 Uk^T  (K x C)
  const R = zeros(C, C);
  for (let r = 0; r < C; r++) for (let c = 0; c < C; c++) { let s = 0; for (let a = 0; a < K; a++) s += CU[r][a] * Z[a][c]; R[r][c] = s; }
  // repair must not add energy: tr(R cov R^T) > tr(cov) -> pass the block through
  let eOut = 0, eIn = 0;
  for (let r = 0; r < C; r++) {
    eIn += sym[r][r];
    for (let a = 0; a < C; a++) { let s = 0; for (let b = 0; b < C; b++) s += sym[a][b] * R[r][b]; eOut += R[r][a] * s; }
  }
  if (eOut > eIn) return { R: eye(C), removed: 0 };
  return { R, removed: C - K };
}

/** Streaming cleaner. push() returns cleaned samples in order, delayed by S + L samples. */
export class AsrStream {
  readonly W: number; readonly S: number; readonly L: number; readonly C: number;
  removed = 0;
  private buf: Float64Array[];
  private len = 0; private offset = 0; private next = 0;
  private Rprev: Mat;
  private blend: Float64Array;
  constructor(private st: AsrState, fs: number) {
    ({ W: this.W, S: this.S, L: this.L } = sizes(fs));
    this.C = st.n;
    this.buf = Array.from({ length: this.C }, () => new Float64Array(Math.max(4 * this.W, 1024)));
    this.Rprev = eye(this.C);
    this.blend = new Float64Array(this.S);
    for (let j = 0; j < this.S; j++) this.blend[j] = (1 - Math.cos(Math.PI * (j + 1) / this.S)) / 2;
  }
  get delay(): number { return this.S + this.L; }

  private append(row: ArrayLike<number>) {
    if (this.len === this.buf[0].length) this.buf = this.buf.map(b => { const n = new Float64Array(b.length * 2); n.set(b); return n; });
    for (let c = 0; c < this.C; c++) this.buf[c][this.len] = row[c];
    this.len++;
  }

  push(row: ArrayLike<number>): Float64Array[] {
    this.append(row);
    const out: Float64Array[] = [];
    const total = this.offset + this.len;
    while (this.next + this.S + this.L <= total) {
      const b0 = this.next, we = b0 + this.S + this.L, ws = Math.max(0, we - this.W), n = we - ws;
      const cov = zeros(this.C, this.C);
      for (let i = 0; i < this.C; i++) for (let j = i; j < this.C; j++) {
        let s = 0;
        const bi = this.buf[i], bj = this.buf[j];
        for (let t = ws - this.offset; t < we - this.offset; t++) s += bi[t] * bj[t];
        cov[i][j] = cov[j][i] = s / n;
      }
      const { R, removed } = reconstruction(this.st, cov);
      this.removed = removed;
      for (let j = 0; j < this.S; j++) {
        const t = b0 - this.offset + j, w = this.blend[j], o = new Float64Array(this.C);
        for (let r = 0; r < this.C; r++) {
          let a = 0, b = 0;
          for (let c = 0; c < this.C; c++) { const x = this.buf[c][t]; a += R[r][c] * x; b += this.Rprev[r][c] * x; }
          o[r] = w * a + (1 - w) * b;
        }
        out.push(o);
      }
      this.Rprev = R;
      this.next += this.S;
    }
    const keepFrom = Math.max(0, this.next + this.S + this.L - this.W) - this.offset;
    if (keepFrom > 0) {
      for (let c = 0; c < this.C; c++) this.buf[c].copyWithin(0, keepFrom, this.len);
      this.len -= keepFrom;
      this.offset += keepFrom;
    }
    return out;
  }
}
