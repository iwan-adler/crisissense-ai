// LooMap ML engine — gradient-boosted regression trees that train in the browser.
// Supports: squared-error and quantile (pinball) objectives, monotone constraints,
// per-prediction feature contributions (tree-path attribution), spatial CV,
// ridge baseline and Getis-Ord Gi* hotspot statistics.

const NBINS = 32;

// ---------- binning ----------
function makeBins(col) {
  const v = Array.from(col).filter(Number.isFinite).sort((a, b) => a - b);
  const edges = [];
  if (!v.length) return edges;
  for (let i = 1; i < NBINS; i++) {
    const e = v[Math.floor((i * v.length) / NBINS)];
    if (!edges.length || e > edges[edges.length - 1]) edges.push(e);
  }
  return edges; // value x goes to bin = #edges <= x  (upper-bound search)
}
function binOf(edges, x) {
  if (!Number.isFinite(x)) return 0;
  let lo = 0, hi = edges.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (edges[m] <= x) lo = m + 1; else hi = m; }
  return lo;
}

function quantile(arr, q) {
  if (!arr.length) return 0;
  const s = Float64Array.from(arr).sort();
  const p = (s.length - 1) * q, i = Math.floor(p), f = p - i;
  return i + 1 < s.length ? s[i] * (1 - f) + s[i + 1] * f : s[i];
}

// ---------- GBM ----------
export class GBM {
  constructor(opts = {}) {
    this.o = Object.assign({
      nTrees: 160, lr: 0.08, maxDepth: 4, minLeaf: 15, lambda: 1.0,
      objective: 'l2', alpha: 0.5, monotone: null, subsample: 0.8, seed: 7,
    }, opts);
  }

  fit(X, y, featureNames) {
    const o = this.o, n = y.length, F = X[0].length;
    this.names = featureNames;
    this.mono = o.monotone || new Array(F).fill(0);
    const cols = [];
    for (let f = 0; f < F; f++) cols.push(X.map(r => r[f]));
    this.edges = cols.map(makeBins);
    const XB = cols.map((c, f) => Uint8Array.from(c, x => binOf(this.edges[f], x)));
    this.base = o.objective === 'quantile' ? quantile(y, o.alpha) : y.reduce((a, b) => a + b, 0) / n;
    const pred = new Float64Array(n).fill(this.base);
    this.trees = [];
    let seed = o.seed;
    const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    for (let t = 0; t < o.nTrees; t++) {
      const idx = [];
      for (let i = 0; i < n; i++) if (rand() < o.subsample) idx.push(i);
      const g = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (o.objective === 'quantile') g[i] = y[i] < pred[i] ? 1 - o.alpha : -o.alpha;
        else g[i] = pred[i] - y[i];
      }
      const tree = this._build(XB, g, y, pred, idx);
      this.trees.push(tree);
      for (let i = 0; i < n; i++) pred[i] += o.lr * this._leafB(tree, XB, i);
    }
    return this;
  }

  _leafValue(ids, g, y, pred, lo, hi) {
    const o = this.o;
    let v;
    if (o.objective === 'quantile') {
      v = quantile(ids.map(i => y[i] - pred[i]), o.alpha);
    } else {
      let G = 0; for (const i of ids) G += g[i];
      v = -G / (ids.length + o.lambda);
    }
    return Math.min(hi, Math.max(lo, v));
  }

  _build(XB, g, y, pred, ids) {
    const o = this.o, F = XB.length;
    const nodes = [];
    const grow = (ids, depth, lo, hi) => {
      const node = { v: this._leafValue(ids, g, y, pred, lo, hi) };
      nodes.push(node);
      if (depth >= o.maxDepth || ids.length < 2 * o.minLeaf) return node;
      let best = null;
      let Gt = 0; for (const i of ids) Gt += g[i];
      const Nt = ids.length;
      const parentScore = (Gt * Gt) / (Nt + o.lambda);
      for (let f = 0; f < F; f++) {
        const nb = this.edges[f].length + 1;
        if (nb < 2) continue;
        const hg = new Float64Array(nb), hn = new Uint32Array(nb);
        const xb = XB[f];
        for (const i of ids) { hg[xb[i]] += g[i]; hn[xb[i]]++; }
        let GL = 0, NL = 0;
        for (let b = 0; b < nb - 1; b++) {
          GL += hg[b]; NL += hn[b];
          const NR = Nt - NL;
          if (NL < o.minLeaf || NR < o.minLeaf) continue;
          const GR = Gt - GL;
          let vl = -GL / (NL + o.lambda), vr = -GR / (NR + o.lambda);
          const m = this.mono[f];
          if (m) {
            vl = Math.min(hi, Math.max(lo, vl)); vr = Math.min(hi, Math.max(lo, vr));
            if (m * (vr - vl) < 0) continue;
          }
          const gain = (GL * GL) / (NL + o.lambda) + (GR * GR) / (NR + o.lambda) - parentScore;
          if (gain > 1e-9 && (!best || gain > best.gain)) best = { gain, f, b, vl, vr };
        }
      }
      if (!best) return node;
      const L = [], R = [];
      for (const i of ids) (XB[best.f][i] <= best.b ? L : R).push(i);
      node.f = best.f; node.b = best.b; node.t = this.edges[best.f][best.b]; node.gain = best.gain;
      let lL = lo, hL = hi, lR = lo, hR = hi;
      const m = this.mono[best.f];
      if (m) {
        const mid = (best.vl + best.vr) / 2;
        if (m > 0) { hL = mid; lR = mid; } else { lL = mid; hR = mid; }
      }
      node.l = grow(L, depth + 1, lL, hL);
      node.r = grow(R, depth + 1, lR, hR);
      return node;
    };
    const root = grow(ids, 0, -Infinity, Infinity);
    return root;
  }

  _leafB(node, XB, i) {
    while (node.l) node = XB[node.f][i] <= node.b ? node.l : node.r;
    return node.v;
  }

  // raw-value prediction; x[f] <= t goes left (t = bin upper edge)
  predict(x) {
    let s = this.base;
    for (const tr of this.trees) {
      let nd = tr;
      while (nd.l) nd = !(x[nd.f] >= nd.t) ? nd.l : nd.r;
      s += this.o.lr * nd.v;
    }
    return s;
  }

  // Tree-path attribution: base + sum(contrib) == predict(x)
  explain(x) {
    const F = this.names.length, c = new Float64Array(F);
    let base = this.base;
    for (const tr of this.trees) {
      let nd = tr; base += this.o.lr * nd.v;
      while (nd.l) {
        const nx = !(x[nd.f] >= nd.t) ? nd.l : nd.r;
        c[nd.f] += this.o.lr * (nx.v - nd.v);
        nd = nx;
      }
    }
    return { base, contrib: Array.from(c) };
  }

  importance() {
    const imp = new Array(this.names.length).fill(0);
    const walk = nd => { if (nd.l) { imp[nd.f] += nd.gain; walk(nd.l); walk(nd.r); } };
    this.trees.forEach(walk);
    const s = imp.reduce((a, b) => a + b, 0) || 1;
    return imp.map(v => v / s);
  }
}

// Note on thresholds: during training a row goes left when bin(x) <= b, i.e. x < edges[b]
// (bin = number of edges <= x). predict() uses x < t with t = edges[b], which matches.

// ---------- ridge regression baseline ----------
export function fitRidge(X, y, lambda = 1) {
  const n = X.length, F = X[0].length;
  const mu = new Array(F).fill(0), sd = new Array(F).fill(0);
  for (const r of X) r.forEach((v, f) => (mu[f] += v / n));
  for (const r of X) r.forEach((v, f) => (sd[f] += ((v - mu[f]) ** 2) / n));
  for (let f = 0; f < F; f++) sd[f] = Math.sqrt(sd[f]) || 1;
  const ym = y.reduce((a, b) => a + b, 0) / n;
  const A = Array.from({ length: F }, () => new Float64Array(F)), bv = new Float64Array(F);
  X.forEach((r, i) => {
    const z = r.map((v, f) => (v - mu[f]) / sd[f]);
    for (let a = 0; a < F; a++) { bv[a] += z[a] * (y[i] - ym); for (let b = 0; b < F; b++) A[a][b] += z[a] * z[b]; }
  });
  for (let a = 0; a < F; a++) A[a][a] += lambda;
  // Gaussian elimination
  const M = A.map((row, i) => [...row, bv[i]]);
  for (let c = 0; c < F; c++) {
    let p = c; for (let r = c + 1; r < F; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < F; r++) if (r !== c) {
      const k = M[r][c] / M[c][c];
      for (let k2 = c; k2 <= F; k2++) M[r][k2] -= k * M[c][k2];
    }
  }
  const w = M.map((row, i) => row[F] / row[i]);
  return { predict: x => ym + x.reduce((s, v, f) => s + w[f] * (v - mu[f]) / sd[f], 0) };
}

// ---------- metrics ----------
export const mae = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length;
export const rmse = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
export function r2(pred, y) {
  const m = y.reduce((a, b) => a + b, 0) / y.length;
  const ss = y.reduce((s, v) => s + (v - m) ** 2, 0);
  return 1 - y.reduce((s, v, i) => s + (v - pred[i]) ** 2, 0) / ss;
}

// Hotspot agreement: top-q blocks by prediction vs by observation, per scene
export function hotspotF1(pred, y, sceneIds, q = 0.1) {
  let tp = 0, fp = 0, fn = 0;
  const scenes = [...new Set(sceneIds)];
  for (const s of scenes) {
    const ids = sceneIds.map((v, i) => (v === s ? i : -1)).filter(i => i >= 0);
    if (ids.length < 10) continue;
    const k = Math.max(1, Math.round(ids.length * q));
    const tp_ = new Set([...ids].sort((a, b) => pred[b] - pred[a]).slice(0, k));
    const to_ = new Set([...ids].sort((a, b) => y[b] - y[a]).slice(0, k));
    for (const i of tp_) to_.has(i) ? tp++ : fp++;
    for (const i of to_) if (!tp_.has(i)) fn++;
  }
  const p = tp / (tp + fp || 1), r = tp / (tp + fn || 1);
  return { precision: p, recall: r, f1: (2 * p * r) / (p + r || 1) };
}

// ---------- Getis-Ord Gi* ----------
// values: array over cells (NaN allowed), neighbours: array of index arrays (incl. self)
export function getisOrd(values, neighbours) {
  const ok = values.map(Number.isFinite);
  const xs = values.filter((_, i) => ok[i]);
  const n = xs.length;
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const S = Math.sqrt(xs.reduce((a, b) => a + b * b, 0) / n - mean * mean) || 1;
  return values.map((_, i) => {
    if (!ok[i]) return NaN;
    const nb = neighbours[i].filter(j => ok[j]);
    const W = nb.length;
    const sx = nb.reduce((a, j) => a + values[j], 0);
    const den = S * Math.sqrt((n * W - W * W) / (n - 1));
    return den ? (sx - mean * W) / den : 0;
  });
}
