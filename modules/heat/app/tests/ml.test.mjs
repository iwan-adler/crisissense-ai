// Unit tests for the in-browser ML engine. Run: npm test  (or: node --test)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GBM, fitRidge, mae, r2, hotspotF1, getisOrd } from '../src/ml.js';

// deterministic pseudo-random numbers
function rng(seed = 1) { let s = seed; return () => ((s = (s * 16807) % 2147483647) / 2147483647); }

function synthetic(n = 3000, seed = 1) {
  const R = rng(seed), X = [], y = [];
  for (let i = 0; i < n; i++) {
    const ndvi = R() * 0.7, ndbi = R() - 0.5, air = 25 + R() * 20, elev = R() * 100;
    X.push([ndvi, ndbi, air, elev]);
    y.push(air + 12 * ndbi - 15 * ndvi + 3 * Math.sin(elev / 10) + (R() - 0.5) * 2);
  }
  return { X, y };
}
const NAMES = ['ndvi', 'ndbi', 'air', 'elev'];

test('GBM learns a non-linear signal and beats a ridge baseline', () => {
  const { X, y } = synthetic();
  const tr = X.slice(0, 2400), ty = y.slice(0, 2400), te = X.slice(2400), yt = y.slice(2400);
  const m = new GBM({ monotone: [-1, 1, 0, 0] }).fit(tr, ty, NAMES);
  const p = te.map(x => m.predict(x));
  const ridge = fitRidge(tr, ty);
  assert.ok(r2(p, yt) > 0.95, `R² too low: ${r2(p, yt)}`);
  assert.ok(mae(p, yt) < mae(te.map(ridge.predict), yt), 'GBM should beat ridge on non-linear data');
});

test('monotone constraints hold: greener never hotter, more built-up never cooler', () => {
  const { X, y } = synthetic();
  const m = new GBM({ monotone: [-1, 1, 0, 0] }).fit(X, y, NAMES);
  let violations = 0;
  for (const x of X.slice(0, 500)) {
    const base = m.predict(x);
    for (let d = 0.05; d < 0.6; d += 0.05) {
      if (m.predict([x[0] + d, x[1], x[2], x[3]]) > base + 1e-9) violations++;
      if (m.predict([x[0], x[1] + d, x[2], x[3]]) < base - 1e-9) violations++;
    }
  }
  assert.equal(violations, 0);
});

test('explanations are exactly additive (base + contributions = prediction)', () => {
  const { X, y } = synthetic(1000);
  const m = new GBM().fit(X, y, NAMES);
  for (const x of X.slice(0, 50)) {
    const { base, contrib } = m.explain(x);
    assert.ok(Math.abs(base + contrib.reduce((a, b) => a + b, 0) - m.predict(x)) < 1e-9);
  }
});

test('quantile models bracket most observations', () => {
  const { X, y } = synthetic();
  const q10 = new GBM({ objective: 'quantile', alpha: 0.1, nTrees: 120 }).fit(X, y, NAMES);
  const q90 = new GBM({ objective: 'quantile', alpha: 0.9, nTrees: 120 }).fit(X, y, NAMES);
  const cover = X.filter((x, i) => y[i] >= q10.predict(x) && y[i] <= q90.predict(x)).length / X.length;
  assert.ok(cover > 0.6 && cover < 0.95, `coverage ${cover}`);
});

test('missing values are routed consistently in training and prediction', () => {
  const { X, y } = synthetic(800);
  X.forEach((x, i) => { if (i % 10 === 0) x[3] = NaN; });
  const m = new GBM().fit(X, y, NAMES);
  assert.ok(X.every(x => Number.isFinite(m.predict(x))));
});

test('hotspot F1 is 1 for a perfect ranking and low for a reversed one', () => {
  const y = Array.from({ length: 100 }, (_, i) => i), s = new Array(100).fill(0);
  assert.equal(hotspotF1(y, y, s).f1, 1);
  assert.equal(hotspotF1(y.map(v => -v), y, s).f1, 0);
});

test('Getis-Ord Gi* flags the hot end of a gradient positive and the cold end negative', () => {
  const v = [1, 2, 3, 10, 11, 12];
  const nb = [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5]];
  const z = getisOrd(v, nb);
  assert.ok(z[5] > 1 && z[0] < -1);
});
