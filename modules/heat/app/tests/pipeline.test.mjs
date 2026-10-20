// End-to-end test of training, evaluation, forecasting and explanations on a synthetic city
// (no network needed). Run: npm test  (or: node --test)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeGrid } from '../src/data.js';
import { trainAndEvaluate, predictDay, explainCell, sunFactor, datasetCSV } from '../src/pipeline.js';

function rng(seed = 3) { let s = seed; return () => ((s = (s * 16807) % 2147483647) / 2147483647); }

function syntheticCity() {
  const R = rng(), grid = makeGrid(26.9, 75.8, 6);
  const stat = { road: [], dWater: [], dGreen: [], elev: [] };
  const surf = grid.cells.map(() => ({ ndvi: R() * 0.6, ndbi: R() * 0.4 - 0.2, water: R() < 0.05 ? 0.8 : 0 }));
  grid.cells.forEach(() => { stat.road.push(R() * 20); stat.dWater.push(R() * 5); stat.dGreen.push(R() * 3); stat.elev.push(R() * 40 - 20); });
  const scenes = [], rows = [];
  for (let k = 0; k < 8; k++) {
    const weather = { air: 15 + k * 3 + R() * 2, rh: 30 + R() * 40, wind: R() * 15 };
    scenes.push({ id: `S${k}`, datetime: `2025-0${k + 1}-10T05:20:00Z`, weather });
    grid.cells.forEach((c, i) => {
      const b = surf[i];
      const y = 1.3 * weather.air + 8 + 10 * b.ndbi - 12 * b.ndvi - 6 * b.water + 0.05 * stat.road[i] + (R() - 0.5);
      rows.push({ cell: i, scene: k, y, x: { ...b, road: stat.road[i], dWater: stat.dWater[i], dGreen: stat.dGreen[i], elev: stat.elev[i], ...weather } });
    });
  }
  return { place: { name: 'Testville' }, grid, rows, scenes, stat, surface: surf.map(b => ({ ...b, lst: 30 })), sceneCells: [] };
}

const ds = syntheticCity();
const M = await trainAndEvaluate(ds, () => {});
const day = { date: '2025-05-01', air: 32, rh: 40, wind: 5, appMax: 35 };

test('block model beats the same-day city-average baseline on held-out neighbourhoods', () => {
  assert.ok(M.metrics.gbm.mae < M.metrics.mean.mae * 0.5, JSON.stringify(M.metrics.gbm));
  assert.ok(M.metrics.gbm.hot.f1 > M.metrics.mean.hot.f1);
});

test('calibrated confidence band covers about 80% of held-out data', () => {
  assert.ok(Math.abs(M.metrics.interval.calibrated - 0.8) < 0.03, String(M.metrics.interval.calibrated));
});

test('city-level model recovers the air-temperature slope and the physics audit passes', () => {
  assert.ok(Math.abs(M.metrics.level.slope - 1.3) < 0.1, String(M.metrics.level.slope));
  assert.equal(M.metrics.mono.viol, 0);
});

test('forecast, risk, hotspots and explanations are consistent', () => {
  const p = predictDay(ds, M, day);
  const cells = p.cells.filter(Boolean);
  assert.equal(cells.length, ds.grid.cells.length);
  for (const c of cells) {
    assert.ok(c.lo <= c.lst && c.lst <= c.hi);
    assert.ok(Number.isInteger(c.risk) && c.risk >= 0);
    assert.ok(Number.isFinite(c.gi));
  }
  const e = explainCell(M, cells[0].x, day.date);
  assert.ok(Math.abs(e.base + e.items.reduce((a, b) => a + b.c, 0) - cells[0].lst) < 1e-6);
});

test('what-if: more trees cools, new development heats', () => {
  const base = predictDay(ds, M, day);
  const ids = new Set([0, 1, 2, 3, 4]);
  const trees = predictDay(ds, M, day, { trees: 30, built: 0, cells: ids });
  const built = predictDay(ds, M, day, { trees: 0, built: 30, cells: ids });
  for (const i of ids) {
    assert.ok(trees.cells[i].lst <= base.cells[i].lst + 1e-9);
    assert.ok(built.cells[i].lst >= base.cells[i].lst - 1e-9);
  }
});

test('sun factor peaks near the local summer solstice', () => {
  assert.ok(sunFactor(26.9, '2025-06-21') > sunFactor(26.9, '2025-12-21'));
  assert.ok(sunFactor(-33.9, '2025-12-21') > sunFactor(-33.9, '2025-06-21'));
});

test('dataset exports as CSV with one row per block-date', () => {
  const lines = datasetCSV(ds).split('\n');
  assert.equal(lines.length, ds.rows.length + 1);
  assert.match(lines[0], /^block,row,col,lat,lon,date,ndvi,ndbi/);
});
