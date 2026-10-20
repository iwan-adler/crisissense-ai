import { makeGrid, findScenes, loadScene, loadOSM, loadElevation, loadSceneWeather, loadForecast } from './data.js';
import { GBM, fitRidge, mae, rmse, r2, hotspotF1, getisOrd } from './ml.js';

export const FEATURES = [
  { key: 'ndvi', label: 'Vegetation (NDVI)', mono: -1, fmt: v => v.toFixed(2) },
  { key: 'ndbi', label: 'Built-up surface (NDBI)', mono: 1, fmt: v => v.toFixed(2) },
  { key: 'water', label: 'Open water share', mono: -1, fmt: v => (v * 100).toFixed(0) + '%' },
  { key: 'road', label: 'Road density', mono: 0, fmt: v => v.toFixed(1) + ' km/km²' },
  { key: 'dWater', label: 'Distance to water', mono: 0, fmt: v => v.toFixed(1) + ' km' },
  { key: 'dGreen', label: 'Distance to park/green', mono: 0, fmt: v => v.toFixed(1) + ' km' },
  { key: 'elev', label: 'Relative elevation', mono: 0, fmt: v => v.toFixed(0) + ' m' },
  { key: 'air', label: 'Air temperature', mono: 1, fmt: v => v.toFixed(1) + '°C' },
  { key: 'rh', label: 'Humidity', mono: 0, fmt: v => v.toFixed(0) + '%' },
  { key: 'wind', label: 'Wind speed', mono: 0, fmt: v => v.toFixed(0) + ' km/h' },
];
const FK = FEATURES.map(f => f.key);
const MONO = FEATURES.map(f => f.mono);

// Feels-like risk bands (US NWS heat-index categories, °C)
export const RISK = [
  { name: 'Low', max: 27, color: '#7FA58E' },
  { name: 'Caution', max: 32, color: '#E9C77B' },
  { name: 'Extreme caution', max: 41, color: '#E8A33D' },
  { name: 'Danger', max: 54, color: '#D9572B' },
  { name: 'Extreme danger', max: Infinity, color: '#8E2414' },
];
export const riskOf = v => RISK.findIndex(b => v < b.max);
// Assumed share of a block's surface-heat anomaly that reaches the air people feel.
export const SURFACE_TO_AIR = 0.3;

const tick = () => new Promise(r => setTimeout(r, 0));
const median = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };

// ---------- 1. data ----------
export async function buildDataset(place, halfKm, log) {
  const grid = makeGrid(place.lat, place.lon, halfKm);
  log(`Grid: ${grid.n}×${grid.n} blocks of ${grid.cellKm * 1000} m (${grid.cells.length} blocks)`);

  log('Searching Landsat 8/9 archive (last 3 years)…');
  const found = await findScenes(grid);
  if (!found.length) throw new Error('No cloud-free Landsat scenes cover this area. Try a smaller radius or another city.');
  log(`Found ${found.length} clear scenes — downloading thermal + optical bands…`);

  const scenes = [];
  await Promise.all(found.map(async sc => {
    try {
      const s = await loadScene(grid, sc);
      if (s.valid >= 0.6) scenes.push(s);
      log(`  ✓ ${sc.datetime.slice(0, 10)} ${sc.platform} — ${(s.valid * 100).toFixed(0)}% usable`);
    } catch (e) { log(`  ✗ ${sc.datetime.slice(0, 10)} skipped (${e.message})`); }
  }));
  scenes.sort((a, b) => a.datetime.localeCompare(b.datetime));
  if (scenes.length < 2) throw new Error('Not enough usable satellite scenes for this area.');

  log('Fetching OpenStreetMap roads, water and green space…');
  const [osm, elev, weather] = await Promise.all([
    loadOSM(grid).then(o => { log(`  ✓ OSM: ${o.counts.roads} roads, ${o.counts.water} water, ${o.counts.green} green features`); return o; })
      .catch(e => { log(`  ! OSM unavailable (${e.message}) — continuing without it`); return null; }),
    loadElevation(grid).then(v => { log('  ✓ Elevation (Copernicus DEM via Open-Meteo)'); return v; }),
    loadSceneWeather(grid, scenes).then(v => { log('  ✓ ERA5 weather at each satellite overpass'); return v; }),
  ]);

  const N = grid.cells.length;
  const stat = {
    road: osm ? osm.roadDensity : new Array(N).fill(0),
    dWater: osm ? osm.distWater : new Array(N).fill(6),
    dGreen: osm ? osm.distGreen : new Array(N).fill(6),
    elev,
  };
  const rows = [];
  scenes.forEach((s, si) => {
    const w = weather[si];
    if (!w) return;
    s.weather = w;
    s.cells.forEach((c, ci) => {
      if (!c) return;
      rows.push({
        cell: ci, scene: si, y: c.lst,
        x: { ndvi: c.ndvi, ndbi: c.ndbi, water: c.water, road: stat.road[ci], dWater: stat.dWater[ci],
             dGreen: stat.dGreen[ci], elev: stat.elev[ci], air: w.air, rh: w.rh, wind: w.wind },
      });
    });
  });

  // latest valid surface state per block, used for forecasting
  const surface = grid.cells.map((_, ci) => {
    for (let k = scenes.length - 1; k >= 0; k--) if (scenes[k].cells[ci]) return { ...scenes[k].cells[ci], date: scenes[k].datetime.slice(0, 10) };
    return null;
  });
  const overpassUtc = median(scenes.map(s => new Date(s.datetime).getUTCHours() + new Date(s.datetime).getUTCMinutes() / 60));
  log(`Training set: ${rows.length.toLocaleString()} rows (blocks × dates) from ${scenes.length} scenes`);
  return {
    place, halfKm, grid: { ...grid }, osmOk: !!osm, osmCounts: osm?.counts,
    scenes: scenes.map(s => ({ id: s.id, datetime: s.datetime, platform: s.platform, cloud: s.cloud, valid: s.valid, weather: s.weather })),
    sceneCells: scenes.map(s => s.cells.map(c => (c ? +c.lst.toFixed(2) : null))),
    stat, surface, rows, overpassUtc,
  };
}

// ---------- 2. model ----------
const vec = x => FK.map(k => x[k]);

export async function trainAndEvaluate(ds, log) {
  const X = ds.rows.map(r => vec(r.x)), y = ds.rows.map(r => r.y);
  const n = ds.grid.n;
  // spatial blocks: 4×4 macro-blocks → 4 folds (no block's neighbours leak into its test fold)
  const macro = ds.rows.map(r => {
    const rr = Math.floor(r.cell / n), cc = r.cell % n;
    return Math.floor((rr * 4) / n) * 4 + Math.floor((cc * 4) / n);
  });
  const foldOf = m => ((Math.floor(m / 4) % 2) * 2 + (m % 4) % 2);
  const fold = macro.map(foldOf);

  log('Spatial cross-validation (4 folds of held-out neighbourhoods)…');
  const P = { gbm: [], lin: [], green: [], mean: [] }, Q10 = [], Q90 = [], Y = [], S = [];
  for (let k = 0; k < 4; k++) {
    const tr = [], te = [];
    fold.forEach((f, i) => (f === k ? te : tr).push(i));
    if (te.length < 20 || tr.length < 50) continue;
    const Xt = tr.map(i => X[i]), yt = tr.map(i => y[i]);
    const g = new GBM({ monotone: MONO, nTrees: 120 }).fit(Xt, yt, FK);
    await tick();
    const q1 = new GBM({ objective: 'quantile', alpha: 0.1, nTrees: 80, monotone: MONO }).fit(Xt, yt, FK);
    const q9 = new GBM({ objective: 'quantile', alpha: 0.9, nTrees: 80, monotone: MONO }).fit(Xt, yt, FK);
    await tick();
    const lin = fitRidge(Xt, yt);
    const gi = [FK.indexOf('ndvi'), FK.indexOf('air')];
    const green = fitRidge(Xt.map(x => gi.map(j => x[j])), yt);
    const sceneMean = {};
    tr.forEach(i => { const s = ds.rows[i].scene; (sceneMean[s] ||= []).push(y[i]); });
    for (const i of te) {
      P.gbm.push(g.predict(X[i])); P.lin.push(lin.predict(X[i]));
      P.green.push(green.predict(gi.map(j => X[i][j])));
      const sm = sceneMean[ds.rows[i].scene];
      P.mean.push(sm ? sm.reduce((a, b) => a + b, 0) / sm.length : 0);
      Q10.push(q1.predict(X[i])); Q90.push(q9.predict(X[i])); Y.push(y[i]); S.push(ds.rows[i].scene);
    }
    log(`  fold ${k + 1}/4 — GBM MAE ${mae(P.gbm.slice(-te.length), Y.slice(-te.length)).toFixed(2)} °C`);
  }
  const metrics = {};
  for (const [name, p] of Object.entries(P)) {
    metrics[name] = { mae: mae(p, Y), rmse: rmse(p, Y), r2: r2(p, Y), hot: hotspotF1(p, Y, S, 0.1) };
  }
  // conformal calibration of the P10–P90 band on held-out data
  const raw = Y.filter((v, i) => v >= Q10[i] && v <= Q90[i]).length / Y.length;
  const scores = Y.map((v, i) => Math.max(Q10[i] - v, v - Q90[i])).sort((a, b) => a - b);
  const widen = scores[Math.min(scores.length - 1, Math.ceil(0.8 * (scores.length + 1)) - 1)];
  const cal = Y.filter((v, i) => v >= Q10[i] - widen && v <= Q90[i] + widen).length / Y.length;
  metrics.interval = { raw, calibrated: cal, widen };
  metrics.nTest = Y.length;

  log('Training final models on all data…');
  await tick();
  const model = new GBM({ monotone: MONO, nTrees: 180 }).fit(X, y, FK);
  await tick();
  const q10 = new GBM({ objective: 'quantile', alpha: 0.1, nTrees: 100, monotone: MONO }).fit(X, y, FK);
  const q90 = new GBM({ objective: 'quantile', alpha: 0.9, nTrees: 100, monotone: MONO }).fit(X, y, FK);

  // monotonicity audit: greener must never be hotter, more built-up never cooler
  let viol = 0, checks = 0;
  for (let i = 0; i < X.length; i += 7) {
    const b = model.predict(X[i]);
    for (const [f, d] of [[0, 0.1], [0, 0.3], [1, 0.1], [1, 0.3]]) {
      const x2 = [...X[i]]; x2[f] += d; const v = model.predict(x2); checks++;
      if (f === 0 ? v > b + 1e-9 : v < b - 1e-9) viol++;
    }
  }
  metrics.mono = { viol, checks };
  metrics.importance = model.importance();
  log(`Done. Held-out MAE ${metrics.gbm.mae.toFixed(2)} °C vs ${metrics.mean.mae.toFixed(2)} °C for the city-average baseline.`);
  return { model, q10, q90, widen, metrics };
}

// ---------- 3. forecast + risk ----------
export function featuresFor(ds, ci, day, scenario) {
  const s = ds.surface[ci];
  if (!s) return null;
  let ndvi = s.ndvi, ndbi = s.ndbi;
  if (scenario && scenario.cells.has(ci)) {
    ndvi = Math.min(0.9, ndvi + 0.005 * scenario.trees - 0.004 * scenario.built);
    ndbi = Math.min(0.6, ndbi + 0.004 * scenario.built - 0.002 * scenario.trees);
  }
  return { ndvi, ndbi, water: s.water, road: ds.stat.road[ci], dWater: ds.stat.dWater[ci], dGreen: ds.stat.dGreen[ci],
    elev: ds.stat.elev[ci], air: day.air, rh: day.rh, wind: day.wind };
}

export function predictDay(ds, M, day, scenario) {
  const out = ds.grid.cells.map((_, ci) => {
    const f = featuresFor(ds, ci, day, scenario);
    if (!f) return null;
    const x = vec(f);
    const p = M.model.predict(x);
    return { lst: p, lo: Math.min(p, M.q10.predict(x) - M.widen), hi: Math.max(p, M.q90.predict(x) + M.widen), x: f };
  });
  const med = median(out.filter(Boolean).map(o => o.lst));
  const gi = getisOrd(out.map(o => (o ? o.lst : NaN)), ds.grid.neighbours);
  out.forEach((o, i) => {
    if (!o) return;
    o.anom = o.lst - med;
    o.feels = day.appMax + SURFACE_TO_AIR * o.anom;
    o.risk = riskOf(o.feels);
    o.gi = gi[i];
  });
  return { cells: out, median: med };
}

export function explainCell(M, x) {
  const { base, contrib } = M.model.explain(vec(x));
  return { base, items: FEATURES.map((f, i) => ({ ...f, value: x[f.key], c: contrib[i] })).sort((a, b) => Math.abs(b.c) - Math.abs(a.c)) };
}

export function datasetCSV(ds) {
  const head = ['block', 'row', 'col', 'lat', 'lon', 'date', ...FK, 'lst_c'];
  const lines = [head.join(',')];
  for (const r of ds.rows) {
    const c = ds.grid.cells[r.cell];
    lines.push([r.cell, c.r, c.c, c.lat.toFixed(5), c.lon.toFixed(5), ds.scenes[r.scene].datetime.slice(0, 10),
      ...FK.map(k => +r.x[k].toFixed(4)), r.y.toFixed(2)].join(','));
  }
  return lines.join('\n');
}
