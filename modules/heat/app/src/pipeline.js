import { makeGrid, findScenes, loadScene, loadOSM, loadElevation, loadSceneWeather, loadForecast } from './data.js';
import { GBM, fitRidge, mae, rmse, r2, hotspotF1, getisOrd } from './ml.js';

export const FEATURES = [
  { key: 'ndvi', label: 'Vegetation (NDVI)', mono: -1, fmt: v => v.toFixed(2) },
  { key: 'ndbi', label: 'Built-up surface (NDBI)', mono: 1, fmt: v => v.toFixed(2) },
  { key: 'water', label: 'Open water share', mono: -1, fmt: v => (v * 100).toFixed(0) + '%' },
  { key: 'road', label: 'Main-road density', mono: 0, fmt: v => v.toFixed(1) + ' km/km²' },
  { key: 'dWater', label: 'Distance to water', mono: 0, fmt: v => v.toFixed(1) + ' km' },
  { key: 'dGreen', label: 'Distance to park/green', mono: 0, fmt: v => v.toFixed(1) + ' km' },
  { key: 'elev', label: 'Relative elevation', mono: 0, fmt: v => v.toFixed(0) + ' m' },
  { key: 'air', label: 'Air temperature', mono: 1, fmt: v => v.toFixed(1) + '°C' },
  { key: 'rh', label: 'Humidity', mono: 0, fmt: v => v.toFixed(0) + '%' },
  { key: 'wind', label: 'Wind speed', mono: 0, fmt: v => v.toFixed(0) + ' km/h' },
];
const FK = FEATURES.map(f => f.key);
// air temperature sets the city-wide level separately, so the block model leaves it unconstrained
const MONO = FEATURES.map(f => (f.key === 'air' ? 0 : f.mono));

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

// sin(solar noon elevation) for a latitude and date: a proxy for how much sun the surface gets
export function sunFactor(lat, date) {
  const d = new Date(date);
  const doy = (d - new Date(Date.UTC(d.getUTCFullYear(), 0, 0))) / 864e5;
  const decl = 23.44 * Math.sin((2 * Math.PI * (284 + doy)) / 365);
  return Math.sin(((90 - Math.abs(lat - decl)) * Math.PI) / 180);
}
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
function ols(X, y) {
  const F = X[0].length;
  const A = Array.from({ length: F }, (_, i) => Array.from({ length: F }, (_, j) => X.reduce((s, x) => s + x[i] * x[j], 0) + (i === j ? 1e-6 : 0)));
  const b = Array.from({ length: F }, (_, i) => X.reduce((s, x, n) => s + x[i] * y[n], 0));
  for (let c = 0; c < F; c++) for (let r = 0; r < F; r++) if (r !== c) {
    const f = A[r][c] / A[c][c];
    for (let k = c; k < F; k++) A[r][k] -= f * A[c][k];
    b[r] -= f * b[c];
  }
  return b.map((v, i) => v / A[i][i]);
}

// ---------- 1. data ----------
export async function buildDataset(place, halfKm, log) {
  const grid = makeGrid(place.lat, place.lon, halfKm);
  log(`Grid: ${grid.n}×${grid.n} blocks of ${grid.cellKm * 1000} m (${grid.cells.length} blocks)`);

  // OpenStreetMap and elevation don't depend on the scenes: start them now, in parallel
  log('Fetching OpenStreetMap roads, water and green space in the background…');
  const osmP = loadOSM(grid).then(o => { const c = o.counts, f = (v, w) => (v == null ? `${w} unavailable` : `${v} ${w}`); log(`  ✓ OSM: ${f(c.road, 'main roads')}, ${f(c.blue, 'water')}, ${f(c.green, 'green features')}`); return o; })
    .catch(e => { log(`  ! OSM unavailable (${e.message}) — continuing without it`); return null; });
  const elevP = loadElevation(grid).then(v => { log('  ✓ Elevation (Copernicus DEM via Open-Meteo)'); return v; });
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

  const weatherP = loadSceneWeather(grid, scenes).then(v => { log('  ✓ ERA5 weather at each satellite overpass'); return v; });
  const [osm, elev, weather] = await Promise.all([osmP, elevP, weatherP]);

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
  // Two-part model: a weather model sets the city-wide level for a day;
  // the GBM learns each block's deviation from that level from its surface features.
  const sceneMean = idx => {
    const acc = {};
    for (const i of idx) { const s = ds.rows[i].scene; (acc[s] ||= [0, 0]); acc[s][0] += y[i]; acc[s][1]++; }
    const out = {}; for (const s in acc) out[s] = acc[s][0] / acc[s][1]; return out;
  };
  const P = { gbm: [], lin: [], green: [], mean: [] }, Q10 = [], Q90 = [], Y = [], S = [];
  const gi = [FK.indexOf('ndvi')];
  for (let k = 0; k < 4; k++) {
    const tr = [], te = [];
    fold.forEach((f, i) => (f === k ? te : tr).push(i));
    if (te.length < 20 || tr.length < 50) continue;
    const sm = sceneMean(tr);
    const Xt = tr.map(i => X[i]), rt = tr.map(i => y[i] - sm[ds.rows[i].scene]);
    const g = new GBM({ monotone: MONO, nTrees: 140 }).fit(Xt, rt, FK);
    await tick();
    const q1 = new GBM({ objective: 'quantile', alpha: 0.1, nTrees: 90, monotone: MONO }).fit(Xt, rt, FK);
    const q9 = new GBM({ objective: 'quantile', alpha: 0.9, nTrees: 90, monotone: MONO }).fit(Xt, rt, FK);
    await tick();
    const lin = fitRidge(Xt, rt);
    const green = fitRidge(Xt.map(x => gi.map(j => x[j])), rt);
    for (const i of te) {
      const base = sm[ds.rows[i].scene] ?? 0;
      P.gbm.push(base + g.predict(X[i])); P.lin.push(base + lin.predict(X[i]));
      P.green.push(base + green.predict(gi.map(j => X[i][j]))); P.mean.push(base);
      Q10.push(base + q1.predict(X[i])); Q90.push(base + q9.predict(X[i])); Y.push(y[i]); S.push(ds.rows[i].scene);
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
  const smAll = sceneMean(y.map((_, i) => i));
  const r = y.map((v, i) => v - smAll[ds.rows[i].scene]);
  const model = new GBM({ monotone: MONO, nTrees: 200 }).fit(X, r, FK);
  await tick();
  const q10 = new GBM({ objective: 'quantile', alpha: 0.1, nTrees: 110, monotone: MONO }).fit(X, r, FK);
  const q90 = new GBM({ objective: 'quantile', alpha: 0.9, nTrees: 110, monotone: MONO }).fit(X, r, FK);

  // City-level model: the day's mean surface temperature from air temperature, optionally plus
  // seasonal sun angle. Both are scored leave-one-date-out and the better one is kept.
  const pts = Object.keys(smAll).map(s_ => ({ air: ds.scenes[s_].weather.air, sun: sunFactor(ds.grid.lat, ds.scenes[s_].datetime), y: smAll[s_] }));
  const candidates = [
    { name: 'air temperature', fx: p => [1, p.air] },
    { name: 'air temperature + seasonal sun angle', fx: p => [1, p.air, 10 * p.sun] },
  ];
  let best = null;
  for (const c of candidates) {
    if (pts.length < c.fx(pts[0]).length + 2) continue;
    const errs = pts.map((p, i) => {
      const w = ols(pts.filter((_, j) => j !== i).map(c.fx), pts.filter((_, j) => j !== i).map(q => q.y));
      return Math.abs(dot(c.fx(p), w) - p.y);
    });
    const loo = errs.reduce((x, y_) => x + y_, 0) / errs.length;
    if (!best || loo < best.loo - 0.05) best = { ...c, loo, w: ols(pts.map(c.fx), pts.map(q => q.y)) };
  }
  metrics.level = { uses: best.name, looMae: best.loo, n: pts.length, slope: best.w[1] };
  const level = (air, date) => dot(best.fx({ air, sun: sunFactor(ds.grid.lat, date) }), best.w);

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
  return { model, q10, q90, widen, level, metrics };
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
  const L = M.level(day.air, day.date);
  const out = ds.grid.cells.map((_, ci) => {
    const f = featuresFor(ds, ci, day, scenario);
    if (!f) return null;
    const x = vec(f);
    const p = L + M.model.predict(x);
    return { lst: p, lo: Math.min(p, L + M.q10.predict(x) - M.widen), hi: Math.max(p, L + M.q90.predict(x) + M.widen), x: f };
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
  return { cells: out, median: med, level: L };
}

export function explainCell(M, x, date) {
  const { base, contrib } = M.model.explain(vec(x));
  return { base: M.level(x.air, date) + base, items: FEATURES.map((f, i) => ({ ...f, value: x[f.key], c: contrib[i] })).sort((a, b) => Math.abs(b.c) - Math.abs(a.c)) };
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
