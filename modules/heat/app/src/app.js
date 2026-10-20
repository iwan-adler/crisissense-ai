import { geocode, loadForecast } from './data.js';
import { buildDataset, trainAndEvaluate, predictDay, explainCell, datasetCSV, FEATURES, RISK, SURFACE_TO_AIR } from './pipeline.js';

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* quota or blocked: fine */ } },
};

const S = { ds: null, M: null, fc: null, day: 0, layer: 'heat', obs: false, preds: [], sel: null,
  scenario: null, rects: [], running: false };

// ---------- map ----------
const map = L.map('map', { zoomControl: true, preferCanvas: true }).setView([22, 20], 3);
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/';
L.tileLayer(ESRI + 'World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 16, attribution: 'Basemap © Esri, HERE, Garmin, © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors · Landsat: USGS',
}).addTo(map);
map.createPane('labels');
map.getPane('labels').style.zIndex = 650;
map.getPane('labels').style.pointerEvents = 'none';
L.tileLayer(ESRI + 'World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}', { maxZoom: 16, pane: 'labels' }).addTo(map);
const gridLayer = L.layerGroup().addTo(map);
const selLayer = L.layerGroup().addTo(map);

// ---------- colour scales ----------
const HEAT = ['#2F5D62', '#5E8C7A', '#A9BE8F', '#E9C77B', '#E8A33D', '#D9572B', '#A8321B', '#6E1C10'];
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const p = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), f = p - i;
  const a = stops[i].match(/\w\w/g).map(h => parseInt(h, 16)), b = stops[i + 1].match(/\w\w/g).map(h => parseInt(h, 16));
  return '#' + a.map((v, k) => Math.round(v + (b[k] - v) * f).toString(16).padStart(2, '0')).join('');
}
const UNC = ['#F3ECF7', '#C9B6DA', '#8F72B3', '#4F3A7A'];
const pct = (arr, q) => { const s = arr.filter(Number.isFinite).sort((a, b) => a - b); return s[Math.floor((s.length - 1) * q)]; };

// ---------- search ----------
let sugg = [], suggTimer;
$('#q').addEventListener('input', e => {
  clearTimeout(suggTimer);
  const v = e.target.value.trim();
  if (v.length < 2) { $('#suggest').hidden = true; return; }
  suggTimer = setTimeout(async () => {
    try { sugg = await geocode(v); } catch { sugg = []; }
    const ul = $('#suggest');
    ul.innerHTML = sugg.map((p, i) => `<li role="option" data-i="${i}">${esc(p.name)} <small>${esc([p.admin, p.country].filter(Boolean).join(', '))}</small></li>`).join('');
    ul.hidden = !sugg.length;
  }, 250);
});
$('#suggest').addEventListener('click', e => {
  const li = e.target.closest('li'); if (!li) return;
  const p = sugg[+li.dataset.i]; $('#suggest').hidden = true; $('#q').value = p.name; run(p);
});
document.addEventListener('click', e => { if (!e.target.closest('.search')) $('#suggest').hidden = true; });
$('#search').addEventListener('submit', async e => {
  e.preventDefault();
  $('#suggest').hidden = true;
  const q = $('#q').value.trim(); if (!q) return;
  try {
    const r = sugg.length && sugg[0].name.toLowerCase() === q.toLowerCase() ? sugg : await geocode(q);
    if (!r.length) return toast(`Couldn’t find “${q}”. Try adding the country.`, false);
    run(r[0]);
  } catch (err) { toast('Geocoding failed: ' + err.message, false); }
});

const DEMO = [
  { name: 'Jaipur', country: 'India', lat: 26.9124, lon: 75.7873 },
  { name: 'Delhi', country: 'India', lat: 28.6139, lon: 77.209 },
  { name: 'Ahmedabad', country: 'India', lat: 23.0225, lon: 72.5714 },
  { name: 'Phoenix', country: 'United States', lat: 33.4484, lon: -112.074 },
  { name: 'Cairo', country: 'Egypt', lat: 30.0444, lon: 31.2357 },
  { name: 'Madrid', country: 'Spain', lat: 40.4168, lon: -3.7038 },
];
$('#chips').innerHTML = DEMO.map((p, i) => `<button class="chip" data-i="${i}">${p.name}</button>`).join('');
$('#chips').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) { $('#q').value = DEMO[+b.dataset.i].name; run(DEMO[+b.dataset.i]); } });
$('#aboutBtn').addEventListener('click', () => $('#about').showModal());

function toast(msg, fatal = true) {
  $('#status').hidden = false;
  // a failed search keeps the current city's results on screen, so only flag a real failure
  if (fatal || !S.ds) { $('#phase').textContent = 'Error'; $('#phase').className = 'pill err'; }
  $('#logBox').open = true;
  log('⚠ ' + msg);
}
function log(m) {
  const li = document.createElement('li'); li.textContent = m; $('#log').append(li); li.scrollIntoView({ block: 'nearest' });
  $('#busyText').textContent = m.trim();
}

// ---------- run pipeline ----------
async function run(place, { refetch = false } = {}) {
  if (S.running) return;
  S.running = true;
  const half = +$('#radius').value;
  const key = `loomap:v2:${place.lat.toFixed(3)},${place.lon.toFixed(3)},${half}`;
  Object.assign(S, { ds: null, M: null, fc: null, sel: null, scenario: null, day: 0 });
  closePanel(); gridLayer.clearLayers(); selLayer.clearLayers(); $('#scenarioBar').hidden = true;
  $('#intro').hidden = true; $('#status').hidden = false;
  ['#forecast', '#layers', '#evalCard'].forEach(s => ($(s).hidden = true));
  $('#cityName').textContent = place.name;
  $('#citySub').textContent = [place.admin, place.country].filter(Boolean).join(', ');
  $('#phase').textContent = 'Working…'; $('#phase').className = 'pill';
  $('#log').innerHTML = ''; $('#logBox').open = true;
  $('#busy').hidden = false;
  const half2 = half * 1.05;
  map.fitBounds([[place.lat - half2 / 111, place.lon - half2 / (111 * Math.cos(place.lat * Math.PI / 180))],
    [place.lat + half2 / 111, place.lon + half2 / (111 * Math.cos(place.lat * Math.PI / 180))]]);
  const t0 = performance.now();
  const lg = m => log(`${((performance.now() - t0) / 1000).toFixed(1)}s  ${m}`);
  try {
    let ds = refetch ? null : store.get(key);
    if (ds && ds.rows?.length) lg('Using data cached in this browser (press “Re-fetch data” to refresh)');
    else { ds = await buildDataset(place, half, lg); store.set(key, ds); }
    S.ds = ds;
    S.M = await trainAndEvaluate(ds, lg);
    lg('Fetching 3-day weather forecast…');
    S.fc = await loadForecast(ds.grid, ds.overpassUtc);
    computePreds();
    $('#phase').textContent = 'Ready'; $('#phase').className = 'pill ok';
    $('#logBox').open = false;
    renderForecast(); renderEval(); renderMap();
    ['#forecast', '#layers', '#evalCard'].forEach(s => ($(s).hidden = false));
    $('#citySub').textContent = `${[place.admin, place.country].filter(Boolean).join(', ')} · ${ds.grid.cells.length} blocks · ${ds.scenes.length} Landsat scenes · ${ds.rows.length.toLocaleString()} training rows`;
  } catch (err) {
    console.error(err);
    toast(err.message || String(err));
  } finally {
    $('#busy').hidden = true; S.running = false;
  }
}
$('#rerunBtn').addEventListener('click', () => S.ds && run(S.ds.place, { refetch: true }));
$('#csvBtn').addEventListener('click', () => {
  const blob = new Blob([datasetCSV(S.ds)], { type: 'text/csv' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = `loomap_${S.ds.place.name.replace(/\W+/g, '_').toLowerCase()}_dataset.csv`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
});

function computePreds() {
  S.preds = S.fc.days.map(d => predictDay(S.ds, S.M, d, S.scenario));
  if (S.scenario) S.base = S.fc.days.map(d => predictDay(S.ds, S.M, d, null));
  else S.base = S.preds;
}

// ---------- sidebar ----------
const riskBadge = i => `<span class="badge" style="background:${RISK[i].color};color:${i >= 3 ? '#fff' : '#1F1612'}">${RISK[i].name}</span>`;
const riskIdx = v => RISK.findIndex(b => v < b.max);
function renderForecast() {
  const fd = new Intl.DateTimeFormat('en', { weekday: 'short', day: 'numeric', month: 'short' });
  $('#days').innerHTML = S.fc.days.map((d, i) => {
    const hot = S.preds[i].cells.filter(c => c && c.risk >= 3).length;
    return `<button class="day ${i === S.day ? 'on' : ''}" data-i="${i}">
      <div class="d">${fd.format(new Date(d.date + 'T12:00'))}</div>
      <div class="t">${fmt(d.appMax, 0)}°</div>
      <div class="s">feels-like max · air ${fmt(d.tMax, 0)}°</div>
      ${riskBadge(riskIdx(d.appMax))}
      <div class="s">${hot} block${hot === 1 ? '' : 's'} at Danger+</div></button>`;
  }).join('');
  $('#fcNote').textContent = `Block forecasts use weather at ${String(S.fc.localHour).padStart(2, '0')}:00 local (${S.fc.tz}), the time Landsat passes over.`;
}
$('#days').addEventListener('click', e => {
  const b = e.target.closest('.day'); if (!b) return;
  S.day = +b.dataset.i; renderForecast(); renderMap(); if (S.sel != null) openPanel(S.sel);
});
$('#layerSeg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  S.layer = b.dataset.layer;
  [...$('#layerSeg').children].forEach(x => x.classList.toggle('on', x === b));
  renderMap();
});
$('#obsToggle').addEventListener('change', e => { S.obs = e.target.checked; renderMap(); });

function renderEval() {
  const m = S.M.metrics;
  const rows = [['LooMap (GBM)', m.gbm], ['Linear model', m.lin], ['Greenery-only', m.green], ['City average', m.mean]];
  $('#evalTbl').innerHTML = `<tr><th>Model</th><th>MAE °C</th><th>R²</th><th>Hotspot F1</th></tr>` +
    rows.map(([n, v], i) => `<tr class="${i === 0 ? 'best' : ''}"><td>${n}</td><td>${fmt(v.mae, 2)}</td><td>${fmt(v.r2, 2)}</td><td>${fmt(v.hot.f1, 2)}</td></tr>`).join('');
  const gain = (1 - m.gbm.mae / m.mean.mae) * 100;
  $('#evalMore').innerHTML = `
    <p class="ok-line">✓ <span><b>${fmt(gain, 0)}% lower error</b> than the same-day city average, on ${m.nTest.toLocaleString()} held-out block-days from neighbourhoods the model never saw.</span></p>
    <p class="ok-line">✓ <span>Confidence band: raw P10–P90 covered ${fmt(m.interval.raw * 100, 0)}% → <b>${fmt(m.interval.calibrated * 100, 0)}% after calibration</b> (target 80%).</span></p>
    <p class="ok-line">${m.mono.viol ? '✗' : '✓'} <span>Physics check: <b>${m.mono.viol} violations</b> in ${m.mono.checks.toLocaleString()} tests (greener never hotter, more built-up never cooler).</span></p>
    <p class="ok-line">ℹ <span>City-wide daily level predicted from ${m.level.uses} (leave-one-date-out error ${fmt(m.level.looMae, 1)}°C over ${m.level.n} dates). The table above scores the block-level model.</span></p>
    <p class="ok-line">ℹ <span>Hotspot F1 = overlap between predicted and observed hottest 10% of blocks.</span></p>`;
  const imp = FEATURES.map((f, i) => ({ f, v: m.importance[i] })).sort((a, b) => b.v - a.v);
  const max = imp[0].v || 1;
  $('#imp').innerHTML = imp.map(({ f, v }) => `<div class="bar"><span>${f.label}</span><span class="track"><span class="fill" style="width:${(v / max) * 100}%"></span></span><span class="v">${fmt(v * 100, 0)}%</span></div>`).join('');
}

// ---------- map rendering ----------
function cellValue(i) {
  const p = S.preds[S.day].cells[i];
  if (S.obs) {
    const s = S.ds.surface[i];
    return s ? { v: s.lst, p } : null;
  }
  return p ? { v: p.lst, p } : null;
}
function renderMap() {
  gridLayer.clearLayers();
  const ds = S.ds; if (!ds) return;
  const vals = ds.grid.cells.map((_, i) => cellValue(i));
  const heatVals = vals.map(o => (o ? o.v : NaN));
  const lo = pct(heatVals, 0.02), hi = pct(heatVals, 0.98);
  const unc = S.preds[S.day].cells.map(c => (c ? c.hi - c.lo : NaN));
  const uLo = pct(unc, 0.02), uHi = pct(unc, 0.98);
  ds.grid.cells.forEach((c, i) => {
    const o = vals[i]; if (!o) return;
    const p = o.p;
    let color, op = 0.72;
    if (S.layer === 'heat') color = ramp(HEAT, (o.v - lo) / (hi - lo || 1));
    else if (S.layer === 'risk') color = RISK[p.risk].color;
    else if (S.layer === 'unc') color = ramp(UNC, (p.hi - p.lo - uLo) / (uHi - uLo || 1));
    else {
      if (p.gi >= 2.58) color = '#8E2414'; else if (p.gi >= 1.96) color = '#D9572B';
      else if (p.gi <= -1.96) color = '#3E6E73'; else { color = '#FFFFFF'; op = 0.15; }
    }
    const r = L.rectangle(c.bounds, { stroke: true, weight: 0.4, color: '#FFFAF3', fillColor: color, fillOpacity: op });
    r.bindTooltip(() => `<b>${fmt(o.v)}°C</b> ${S.obs ? 'observed' : 'predicted'} surface<br>${RISK[p.risk].name} · feels ~${fmt(p.feels, 0)}°C`, { className: 'tip', sticky: true });
    r.on('click', () => openPanel(i));
    gridLayer.addLayer(r);
  });
  renderLegend(lo, hi, uLo, uHi);
  drawSelection();
}
function renderLegend(lo, hi, uLo, uHi) {
  const L_ = $('#legend');
  if (S.layer === 'heat') L_.innerHTML = `<div class="ramp" style="background:linear-gradient(90deg,${HEAT.join(',')})"></div><div class="ramp-l"><span>${fmt(lo, 0)}°C</span><span>${S.obs ? 'observed' : 'predicted'} land-surface temperature</span><span>${fmt(hi, 0)}°C</span></div>`;
  else if (S.layer === 'risk') L_.innerHTML = `<div class="keys">${RISK.map(r => `<span class="key"><i class="sw" style="background:${r.color}"></i>${r.name}</span>`).join('')}</div><p class="muted small" style="margin-top:6px">Feels-like = forecast apparent max + ${SURFACE_TO_AIR} × block surface anomaly (NWS bands).</p>`;
  else if (S.layer === 'unc') L_.innerHTML = `<div class="ramp" style="background:linear-gradient(90deg,${UNC.join(',')})"></div><div class="ramp-l"><span>${fmt(uLo, 1)}°</span><span>width of 80% confidence band</span><span>${fmt(uHi, 1)}°</span></div>`;
  else L_.innerHTML = `<div class="keys"><span class="key"><i class="sw" style="background:#8E2414"></i>Hotspot 99%</span><span class="key"><i class="sw" style="background:#D9572B"></i>Hotspot 95%</span><span class="key"><i class="sw" style="background:#3E6E73"></i>Cool spot 95%</span></div><p class="muted small" style="margin-top:6px">Getis-Ord Gi* on predicted heat, 3×3 neighbourhood.</p>`;
}

// ---------- block panel ----------
function closePanel() { $('#panel').hidden = true; S.sel = null; selLayer.clearLayers(); setTimeout(() => map.invalidateSize(), 50); }
$('#closePanel').addEventListener('click', closePanel);
function drawSelection() {
  selLayer.clearLayers();
  if (!S.ds) return;
  const ids = new Set(S.scenario ? [...S.scenario.cells] : []);
  if (S.sel != null) ids.add(S.sel);
  for (const i of ids) selLayer.addLayer(L.rectangle(S.ds.grid.cells[i].bounds, { color: '#1F1612', weight: i === S.sel ? 2.5 : 1.2, fill: false, dashArray: i === S.sel ? null : '3 3', interactive: false }));
}

function sparkline(i) {
  const ds = S.ds;
  const pts = ds.scenes.map((s, k) => ({ d: s.datetime.slice(0, 10), v: ds.sceneCells[k][i], air: s.weather?.air })).filter(p => p.v != null);
  if (pts.length < 2) return '<p class="muted small">Not enough clear observations.</p>';
  const W = 330, H = 70, vs = pts.map(p => p.v), mn = Math.min(...vs) - 1, mx = Math.max(...vs) + 1;
  const x = k => 8 + (k / (pts.length - 1)) * (W - 16), y = v => H - 12 - ((v - mn) / (mx - mn)) * (H - 24);
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="Observed surface temperature on each satellite date">
    <polyline fill="none" stroke="#D9572B" stroke-width="2" points="${pts.map((p, k) => `${x(k)},${y(p.v)}`).join(' ')}"/>
    ${pts.map((p, k) => `<circle cx="${x(k)}" cy="${y(p.v)}" r="3" fill="#A8321B"><title>${p.d}: ${p.v.toFixed(1)}°C surface, air ${fmt(p.air)}°C</title></circle>`).join('')}
    <text x="4" y="${H - 1}" font-size="10" fill="#7A6658" font-family="JetBrains Mono">${pts[0].d}</text>
    <text x="${W - 4}" y="${H - 1}" font-size="10" fill="#7A6658" font-family="JetBrains Mono" text-anchor="end">${pts[pts.length - 1].d}</text>
  </svg>`;
}

function openPanel(i) {
  S.sel = i;
  const p = S.preds[S.day].cells[i];
  const c = S.ds.grid.cells[i];
  if (!p) { closePanel(); return; }
  const allP = S.preds[S.day].cells.filter(Boolean);
  const rank = allP.filter(o => o.lst > p.lst).length + 1;
  const ex = explainCell(S.M, p.x, S.fc.days[S.day].date);
  const maxC = Math.max(...ex.items.map(it => Math.abs(it.c)), 0.5);
  const lo = Math.min(...allP.map(o => o.lo)), hi = Math.max(...allP.map(o => o.hi));
  const X = v => ((v - lo) / (hi - lo)) * 100;
  const gi = p.gi >= 2.58 ? 'Hotspot (99% confidence)' : p.gi >= 1.96 ? 'Hotspot (95% confidence)' : p.gi <= -1.96 ? 'Cool spot (95%)' : 'Not a significant cluster';
  const day = S.fc.days[S.day];
  const sc = S.scenario || { trees: 0, built: 0, cells: new Set() };
  $('#panelBody').innerHTML = `
    <p class="eyebrow">Block ${c.r + 1}-${c.c + 1} · ${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}</p>
    <div class="row between"><div><div class="big">${fmt(p.lst)}°C</div><p class="muted small">predicted surface temp · ${day.date}</p></div>${riskBadge(p.risk)}</div>
    <div class="band" title="80% confidence band"><span class="rng" style="left:${X(p.lo)}%;width:${X(p.hi) - X(p.lo)}%"></span><span class="pt" style="left:${X(p.lst)}%"></span></div>
    <p class="muted small">80% band ${fmt(p.lo)}–${fmt(p.hi)}°C</p>
    <div class="kv">
      <span>Feels-like (est.)</span><span>${fmt(p.feels, 0)}°C</span>
      <span>vs city median block</span><span>${p.anom >= 0 ? '+' : ''}${fmt(p.anom)}°C</span>
      <span>Rank</span><span>#${rank} of ${allP.length}</span>
      <span>Cluster (Gi* z=${fmt(p.gi, 2)})</span><span>${gi}</span>
    </div>
    <hr class="sectionline">
    <h3>Why is it ${p.anom >= 0 ? 'hot' : 'cooler'} here?</h3>
    <p class="muted small">How each factor pushes this block above or below the model’s average (${fmt(ex.base)}°C).</p>
    ${ex.items.slice(0, 7).map(it => {
      const w = (Math.abs(it.c) / maxC) * 50;
      return `<div class="drv"><span class="lbl">${it.label}<small>${it.fmt(it.value)}</small></span>
        <span class="axis"><span class="seg2" style="${it.c >= 0 ? `left:50%;width:${w}%;background:#D9572B` : `left:${50 - w}%;width:${w}%;background:#3E6E73`}"></span></span>
        <span class="num">${it.c >= 0 ? '+' : ''}${it.c.toFixed(1)}°</span></div>`;
    }).join('')}
    <hr class="sectionline">
    <h3>Observed by Landsat</h3>
    ${sparkline(i)}
    <hr class="sectionline">
    <h3>What if…</h3>
    <div class="slider"><label for="sTrees"><span>More tree cover</span><span id="vTrees">+${sc.trees}%</span></label><input id="sTrees" type="range" min="0" max="40" step="5" value="${sc.trees}"></div>
    <div class="slider"><label for="sBuilt"><span>New development (built-up)</span><span id="vBuilt">+${sc.built}%</span></label><input id="sBuilt" type="range" min="0" max="40" step="5" value="${sc.built}"></div>
    <div class="row gap" style="margin-top:4px">
      <button class="btn small" id="applyBlock">Apply to this block</button>
      <button class="btn small" id="applyHot">Apply to all hotspots</button>
      <button class="btn small ghost" id="clearSc">Reset</button>
    </div>
    <div id="scResult"></div>`;
  $('#panel').hidden = false;
  setTimeout(() => map.invalidateSize(), 50);
  drawSelection();
  const upd = () => { $('#vTrees').textContent = `+${$('#sTrees').value}%`; $('#vBuilt').textContent = `+${$('#sBuilt').value}%`; };
  $('#sTrees').oninput = upd; $('#sBuilt').oninput = upd;
  $('#applyBlock').onclick = () => applyScenario(new Set([i]));
  $('#applyHot').onclick = () => applyScenario(new Set(S.base[S.day].cells.map((o, k) => (o && o.gi >= 1.96 ? k : -1)).filter(k => k >= 0)));
  $('#clearSc').onclick = () => { S.scenario = null; computePreds(); renderForecast(); renderMap(); openPanel(i); $('#scenarioBar').hidden = true; };
  renderScenarioResult();
}

function applyScenario(cells) {
  const trees = +$('#sTrees').value, built = +$('#sBuilt').value;
  if (!cells.size || (!trees && !built)) { S.scenario = null; } else S.scenario = { trees, built, cells };
  computePreds(); renderForecast(); renderMap();
  const keep = S.sel; openPanel(keep);
}
function renderScenarioResult() {
  const box = $('#scResult'), bar = $('#scenarioBar');
  if (!S.scenario) { box.innerHTML = ''; bar.hidden = true; return; }
  const ids = [...S.scenario.cells], d = S.day;
  const dl = ids.map(k => S.preds[d].cells[k].lst - S.base[d].cells[k].lst);
  const df = ids.map(k => S.preds[d].cells[k].feels - S.base[d].cells[k].feels);
  const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
  const before = ids.filter(k => S.base[d].cells[k].risk >= 3).length, after = ids.filter(k => S.preds[d].cells[k].risk >= 3).length;
  const txt = `${S.scenario.trees ? `+${S.scenario.trees}% trees` : ''}${S.scenario.trees && S.scenario.built ? ', ' : ''}${S.scenario.built ? `+${S.scenario.built}% built-up` : ''} on ${ids.length} block${ids.length > 1 ? 's' : ''}`;
  box.innerHTML = `<div class="result"><p class="muted small">${txt}</p>
    <p>Surface <b>${avg(dl) >= 0 ? '+' : ''}${fmt(avg(dl))}°C</b> · feels-like <b>${avg(df) >= 0 ? '+' : ''}${fmt(avg(df))}°C</b></p>
    <p class="small">Blocks at Danger or worse: ${before} → ${after}</p></div>`;
  bar.hidden = false;
  bar.innerHTML = `<span>Scenario on: ${txt} → ${avg(dl) >= 0 ? '+' : ''}${fmt(avg(dl))}°C surface</span><button class="btn small" id="barReset">Reset</button>`;
  $('#barReset').onclick = () => { S.scenario = null; computePreds(); renderForecast(); renderMap(); if (S.sel != null) openPanel(S.sel); bar.hidden = true; };
}

// deep link: ?city=Name
const qp = new URLSearchParams(location.search).get('city');
if (qp) { $('#q').value = qp; geocode(qp).then(r => r.length && run(r[0])).catch(() => {}); }
