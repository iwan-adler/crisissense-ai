// LooMap data pipeline — everything is fetched live from open, CORS-enabled APIs.
//   Landsat 8/9 Collection-2 Level-2 (USGS)  via Microsoft Planetary Computer
//   OpenStreetMap                            via Overpass API
//   Elevation, ERA5 weather + forecasts      via Open-Meteo

const PC = 'https://planetarycomputer.microsoft.com/api';
const OVERPASS = [
  'https://overpass-api.de/api/interpreter',
  'https://overpass.kumi.systems/api/interpreter',
  'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
];
export const PX_PER_CELL = 4;

async function getJSON(url, opts = {}, timeoutMs = 60000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { ...opts, signal: ctl.signal });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return await r.json();
  } finally { clearTimeout(t); }
}

// ---------- geocoding ----------
export async function geocode(q) {
  const j = await getJSON(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(q)}&count=6&language=en`);
  return (j.results || []).map(r => ({
    name: r.name, admin: r.admin1 || '', country: r.country || '', lat: r.latitude, lon: r.longitude,
    pop: r.population || 0,
  }));
}

// ---------- grid ----------
export function makeGrid(lat, lon, halfKm = 6, cellKm = 0.5) {
  const n = Math.round((2 * halfKm) / cellKm);
  const dLat = cellKm / 110.574, dLon = cellKm / (111.32 * Math.cos((lat * Math.PI) / 180));
  const north = lat + (n / 2) * dLat, west = lon - (n / 2) * dLon;
  const bbox = [west, north - n * dLat, west + n * dLon, north]; // w s e n
  const cells = [];
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    cells.push({
      i: r * n + c, r, c,
      lat: north - (r + 0.5) * dLat, lon: west + (c + 0.5) * dLon,
      bounds: [[north - (r + 1) * dLat, west + c * dLon], [north - r * dLat, west + (c + 1) * dLon]],
    });
  }
  const neighbours = cells.map(({ r, c }) => {
    const out = [];
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      const rr = r + dr, cc = c + dc;
      if (rr >= 0 && rr < n && cc >= 0 && cc < n) out.push(rr * n + cc);
    }
    return out;
  });
  return { n, cellKm, dLat, dLon, bbox, cells, neighbours, lat, lon };
}

// ---------- .npy parsing ----------
function parseNpy(buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x93) throw new Error('not an npy file');
  const major = u8[6];
  const dv = new DataView(buf);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(u8.slice(hstart, hstart + hlen));
  const descr = header.match(/'descr':\s*'([^']+)'/)[1];
  const shape = header.match(/'shape':\s*\(([^)]*)\)/)[1].split(',').map(s => s.trim()).filter(Boolean).map(Number);
  const off = hstart + hlen;
  const T = { '<u2': Uint16Array, '<f4': Float32Array, '<f8': Float64Array, '|u1': Uint8Array, '<i2': Int16Array }[descr];
  if (!T) throw new Error('unsupported dtype ' + descr);
  const data = new T(buf.slice(off));
  return { shape, data };
}

// ---------- Landsat ----------
export async function findScenes(grid, { maxScenes = 8, years = 3 } = {}) {
  const end = new Date(Date.now() - 7 * 864e5), start = new Date(end.getTime() - years * 365 * 864e5);
  const body = {
    collections: ['landsat-c2-l2'], bbox: grid.bbox,
    datetime: `${start.toISOString()}/${end.toISOString()}`,
    query: { 'eo:cloud_cover': { lt: 20 }, platform: { in: ['landsat-8', 'landsat-9'] } },
    sortby: [{ field: 'eo:cloud_cover', direction: 'asc' }], limit: 150,
  };
  const j = await getJSON(`${PC}/stac/v1/search`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  });
  const [w, s, e, n] = grid.bbox;
  // Landsat footprints are tilted: test the real polygon, not its bounding box.
  const inRing = (ring, x, y) => {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  };
  const rings = f => f.geometry.type === 'Polygon' ? [f.geometry.coordinates[0]] : f.geometry.coordinates.map(p => p[0]);
  const pts = [[w, s], [w, n], [e, s], [e, n], [(w + e) / 2, (s + n) / 2]];
  const covers = f => pts.every(([x, y]) => rings(f).some(r => inRing(r, x, y)));
  const seen = new Set(), out = [];
  for (const f of j.features) {
    const d = f.properties.datetime.slice(0, 10);
    if (seen.has(d) || !covers(f)) continue;
    seen.add(d); out.push({ id: f.id, datetime: f.properties.datetime, cloud: f.properties['eo:cloud_cover'], platform: f.properties.platform });
  }
  // keep the clearest scenes, but spread them through the year so the model sees different weather
  out.sort((a, b) => a.cloud - b.cloud);
  const picked = [], months = new Map();
  for (const sc of out) {
    const m = sc.datetime.slice(5, 7);
    if ((months.get(m) || 0) >= 1 && out.length > maxScenes * 2) continue;
    months.set(m, (months.get(m) || 0) + 1);
    picked.push(sc);
    if (picked.length >= maxScenes) break;
  }
  for (const sc of out) { if (picked.length >= maxScenes) break; if (!picked.includes(sc)) picked.push(sc); }
  return picked.sort((a, b) => a.datetime.localeCompare(b.datetime));
}

const ASSETS = ['lwir11', 'red', 'nir08', 'swir16', 'qa_pixel'];

export async function loadScene(grid, scene) {
  const W = grid.n * PX_PER_CELL;
  const [w, s, e, n] = grid.bbox;
  const q = `collection=landsat-c2-l2&item=${scene.id}&` + ASSETS.map(a => `assets=${a}`).join('&') + '&asset_as_band=true';
  const url = `${PC}/data/v1/item/bbox/${w},${s},${e},${n}/${W}x${W}.npy?${q}`;
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 90000);
  let buf;
  try {
    const r = await fetch(url, { signal: ctl.signal });
    if (!r.ok) throw new Error(`Planetary Computer ${r.status}`);
    buf = await r.arrayBuffer();
  } finally { clearTimeout(t); }
  const { shape, data } = parseNpy(buf);
  const [B, H, Wd] = shape;
  const band = k => data.subarray(k * H * Wd, (k + 1) * H * Wd);
  const [lw, red, nir, sw, qa] = [0, 1, 2, 3, 4].map(band);
  const mask = B > ASSETS.length ? band(B - 1) : null;
  const refl = v => v * 0.0000275 - 0.2;
  const per = grid.cells.map(() => ({ lst: 0, ndvi: 0, ndbi: 0, water: 0, k: 0 }));
  for (let y = 0; y < H; y++) for (let x = 0; x < Wd; x++) {
    const p = y * Wd + x;
    if (mask && mask[p] === 0) continue;
    const q_ = qa[p];
    if (!lw[p] || (q_ & 0b11011) !== 0) continue; // fill, dilated cloud, cloud, shadow
    const cell = Math.floor(y / PX_PER_CELL) * grid.n + Math.floor(x / PX_PER_CELL);
    const R = refl(red[p]), N = refl(nir[p]), S = refl(sw[p]);
    const o = per[cell];
    o.lst += lw[p] * 0.00341802 + 149 - 273.15;
    o.ndvi += (N - R) / (N + R || 1e-6);
    o.ndbi += (S - N) / (S + N || 1e-6);
    o.water += (q_ >> 7) & 1;
    o.k++;
  }
  const minPx = (PX_PER_CELL * PX_PER_CELL) / 2;
  const cells = per.map(o => o.k >= minPx
    ? { lst: o.lst / o.k, ndvi: o.ndvi / o.k, ndbi: o.ndbi / o.k, water: o.water / o.k }
    : null);
  const valid = cells.filter(Boolean).length / cells.length;
  return { ...scene, cells, valid };
}

// ---------- OpenStreetMap ----------
async function overpass(query) {
  let lastErr;
  for (const ep of OVERPASS) {
    try {
      return await getJSON(ep, { method: 'POST', body: 'data=' + encodeURIComponent(query),
        headers: { 'content-type': 'application/x-www-form-urlencoded' } }, 100000);
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

const hav = (a, b) => {
  const R = 6371, toR = Math.PI / 180;
  const dLa = (b.lat - a.lat) * toR, dLo = (b.lon - a.lon) * toR;
  const h = Math.sin(dLa / 2) ** 2 + Math.cos(a.lat * toR) * Math.cos(b.lat * toR) * Math.sin(dLo / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
};

function cellOf(grid, lat, lon) {
  const [w, , , n] = grid.bbox;
  const r = Math.floor((n - lat) / grid.dLat), c = Math.floor((lon - w) / grid.dLon);
  return r >= 0 && r < grid.n && c >= 0 && c < grid.n ? r * grid.n + c : -1;
}

function distanceTransform(grid, seeds) {
  const N = grid.n, INF = 1e9, d = new Float64Array(N * N).fill(INF);
  seeds.forEach((v, i) => { if (v) d[i] = 0; });
  if (!seeds.some(Boolean)) return Array.from(d, () => 6);
  for (let pass = 0; pass < 2; pass++) {
    const order = pass === 0 ? [...Array(N * N).keys()] : [...Array(N * N).keys()].reverse();
    for (const i of order) {
      const r = Math.floor(i / N), c = i % N;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const rr = r + dr, cc = c + dc;
        if ((dr || dc) && rr >= 0 && rr < N && cc >= 0 && cc < N) {
          const v = d[rr * N + cc] + Math.hypot(dr, dc);
          if (v < d[i]) d[i] = v;
        }
      }
    }
  }
  return Array.from(d, v => Math.min(6, v * grid.cellKm));
}

export async function loadOSM(grid) {
  const [w, s, e, n] = grid.bbox;
  const bb = `(${s},${w},${n},${e})`;
  const q = `[out:json][timeout:90];
(way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street)$"]${bb};)->.roads;
(way["natural"="water"]${bb};way["waterway"~"^(river|canal|stream)$"]${bb};way["landuse"="reservoir"]${bb};)->.blue;
(way["leisure"~"^(park|garden|nature_reserve)$"]${bb};way["landuse"~"^(forest|grass|recreation_ground|meadow|orchard)$"]${bb};way["natural"~"^(wood|scrub)$"]${bb};)->.green;
.roads out tags geom qt; .blue out tags geom qt; .green out tags geom qt;`;
  const j = await overpass(q);
  const N = grid.cells.length;
  const road = new Float64Array(N), blue = new Array(N).fill(false), green = new Array(N).fill(false);
  let nRoads = 0, nBlue = 0, nGreen = 0;
  for (const el of j.elements) {
    if (!el.geometry) continue;
    const t = el.tags || {};
    const kind = t.highway ? 'road' : (t.natural === 'water' || t.waterway || t.landuse === 'reservoir') ? 'blue' : 'green';
    kind === 'road' ? nRoads++ : kind === 'blue' ? nBlue++ : nGreen++;
    const g = el.geometry;
    for (let k = 1; k < g.length; k++) {
      const a = g[k - 1], b = g[k];
      const L = hav(a, b), steps = Math.max(1, Math.ceil(L / 0.1));
      for (let s_ = 0; s_ < steps; s_++) {
        const f = (s_ + 0.5) / steps;
        const ci = cellOf(grid, a.lat + (b.lat - a.lat) * f, a.lon + (b.lon - a.lon) * f);
        if (ci < 0) continue;
        if (kind === 'road') road[ci] += L / steps;
        else if (kind === 'blue') blue[ci] = true;
        else green[ci] = true;
      }
    }
  }
  const area = grid.cellKm * grid.cellKm;
  return {
    roadDensity: Array.from(road, v => v / area),
    distWater: distanceTransform(grid, blue),
    distGreen: distanceTransform(grid, green),
    counts: { roads: nRoads, water: nBlue, green: nGreen },
  };
}

// ---------- elevation ----------
export async function loadElevation(grid) {
  const out = [];
  for (let k = 0; k < grid.cells.length; k += 100) {
    const chunk = grid.cells.slice(k, k + 100);
    const j = await getJSON(`https://api.open-meteo.com/v1/elevation?latitude=${chunk.map(c => c.lat.toFixed(5)).join(',')}&longitude=${chunk.map(c => c.lon.toFixed(5)).join(',')}`);
    out.push(...j.elevation);
  }
  const med = [...out].sort((a, b) => a - b)[Math.floor(out.length / 2)];
  return out.map(v => v - med);
}

// ---------- weather ----------
export async function loadSceneWeather(grid, scenes) {
  const dates = scenes.map(s => s.datetime.slice(0, 10)).sort();
  const url = `https://archive-api.open-meteo.com/v1/archive?latitude=${grid.lat}&longitude=${grid.lon}` +
    `&start_date=${dates[0]}&end_date=${dates[dates.length - 1]}` +
    `&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=GMT`;
  const j = await getJSON(url, {}, 90000);
  const idx = new Map(j.hourly.time.map((t, i) => [t.slice(0, 13), i]));
  return scenes.map(s => {
    const d = new Date(s.datetime); d.setUTCMinutes(d.getUTCMinutes() + 30);
    const key = d.toISOString().slice(0, 13);
    const i = idx.get(key);
    if (i === undefined || j.hourly.temperature_2m[i] == null) return null;
    return { air: j.hourly.temperature_2m[i], rh: j.hourly.relative_humidity_2m[i], wind: j.hourly.wind_speed_10m[i] };
  });
}

export async function loadForecast(grid, overpassUtcHour) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${grid.lat}&longitude=${grid.lon}` +
    `&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m` +
    `&daily=apparent_temperature_max,temperature_2m_max,temperature_2m_min&forecast_days=3&timezone=auto`;
  const j = await getJSON(url);
  const off = j.utc_offset_seconds / 3600;
  const localHour = Math.round((((overpassUtcHour + off) % 24) + 24) % 24);
  return {
    tz: j.timezone, offsetH: off, localHour,
    days: j.daily.time.map((date, d) => {
      const hi = j.hourly.time.findIndex(t => t === `${date}T${String(localHour).padStart(2, '0')}:00`);
      return {
        date, appMax: j.daily.apparent_temperature_max[d], tMax: j.daily.temperature_2m_max[d], tMin: j.daily.temperature_2m_min[d],
        air: j.hourly.temperature_2m[hi], rh: j.hourly.relative_humidity_2m[hi], wind: j.hourly.wind_speed_10m[hi],
      };
    }),
  };
}
