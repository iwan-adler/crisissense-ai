# CrisisSense AI

**Heat risk, block by block, for any city, with the reasons behind every hotspot.**

HeatShield AI · Challenge 02

**Live demo (LooMap):** https://loomap-xi.vercel.app/ · **Code:** [modules/heat/app](modules/heat/app/README.md)

City-wide heat forecasts hide the blocks that need cooling shelters, water points and new trees first.
Pick a city and LooMap downloads Landsat thermal imagery, OpenStreetMap and weather data, trains a gradient-boosted
model **in the browser**, and forecasts heat risk for every 500 m block over the next 3 days, with the reasons and a
confidence band for each block.

## Using the demo

1. Enter a city and choose an area: 8×8, 12×12 or 16×16 km.
2. Click **Analyse**. The pipeline log shows progress while data downloads and the model trains.
3. Explore the 3-day forecast, the map layers (surface heat, risk, hotspots, uncertainty, latest satellite observation),
   the spatial cross-validation scores and the drivers of heat in each block.
4. Optionally download the dataset as CSV.

## How it works

1. **Collect:** Landsat 8/9 Collection 2 scenes from the last 3 years via Microsoft Planetary Computer; clouds and shadows removed with the QA band.
2. **Engineer:** per-block vegetation, built-up and water shares, road density, distance to water and green space, relative elevation (Copernicus DEM), and ERA5 air temperature, humidity and wind at satellite overpass.
3. **Learn:** a city-level model predicts mean surface temperature from air temperature; a gradient-boosted tree model learns each block's deviation, with monotone constraints so more vegetation or water never raises predicted heat.
4. **Prove:** spatial 4-fold cross-validation holds out whole neighbourhoods and compares against city-average, linear and greenery-only baselines. P10–P90 bands are split-conformal calibrated to cover ~80% of held-out observations.
5. **Forecast:** each block's surface state plus the Open-Meteo 3-day forecast gives a feels-like risk, banded with US NWS heat-index categories.
6. **Explain and act:** per-block drivers (tree-path attribution), significant hotspots (Getis-Ord Gi*), and what-if scenarios with more trees or new development.

## Results

Held-out MAE in spatial cross-validation: **Jaipur 0.66 °C** (city-average baseline 1.00), **Phoenix 0.76 °C** (1.16),
**Cairo 0.90 °C** (1.40). Details: [modules/heat/app/README.md](modules/heat/app/README.md#results).

More on design choices and what changed from the original plan: [modules/heat/README.md](modules/heat/README.md)

## Run locally

Requires Node.js 18+.

```bash
cd modules/heat/app
npm start      # http://localhost:3000
npm test       # 14 tests, no network needed
```

## Layout

```
modules/
  heat/
    README.md     # design choices, data, results, next steps
    app/          # LooMap web app: runs in the browser, deployed on Vercel
```

## Limits and responsible use

- A **research prototype**, not a heat alert or health warning.
- Landsat measures **surface** temperature at about 10:30 local time, not the air temperature people feel, and not night-time heat.
- OpenStreetMap is weaker in informal settlements, which are often the most heat-vulnerable.
- What-if scenarios change satellite indices, not real materials, so results are directional only.
- **No personal data.** Nothing leaves the browser except requests to public data APIs, and no data is stored in the repo.

## Data attribution

USGS Landsat 8/9 Collection 2 Level 2 · Microsoft Planetary Computer · © OpenStreetMap contributors ·
Open-Meteo (ERA5, ECMWF / DWD / NOAA forecasts, Copernicus DEM)
