# LooMap — street-level heat risk for any city

**HeatShield AI · Challenge 02.** Type any city. LooMap downloads real satellite, map and weather data, **trains a heat model in your browser**, and shows block-by-block (500 m) heat risk for the next 3 days, with the reasons behind every hotspot, a calibrated confidence band, and what-if scenarios.

**Live demo: https://loomap-xi.vercel.app** (try `https://loomap-xi.vercel.app/?city=Jaipur`)

No backend, no API keys, no paid services: it is a static site that runs on Vercel's free tier.

## What it does

| Step | How |
|---|---|
| **Collect** | Landsat 8/9 Collection-2 Level-2 thermal + optical bands (USGS, via Microsoft Planetary Computer), clearest scenes of the last 3 years, cloud/shadow-masked with the QA band. OpenStreetMap roads, water and green space (Overpass API). Copernicus DEM elevation, ERA5 weather at each overpass, and 3-day forecasts (Open-Meteo). |
| **Engineer** | Per 500 m block: NDVI, NDBI, open-water share, main-road density, distance to water, distance to green space, relative elevation, air temperature, humidity, wind. |
| **Learn** | Two parts. (1) A **city-level model** predicts the day's mean surface temperature from air temperature, plus seasonal sun angle when that scores better leave-one-date-out. (2) **Gradient-boosted regression trees** written from scratch in JavaScript (`src/ml.js`) learn each block's deviation from that level: histogram splits, row subsampling, **monotone constraints** (greener or wetter never hotter, more built-up never cooler), and quantile (P10/P90) models. |
| **Prove** | Spatial 4-fold cross-validation (whole neighbourhoods held out) against three baselines: same-day city average, linear (ridge) model and greenery-only model. Reports MAE, R², hotspot F1 (top-10% blocks), split-conformal calibrated interval coverage, and a monotonicity audit. |
| **Forecast** | Latest surface state per block + forecast weather at overpass time → predicted surface temperature per block per day. Feels-like risk = city apparent-temperature max + 0.3 × block surface anomaly, banded with US NWS heat-index categories. |
| **Explain & act** | Per-block driver breakdown (tree-path attribution; contributions sum exactly to the prediction), Getis-Ord Gi* hotspots, what-if sliders (more trees / new development) that re-run the model. |

## Run locally

Any static server works:

```bash
npx serve .
# or
python -m http.server 8000
```

Then open `http://localhost:3000` (or `:8000`). Try `?city=Jaipur` as a deep link.

## Deploy

Import this repo in Vercel. Framework preset: **Other**, no build command, output directory `.` (the root).

## Files

- `index.html`, `styles.css`: the UI
- `src/app.js`: map, panels, what-if
- `src/data.js`: live data pipeline (Planetary Computer, Overpass, Open-Meteo)
- `src/pipeline.js`: dataset building, cross-validation, calibration, forecast and risk
- `src/ml.js`: GBM, quantile loss, monotone constraints, attributions, ridge, metrics, Gi*

## Results (spatial cross-validation, October 2026 runs)

| City | LooMap MAE | Linear | City average | Hotspot F1 (LooMap vs average) |
|---|---|---|---|---|
| Jaipur | 0.66 °C | 0.86 °C | 1.00 °C | 0.49 vs 0.14 |
| Phoenix | 0.76 °C | 0.89 °C | 1.16 °C | 0.39 vs 0.04 |
| Cairo | 0.90 °C | n/a | 1.40 °C | n/a |

The calibrated 80% confidence band covers 80% of held-out observations, and the monotonicity audit finds 0 violations.

## Demo tip

OpenStreetMap's public servers can be slow or rate-limited. Open each demo city once before presenting: its data is then cached in the browser, and it reloads in a few seconds.

## Limits

A research prototype, not a heat alert. Landsat measures *surface* temperature at about 10:30 local time, not air temperature or night-time heat. OpenStreetMap is less complete in informal settlements. What-if scenarios change satellite indices, not real materials. The surface-to-air coupling factor (0.3) is an assumption.

## Data and licences

USGS Landsat (public domain) · Microsoft Planetary Computer · © OpenStreetMap contributors (ODbL) · Open-Meteo (CC BY 4.0; ERA5 © ECMWF/Copernicus, Copernicus DEM) · CARTO basemap.
