# Heat module: street-by-street heat risk (LooMap)

Part of **CrisisSense AI**. Based on the LooMap pitch for HeatShield AI (Challenge 02).

**Status: live** → https://loomap-xi.vercel.app · code, results and run instructions in [`app/`](app/README.md)

For any city, it forecasts heat risk for every ~500 m block over the next 3 days. Each block gets its top drivers, a calibrated confidence band, statistically significant hotspots, and what-if scenarios (more trees, new development).

## What was built vs. the original plan

| Plan | Built | Why |
|---|---|---|
| Earth Engine export for a pilot city | Live download for **any** city from Microsoft Planetary Computer, no account needed | Works for whatever city a judge types; nothing to pre-process |
| Sentinel-2 for NDVI/NDBI | Landsat 8/9 optical bands from the **same scene** as the thermal band | Vegetation and temperature are observed at the same moment |
| LightGBM | Gradient-boosted trees written from scratch in JavaScript, with monotone constraints and quantile loss | Trains in the browser in a few seconds; no server |
| SHAP | Tree-path attribution (contributions add up exactly to each prediction) | Same purpose, fast enough to run on every click |
| One model for everything | **Two parts:** a city-level model for each day's average (air temperature, plus sun angle when it helps), and a block model for each block's deviation | The block map now reflects real urban drivers, not just which day was hot |
| LSTM comparison | Not built | About 8 satellite dates per city is too few for a sequence model; linear, greenery-only and city-average baselines are compared instead |
| Streamlit + Folium dashboard | Static web app (Leaflet) deployed on Vercel | Free hosting, shareable link |
| Census ward population | Not used yet | Next step: weight risk by exposure |

## Data

| Source | Use | Resolution |
|---|---|---|
| Landsat 8/9 C2 L2 (USGS) | Land-surface temperature (training target), NDVI, NDBI, water | 30 m → 500 m blocks |
| ERA5 / Open-Meteo | Air temperature, humidity, wind at overpass; 3-day forecast | Hourly |
| OpenStreetMap (Overpass) | Main-road density, distance to water and green space | Vector |
| Copernicus DEM (Open-Meteo) | Relative elevation | ~90 m |

## Results

Held-out MAE in spatial cross-validation: **Jaipur 0.66 °C** (city-average baseline 1.00), **Phoenix 0.76 °C** (1.16), **Cairo 0.90 °C** (1.40). Details are in [`app/README.md`](app/README.md#results).

## Next steps

- Night-time heat (Landsat night scenes or MODIS night LST)
- Population exposure per block
- Cool-roof scenarios using albedo instead of indices
