# Heat module: street-by-street heat risk forecasting

Part of **CrisisSense AI**. Based on the LooMap (HeatShield) pitch.

Forecasts heat risk for every ~500 m block of a city over the next 72 hours, with the top drivers (SHAP)
and a P10–P90 confidence band per block, plus what-if scenarios (more trees, cool roofs, new construction).

## Planned data (open, free)

| Source | Use | Resolution |
|---|---|---|
| Landsat 8/9 thermal (USGS) | Land-surface temperature (training target) | 30 m |
| Sentinel-2 (Copernicus) | NDVI (vegetation), NDBI (built-up) | 10–20 m |
| ERA5-Land / Open-Meteo | Air temperature, humidity, wind, 72 h forecasts | Hourly |
| OpenStreetMap | Buildings, roads, parks, water | Vector |
| SRTM (NASA) | Elevation, slope | 30 m |
| Census ward data (optional) | Population exposure | Ward |

## Model

- **Where:** LightGBM predicts surface temperature per block; spatial block cross-validation
- **When:** 72-hour forecast blending each block's heat signature with weather forecasts (LSTM as comparison)
- **Why:** SHAP ranks each block's drivers
- **How sure:** quantile models for P10–P90; Getis-Ord Gi* flags significant hotspots
- **Guardrail:** monotonic constraints, so adding trees can never raise predicted risk

## Build plan

1. **Data and baseline:** pilot city, Earth Engine export, 500 m grid, features, baseline model
2. **Model and map:** LightGBM + forecast, SHAP, quantiles, hotspot layer in the dashboard
3. **Scenarios and proof:** what-if sliders, evaluation report, demo
