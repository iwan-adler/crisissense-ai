# LooMap: street-by-street heat risk forecasting

HeatShield challenge. Forecasts heat risk for every ~500 m block of a city over the next 72 hours,
with the top drivers (SHAP) and a P10–P90 confidence band per block, plus what-if scenarios (more trees, cool roofs).

## Planned data (open, free)

| Source | Use | Resolution |
|---|---|---|
| Landsat 8/9 thermal (USGS) | Land-surface temperature (training target) | 30 m |
| Sentinel-2 (Copernicus) | NDVI (vegetation), NDBI (built-up) | 10–20 m |
| ERA5-Land / Open-Meteo | Air temperature, humidity, wind, 72 h forecasts | Hourly |
| OpenStreetMap | Buildings, roads, parks, water | Vector |
| SRTM (NASA) | Elevation, slope | 30 m |
| Census ward data (optional) | Population exposure | Ward |

## Stack

Python · Google Earth Engine · GeoPandas · rasterio · LightGBM · SHAP · Streamlit + Folium

## Build plan

1. **Data and baseline:** pilot city, Earth Engine export, 500 m grid, features, baseline model
2. **Model and map:** LightGBM + forecast, SHAP, quantiles, hotspot layer in Streamlit
3. **Scenarios and proof:** what-if sliders, evaluation report, demo
