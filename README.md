# CrisisSense AI

**Block-level climate hazard intelligence from open satellite data, with the reasons behind every hotspot.**

CrisisSense AI maps and forecasts local climate risk for cities. City-wide averages hide the blocks that need help
first. CrisisSense works at the block level and explains each result, so planners and responders can act where it matters.

## Modules

| Module | Hazard | Question it answers | Status |
|---|---|---|---|
| [Flood](modules/flood/README.md) | Floods | Which areas are flooded right now? (Sentinel-1 radar segmentation) | Data ready |
| [Heat](modules/heat/README.md) | Urban heat | Which blocks will be dangerously hot in the next 72 h, and why? | Planning |

Both modules share one pipeline and one dashboard:

1. **Collect:** open satellite, weather and map data for any city boundary
2. **Engineer:** per-block features (built-up share, vegetation, roads, water distance, elevation)
3. **Learn:** ML models per hazard (segmentation for floods, LightGBM for heat)
4. **Explain:** drivers (SHAP) and an uncertainty band on every prediction
5. **Act:** hazard maps, risk levels and what-if scenarios in a Streamlit + Folium dashboard

## Layout

```
modules/
  flood/          # flood mapping (Sen1Floods11, Sentinel-1)
  heat/           # heat risk forecasting (Landsat LST, Sentinel-2, ERA5, OSM)
scripts/
  inspect_sen1floods11.py   # flood dataset sanity report
data/             # local datasets, NOT committed (see data/README.md)
```

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install rasterio numpy
# download the flood data: see data/README.md
.\.venv\Scripts\python.exe -I scripts\inspect_sen1floods11.py
```

## Responsible use

A research prototype, not an official warning system. It uses only open, permitted datasets with no personal data;
population is used at ward level only. Every prediction carries an uncertainty band.

## Data policy

Open, free datasets only. Raw data is never committed: `.gitignore` excludes `data/` except its README,
and each dataset's source, download command and checksum are documented in `data/README.md`.
