# CrisisSense AI

AI Innovation Hackathon 2026: open-data geospatial AI for climate hazards. This repo holds two projects:

| Project | Hazard | What it does | Status |
|---|---|---|---|
| **CrisisSense** | Floods | Segments flooded/water pixels in Sentinel-1 radar imagery (flood *mapping*) as the basis for impact estimates | Data ready (Sen1Floods11) |
| **[LooMap](loomap/README.md)** | Urban heat | Forecasts heat risk for every ~500 m city block over 72 h and explains the drivers of each hotspot | Planning |

## Layout

```
data/                       # local datasets, NOT committed (see data/README.md)
  sen1floods11_v1.1/        # flood: Sentinel-1 chips, hand-labeled masks, splits
scripts/
  inspect_sen1floods11.py   # dataset sanity report (counts, formats, dims, labels)
loomap/                     # heat-risk project (LooMap)
```

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install rasterio numpy
# download the flood data: see data/README.md
.\.venv\Scripts\python.exe -I scripts\inspect_sen1floods11.py
```

## Data policy

Open, free datasets only. Raw data is never committed: `.gitignore` excludes `data/` except its README,
and each dataset's source, download command and checksum are documented in `data/README.md`.
