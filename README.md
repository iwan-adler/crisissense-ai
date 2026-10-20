# CrisisSense AI

**Block-level climate hazard intelligence from open satellite data, with the reasons behind every hotspot.**

CrisisSense AI maps and forecasts local climate risk for cities. City-wide averages hide the blocks that need help
first. CrisisSense works at the block level and explains each result, so planners and responders can act where it matters.

**Live demo (heat module): https://loomap-xi.vercel.app**

## Modules

| Module | Hazard | Question it answers | Status |
|---|---|---|---|
| [Flood](modules/flood/README.md) | Floods | Which areas are flooded right now? (Sentinel-1 radar segmentation) | Data ready |
| [Heat (LooMap)](modules/heat/README.md) | Urban heat | Which blocks will be dangerously hot in the next 3 days, and why? | **Live:** [web app](https://loomap-xi.vercel.app) · [code](modules/heat/app/README.md) |

Both modules follow the same approach:

1. **Collect:** open satellite, weather and map data for any city
2. **Engineer:** per-block features (built-up share, vegetation, roads, water distance, elevation)
3. **Learn:** ML models per hazard (segmentation for floods; gradient-boosted trees for heat)
4. **Explain:** drivers per block and an uncertainty band on every prediction
5. **Act:** hazard maps, risk levels and what-if scenarios

## Layout

```
modules/
  flood/          # flood mapping (Sen1Floods11, Sentinel-1)
  heat/           # heat risk forecasting
    app/          # LooMap web app: runs in the browser, deployed on Vercel (see app/README.md)
scripts/
  inspect_sen1floods11.py      # flood dataset sanity report
  visualize_sen1floods11.py    # plot S1 VV/VH next to flood masks
  preprocess_sen1floods11.py   # nodata handling, clipping, train-only normalization -> .npy
data/             # local datasets, NOT committed (see data/README.md)
outputs/          # generated figures, NOT committed
```

## Setup

**Heat module (web app, Node.js 18+):**

```bash
cd modules/heat/app
npm start      # http://localhost:3000
npm test       # 14 tests, no network needed
```

**Flood module (Python, Windows PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# download the flood data: see data/README.md, then see modules/flood/README.md for the scripts
```

## Responsible use

A research prototype, not an official warning system. It uses only open, permitted datasets and no personal data.
Every prediction carries an uncertainty band, and each module documents its known blind spots.

## Data policy

Open, free datasets only. Raw data is never committed: `.gitignore` excludes `data/` except its README,
and each dataset's source, download command and checksum are documented in `data/README.md`.
The heat module downloads its data live in the browser and stores nothing in the repo.
