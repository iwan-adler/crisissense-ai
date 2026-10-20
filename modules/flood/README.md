# Flood module: flood mapping from Sentinel-1 radar

Part of **CrisisSense AI**.

Segments water and flooded pixels in Sentinel-1 SAR imagery, which works through clouds and at night.
The resulting flood extent maps feed impact estimates (exposed blocks, population, roads) in the shared dashboard.

## Data

Sen1Floods11 v1.1, hand-labeled subset: 446 chips of 512×512 px with train/valid/test splits.
Source, download steps and folder layout are in [`data/README.md`](../../data/README.md).

- **Input:** `S1GRDHand/*_S1Hand.tif`, 2 bands (VV, VH), dB
- **Target:** `LabelHand/*_LabelHand.tif`: -1 no data, 0 not water, 1 water
- **Baseline to beat:** `S1OtsuLabelHand` (Otsu threshold on S1)

## Scripts (run from the project root)

```powershell
# 1. Sanity report: counts, formats, dimensions, labels, splits
.\.venv\Scripts\python.exe -I scripts\inspect_sen1floods11.py

# 2. Plot VV | VH | false colour | mask for random chips -> outputs/figures/
.\.venv\Scripts\python.exe -I scripts\visualize_sen1floods11.py --split train --n 4
.\.venv\Scripts\python.exe -I scripts\visualize_sen1floods11.py --ids Bolivia_103757

# 3. Build model-ready arrays -> data/processed/sen1floods11_s1/ (~1 GB, ~25 s)
.\.venv\Scripts\python.exe -I scripts\preprocess_sen1floods11.py
```

## Preprocessing

All statistics come from the **train split only** and are then applied to valid, test and Bolivia, so nothing leaks from evaluation data.

| Step | What | Why |
|---|---|---|
| No data | Pixels where VV or VH is NaN get label `-1` and value `0` after normalization | 23/252 train chips have NaNs (~2% of pixels); 24k of those still had a water/not-water label |
| Clip | VV to [-29.31, 3.15] dB, VH to [-46.99, -5.24] dB (train 0.1/99.9 percentiles) | Raw values reach -79…+30 dB; extreme speckle/bright targets would dominate the scaling |
| Normalize | z-score per band: VV mean -10.39 / std 4.01, VH mean -17.24 / std 4.74 | Stable inputs for the network |

Outputs in `data/processed/sen1floods11_s1/`:

| File | Shape / type | Content |
|---|---|---|
| `<split>_x.npy` | float32 (N, 2, 512, 512) | normalized VV, VH |
| `<split>_y.npy` | int8 (N, 512, 512) | -1 ignore, 0 not water, 1 water |
| `<split>_ids.txt` | text | chip IDs in array order |
| `stats.json` | JSON | clip limits, mean/std, nodata counts, train class balance |

Load with `np.load("data/processed/sen1floods11_s1/train_x.npy", mmap_mode="r")` to avoid reading everything into memory.
Water is only ~9.5% of labeled train pixels, so use a weighted or Dice loss, and always ignore `-1`.

## Model (planned)

- U-Net style segmentation on VV/VH; loss and metrics ignore `-1` pixels
- Metrics: IoU and F1 for water on the test split, plus the held-out Bolivia event
- Mapping, not forecasting: it shows what is flooded in an image, not what will flood
