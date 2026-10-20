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

Check the data:

```powershell
.\.venv\Scripts\python.exe -I scripts\inspect_sen1floods11.py
```

## Model (planned)

- U-Net style segmentation on VV/VH; loss and metrics ignore `-1` pixels
- Metrics: IoU and F1 for water on the test split, plus the held-out Bolivia event
- Mapping, not forecasting: it shows what is flooded in an image, not what will flood
