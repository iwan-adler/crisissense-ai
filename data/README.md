# Sen1Floods11 (v1.1, hand-labeled subset)

## Source

- Official project: https://github.com/cloudtostreet/Sen1Floods11 (Cloud to Street / Bonafilia et al., 2020).
  The full dataset (~14 GB, including weakly labeled chips) is in the Google Cloud Storage bucket `gs://sen1floods11`.
- **Copy used here:** Hugging Face mirror [`blumenstiel/Sen1Floods11`](https://huggingface.co/datasets/blumenstiel/Sen1Floods11)
  (revision `0aa4d22937f726bbf699f902b59026a9e3f75c52`). This is a repackaging of v1.1 with only the
  **446 hand-labeled chips**; the weakly labeled S1 data is left out. It's free, needs no account, and is enough for initial development.
- Original mirror README preserved as `README_huggingface.md`.

## Download method

```powershell
# hf CLI (huggingface_hub); downloaded into this folder
hf download blumenstiel/Sen1Floods11 sen1floods11_v1_1.tar.gz --repo-type dataset --local-dir .\data
tar -xzf .\data\sen1floods11_v1_1.tar.gz -C .\data
```

- Archive: `sen1floods11_v1_1.tar.gz`, 1,771,556,385 bytes
- SHA-256: `d836b0c9772f2bbbe1d96912a5fc8b4eb4752309ee864b320cff78fc4f640a4f` (verified against the Hugging Face LFS hash)
- Before extraction, the archive was checked for absolute or `..` paths and for links. None were found.
- `.cache/` is Hugging Face download metadata and can be ignored.

## Folder structure

```
data/sen1floods11_v1.1/
├── Sen1Floods11_Metadata.geojson   # chip footprints / event metadata
├── splits/                         # one chip ID per line, e.g. "Ghana_103272"
│   ├── flood_train_data.txt        # 252 chips
│   ├── flood_valid_data.txt        #  89 chips
│   ├── flood_test_data.txt         #  90 chips
│   └── flood_bolivia_data.txt      #  15 chips (held-out Bolivia event, OOD test)
└── data/                           # 446 GeoTIFFs each, 512x512, EPSG:4326, ~10 m
    ├── S1GRDHand/       <id>_S1Hand.tif        Sentinel-1 GRD, 2 bands (VV, VH), float32, dB   <- model input
    ├── LabelHand/       <id>_LabelHand.tif     hand-labeled mask, int16: -1 no data, 0 not water, 1 water  <- target
    ├── S2L1CHand/       <id>_S2Hand.tif        Sentinel-2 L1C, 13 bands, int16 (reflectance x 10000)
    ├── S1OtsuLabelHand/ <id>_S1OtsuLabelHand.tif  Otsu-threshold water mask from S1 (baseline)
    ├── JRCWaterHand/    <id>_JRCWaterHand.tif  JRC Global Surface Water occurrence (permanent water)
    └── CopernicusDEM/   <id>_DEM.tif           elevation, int16
```

File name = `<Event>_<ChipID>_<Layer>.tif`. A split ID maps to e.g. `S1GRDHand/<ID>_S1Hand.tif` and `LabelHand/<ID>_LabelHand.tif` (verified).
Label pixels: ~77% not water, ~9% water, ~14% no data. Mask `-1` out of the loss and metrics.

## Licensing / usage

- No license file in the GitHub repo (GitHub license API returns 404), and no license tag on the Hugging Face mirror.
  The original project has listed it as **CC BY 4.0**, but that isn't stated on either page checked here,
  so confirm it before redistributing. Either way, attribute and cite:
  > Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops, pp. 210–211.
- Contains modified Copernicus Sentinel data (ESA). Copernicus DEM and JRC Global Surface Water have their own attribution terms.
- **Do not commit this folder to Git.** The repo's `.gitignore` excludes `data/*` except this README.

## Flood mapping vs. flood forecasting

Sen1Floods11 supports **flood mapping**: segmenting water/flooded pixels in a satellite image that has *already been
captured* during or after an event (what is flooded now). It contains no time series, rainfall, river gauge,
or forecast data, so on its own it can't predict **future** flooding.
**Flood forecasting** (where and when flooding will happen) needs extra inputs such as precipitation forecasts,
hydrological/river discharge models (e.g. GloFAS), soil moisture, and terrain. Within CrisisSense AI,
the segmentation model's maps can serve as ground truth or impact layers for a forecasting component. They are not a forecast.
