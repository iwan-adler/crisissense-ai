"""Preprocess Sen1Floods11 Sentinel-1 chips into model-ready arrays.

Steps (all statistics come from the TRAIN split only, then are applied to every split):
  1. No data: a pixel is invalid if VV or VH is NaN/inf. Invalid pixels get label -1 (ignored in
     loss/metrics) and, after normalisation, value 0 (= the train mean).
  2. Clipping: each band is clipped to its train 0.1 / 99.9 percentiles (removes extreme dB outliers).
  3. Normalisation: z-score per band, using mean/std of valid, clipped train pixels.

Outputs (in --out, default data/processed/sen1floods11_s1):
  <split>_x.npy    float32 (N, 2, 512, 512)  bands: VV, VH (normalised)
  <split>_y.npy    int8    (N, 512, 512)     -1 ignore, 0 not water, 1 water
  <split>_ids.txt  chip IDs in array order
  stats.json       clip limits, mean/std, nodata counts, train class frequencies

Usage (PowerShell, from the project root):
    .\\.venv\\Scripts\\python.exe -I scripts\\preprocess_sen1floods11.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT / "data" / "sen1floods11_v1.1"
DEFAULT_OUT = PROJECT / "data" / "processed" / "sen1floods11_s1"

SPLITS = ["train", "valid", "test", "bolivia"]
BANDS = ["VV", "VH"]
CLIP_PERCENTILES = (0.1, 99.9)
HIST_EDGES = np.arange(-100.0, 50.0 + 0.01, 0.01)  # 0.01 dB bins for exact-enough percentiles


def read_split(root: Path, split: str) -> list[str]:
    path = root / "splits" / f"flood_{split}_data.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_chip(data_dir: Path, chip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return S1 (2, H, W) float32, label (H, W) int8 with nodata set to -1, and the valid-pixel mask."""
    with rasterio.open(data_dir / "S1GRDHand" / f"{chip_id}_S1Hand.tif") as src:
        s1 = src.read().astype(np.float32)
    with rasterio.open(data_dir / "LabelHand" / f"{chip_id}_LabelHand.tif") as src:
        label = src.read(1).astype(np.int8)
    if s1.shape[1:] != label.shape:
        raise ValueError(f"{chip_id}: S1 shape {s1.shape} does not match label shape {label.shape}")
    valid = np.isfinite(s1).all(axis=0)
    label[~valid] = -1
    return s1, label, valid


def percentile_from_hist(counts: np.ndarray, q: float) -> float:
    cdf = np.cumsum(counts) / counts.sum()
    return float(HIST_EDGES[np.searchsorted(cdf, q / 100.0) + 1])


def compute_train_stats(data_dir: Path, train_ids: list[str]) -> dict:
    # Pass 1: clip limits from a histogram of valid train pixels.
    hists = np.zeros((len(BANDS), len(HIST_EDGES) - 1), dtype=np.int64)
    for chip_id in train_ids:
        s1, _, valid = read_chip(data_dir, chip_id)
        for b in range(len(BANDS)):
            hists[b] += np.histogram(s1[b][valid], bins=HIST_EDGES)[0]
    clip = [[percentile_from_hist(hists[b], q) for q in CLIP_PERCENTILES] for b in range(len(BANDS))]

    # Pass 2: mean/std of clipped valid train pixels, plus label frequencies.
    total = np.zeros(len(BANDS)); total_sq = np.zeros(len(BANDS)); n = 0
    label_counts = {-1: 0, 0: 0, 1: 0}
    for chip_id in train_ids:
        s1, label, valid = read_chip(data_dir, chip_id)
        for b in range(len(BANDS)):
            v = np.clip(s1[b][valid], *clip[b]).astype(np.float64)
            total[b] += v.sum()
            total_sq[b] += np.square(v).sum()
        n += int(valid.sum())
        values, counts = np.unique(label, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            label_counts[value] += count
    mean = total / n
    std = np.sqrt(total_sq / n - mean**2)

    labelled = label_counts[0] + label_counts[1]
    return {
        "bands": BANDS,
        "computed_on": "train",
        "clip_percentiles": list(CLIP_PERCENTILES),
        "clip_db": {band: clip[b] for b, band in enumerate(BANDS)},
        "mean_db": {band: float(mean[b]) for b, band in enumerate(BANDS)},
        "std_db": {band: float(std[b]) for b, band in enumerate(BANDS)},
        "train_valid_pixels": n,
        "train_label_pixels": {str(k): v for k, v in label_counts.items()},
        "train_water_fraction_of_labelled": label_counts[1] / labelled,
    }


def write_split(data_dir: Path, out_dir: Path, split: str, chip_ids: list[str], stats: dict) -> dict:
    clip = np.array([stats["clip_db"][b] for b in BANDS], dtype=np.float32)
    mean = np.array([stats["mean_db"][b] for b in BANDS], dtype=np.float32)[:, None, None]
    std = np.array([stats["std_db"][b] for b in BANDS], dtype=np.float32)[:, None, None]

    _, first_label, _ = read_chip(data_dir, chip_ids[0])
    h, w = first_label.shape
    x_out = np.lib.format.open_memmap(out_dir / f"{split}_x.npy", mode="w+", dtype=np.float32,
                                      shape=(len(chip_ids), len(BANDS), h, w))
    y_out = np.lib.format.open_memmap(out_dir / f"{split}_y.npy", mode="w+", dtype=np.int8,
                                      shape=(len(chip_ids), h, w))
    invalid_pixels = relabelled = 0
    for i, chip_id in enumerate(chip_ids):
        s1, label, valid = read_chip(data_dir, chip_id)
        with rasterio.open(data_dir / "LabelHand" / f"{chip_id}_LabelHand.tif") as src:
            relabelled += int(((src.read(1) != -1) & ~valid).sum())
        x = np.clip(np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0),
                    clip[:, 0, None, None], clip[:, 1, None, None])
        x = (x - mean) / std
        x[:, ~valid] = 0.0
        x_out[i] = x
        y_out[i] = label
        invalid_pixels += int((~valid).sum())
    x_out.flush(); y_out.flush()
    del x_out, y_out
    (out_dir / f"{split}_ids.txt").write_text("\n".join(chip_ids) + "\n")
    return {"chips": len(chip_ids), "s1_nodata_pixels": invalid_pixels, "labels_set_to_ignore": relabelled}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="extracted sen1floods11_v1.1 folder")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder for .npy arrays")
    args = parser.parse_args()

    root, out_dir = args.root.resolve(), args.out.resolve()
    data_dir = root / "data"
    if not data_dir.is_dir():
        raise SystemExit(f"Not found: {data_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; choose another --out or remove it yourself first")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {split: read_split(root, split) for split in SPLITS}
    overlap = set(splits["train"]) & (set(splits["valid"]) | set(splits["test"]) | set(splits["bolivia"]))
    if overlap:
        raise SystemExit(f"Train overlaps other splits: {sorted(overlap)[:5]}")

    print("Computing clip limits and normalisation stats on the train split...")
    stats = compute_train_stats(data_dir, splits["train"])
    for band in BANDS:
        lo, hi = stats["clip_db"][band]
        print(f"  {band}: clip [{lo:.2f}, {hi:.2f}] dB, mean {stats['mean_db'][band]:.3f}, std {stats['std_db'][band]:.3f}")

    stats["splits"] = {}
    for split, chip_ids in splits.items():
        stats["splits"][split] = write_split(data_dir, out_dir, split, chip_ids, stats)
        print(f"  {split:<8} {stats['splits'][split]}")

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Done. Arrays and stats.json in {out_dir}")


if __name__ == "__main__":
    main()
