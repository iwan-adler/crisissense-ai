"""Plot Sentinel-1 VV / VH next to the hand-labeled flood mask for Sen1Floods11 chips.

Each row is one chip: VV (dB) | VH (dB) | false colour (R=VV, G=VH, B=VV-VH) | flood mask.
NaN (no data) pixels in the S1 image are drawn in magenta so they are easy to spot.

Usage (PowerShell, from the project root):
    .\\.venv\\Scripts\\python.exe -I scripts\\visualize_sen1floods11.py
    .\\.venv\\Scripts\\python.exe -I scripts\\visualize_sen1floods11.py --split test --n 6 --seed 1
    .\\.venv\\Scripts\\python.exe -I scripts\\visualize_sen1floods11.py --ids Bolivia_103757 India_747992
"""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT / "data" / "sen1floods11_v1.1"
DEFAULT_OUT = PROJECT / "outputs" / "figures"

# Fixed display ranges (dB) so chips are comparable with each other.
VV_RANGE = (-25.0, 0.0)
VH_RANGE = (-32.0, -8.0)
RATIO_RANGE = (0.0, 15.0)

MASK_CMAP = ListedColormap(["#9e9e9e", "#efe6d2", "#1f6fd1"])  # -1 no data, 0 not water, 1 water
NODATA_RGB = np.array([1.0, 0.0, 1.0])


def read_chip(data_dir: Path, chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(data_dir / "S1GRDHand" / f"{chip_id}_S1Hand.tif") as src:
        s1 = src.read().astype(np.float32)
    with rasterio.open(data_dir / "LabelHand" / f"{chip_id}_LabelHand.tif") as src:
        label = src.read(1)
    return s1, label


def scale(band: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    lo, hi = value_range
    return np.clip((band - lo) / (hi - lo), 0.0, 1.0)


def grey_with_nodata(band: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    img = np.repeat(scale(band, value_range)[..., None], 3, axis=-1)
    img[~np.isfinite(band)] = NODATA_RGB
    return img


def false_colour(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    img = np.stack([scale(vv, VV_RANGE), scale(vh, VH_RANGE), scale(vv - vh, RATIO_RANGE)], axis=-1)
    img[~(np.isfinite(vv) & np.isfinite(vh))] = NODATA_RGB
    return img


def read_split(root: Path, split: str) -> list[str]:
    path = root / "splits" / f"flood_{split}_data.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def plot_chips(data_dir: Path, chip_ids: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(len(chip_ids), 4, figsize=(14, 3.6 * len(chip_ids)), squeeze=False)
    for row, chip_id in zip(axes, chip_ids):
        s1, label = read_chip(data_dir, chip_id)
        vv, vh = s1
        valid = label != -1
        water = (label == 1).sum() / max(valid.sum(), 1)
        nodata = (~np.isfinite(s1).all(axis=0)).mean()

        row[0].imshow(grey_with_nodata(vv, VV_RANGE))
        row[0].set_title(f"{chip_id}\nVV {VV_RANGE[0]:g}..{VV_RANGE[1]:g} dB", fontsize=9)
        row[1].imshow(grey_with_nodata(vh, VH_RANGE))
        row[1].set_title(f"VH {VH_RANGE[0]:g}..{VH_RANGE[1]:g} dB", fontsize=9)
        row[2].imshow(false_colour(vv, vh))
        row[2].set_title("R=VV  G=VH  B=VV-VH", fontsize=9)
        row[3].imshow(label, cmap=MASK_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        row[3].set_title(f"mask: water {water:.1%} of labelled, S1 nodata {nodata:.1%}", fontsize=9)
        for ax in row:
            ax.set_axis_off()

    fig.legend(
        handles=[
            Patch(color=MASK_CMAP.colors[0], label="no data (-1)"),
            Patch(color=MASK_CMAP.colors[1], label="not water (0)"),
            Patch(color=MASK_CMAP.colors[2], label="water (1)"),
            Patch(color=NODATA_RGB, label="S1 NaN"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="extracted sen1floods11_v1.1 folder")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test", "bolivia"])
    parser.add_argument("--n", type=int, default=4, help="number of random chips from the split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", nargs="+", help="explicit chip IDs (overrides --split/--n)")
    parser.add_argument("--out", type=Path, help="output PNG (default: outputs/figures/s1_mask_<split>.png)")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.ids:
        chip_ids, name = args.ids, "selected"
    else:
        all_ids = read_split(root, args.split)
        chip_ids = random.Random(args.seed).sample(all_ids, min(args.n, len(all_ids)))
        name = args.split
    out_path = args.out or DEFAULT_OUT / f"s1_mask_{name}.png"

    plot_chips(root / "data", chip_ids, out_path)
    print(f"Saved {len(chip_ids)} chips to {out_path}")


if __name__ == "__main__":
    main()
