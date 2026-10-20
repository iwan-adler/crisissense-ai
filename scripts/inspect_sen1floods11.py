"""Inspect the Sen1Floods11 v1.1 hand-labeled subset.

Reports file counts, formats, image dimensions/bands/dtypes/CRS per layer,
label value distribution, and train/valid/test split sizes.

Usage (PowerShell, from the project root):
    .\\.venv\\Scripts\\python.exe -I scripts\\inspect_sen1floods11.py
    .\\.venv\\Scripts\\python.exe -I scripts\\inspect_sen1floods11.py --root data\\sen1floods11_v1.1
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "sen1floods11_v1.1"
LABEL_MEANING = {-1: "no data", 0: "not water", 1: "water"}


def describe_layer(folder: Path) -> None:
    files = sorted(folder.glob("*.tif"))
    exts = Counter(f.suffix.lower() for f in folder.iterdir() if f.is_file())
    print(f"\n[{folder.name}] {len(files)} files, formats: {dict(exts)}")
    if not files:
        return

    shapes, dtypes, crss = Counter(), Counter(), Counter()
    for f in files:
        with rasterio.open(f) as src:
            shapes[(src.count, src.height, src.width)] += 1
            dtypes[src.dtypes[0]] += 1
            crss[str(src.crs)] += 1

    with rasterio.open(files[0]) as src:
        res = src.res
        descriptions = [d for d in src.descriptions if d]
        nodata = src.nodata

    print(f"  (bands, height, width): {dict(shapes)}")
    print(f"  dtype: {dict(dtypes)}  CRS: {dict(crss)}")
    print(f"  pixel size (deg): {res[0]:.6g} x {res[1]:.6g}  nodata: {nodata}")
    if descriptions:
        print(f"  band names: {descriptions}")


def describe_labels(folder: Path) -> None:
    counts = Counter()
    for f in sorted(folder.glob("*.tif")):
        with rasterio.open(f) as src:
            values, n = np.unique(src.read(1), return_counts=True)
            counts.update(dict(zip(values.tolist(), n.tolist())))
    total = sum(counts.values())
    print(f"\n[{folder.name}] pixel label distribution:")
    for value in sorted(counts):
        meaning = LABEL_MEANING.get(value, "unknown")
        print(f"  {value:>3} ({meaning:<9}): {counts[value]:>12,}  ({counts[value] / total:6.2%})")


def describe_splits(splits_dir: Path, data_dir: Path) -> None:
    print(f"\n[splits] {splits_dir}")
    label_ids = {f.name.rsplit("_", 1)[0] for f in (data_dir / "LabelHand").glob("*.tif")}
    for f in sorted(splits_dir.glob("*.txt")):
        ids = [line.strip() for line in f.read_text().splitlines() if line.strip()]
        missing = [i for i in ids if i not in label_ids]
        events = Counter(i.split("_")[0] for i in ids)
        print(f"  {f.name:<24} {len(ids):>4} chips, {len(events)} events, missing labels: {len(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="extracted sen1floods11_v1.1 folder")
    args = parser.parse_args()

    root = args.root.resolve()
    data_dir = root / "data"
    if not data_dir.is_dir():
        raise SystemExit(f"Not found: {data_dir} (extract sen1floods11_v1_1.tar.gz first)")

    print(f"Dataset root: {root}")
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        describe_layer(folder)
    describe_labels(data_dir / "LabelHand")
    describe_splits(root / "splits", data_dir)


if __name__ == "__main__":
    main()
