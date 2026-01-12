#!/usr/bin/env python3
"""
Stage 1 OEM filtering:
Remove tiles where BUILT ENVIRONMENT (building + road + developed)
occupies more than a threshold of pixels.

Runs on RAW OpenEarthMap labels (.tif).
"""

from pathlib import Path
import numpy as np
import rasterio
import argparse
import shutil
from tqdm import tqdm

BUILT_CLASSES = {2, 3, 7}  # developed, road, building
IGNORE_INDEX = 255


def built_env_percentage(label_path):
    with rasterio.open(label_path) as src:
        m = src.read(1)

    valid = m != IGNORE_INDEX
    denom = valid.sum()
    if denom == 0:
        return 0.0

    built = np.isin(m, list(BUILT_CLASSES)).sum()
    return 100.0 * built / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True,
                    help="Path to OpenEarthMap_wo_xBD")
    ap.add_argument("--out-root", required=True,
                    help="Output folder for stage-1 filtered OEM")
    ap.add_argument("--threshold", type=float, default=50.0)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    kept = dropped = 0

    for region in tqdm(sorted(raw_root.iterdir()), desc="Regions"):
        label_dir = region / "labels"
        image_dir = region / "images"

        if not label_dir.exists():
            continue

        for label_path in label_dir.glob("*.tif"):
            pct = built_env_percentage(label_path)
            if pct > args.threshold:
                dropped += 1
                continue

            img_path = image_dir / label_path.name
            if not img_path.exists():
                continue

            (out_root / "images").mkdir(parents=True, exist_ok=True)
            (out_root / "masks").mkdir(parents=True, exist_ok=True)

            stem = f"oem_{region.name}_{label_path.stem}"

            dst_label = out_root / "masks" / f"{stem}.tif"
            dst_img   = out_root / "images" / f"{stem}.tif"

            if args.mode == "symlink":
                dst_label.symlink_to(label_path)
                dst_img.symlink_to(img_path)
            else:
                shutil.copy2(label_path, dst_label)
                shutil.copy2(img_path, dst_img)

            kept += 1

    print("\n[OEM Stage 1]")
    print(f"Threshold: {args.threshold}% built environment")
    print(f"Kept: {kept}")
    print(f"Dropped: {dropped}")


if __name__ == "__main__":
    main()
