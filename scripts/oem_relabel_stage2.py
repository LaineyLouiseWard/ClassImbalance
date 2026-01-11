#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import rasterio


# OEM 8 classes → your 6-class taxonomy
# 0 bareland -> background(0)
# 1 rangeland -> grassland(2)
# 2 developed space -> settlement(4)
# 3 road -> settlement(4)
# 4 tree -> forest(1)
# 5 water -> background(0)
# 6 agriculture -> cropland(3)
# 7 building -> settlement(4)
OEM_ID_TO_TARGET6: Dict[int, int] = {
    0: 0,
    1: 2,
    2: 4,
    3: 4,
    4: 1,
    5: 0,
    6: 3,
    7: 4,
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def read_oem_mask(mask_tif: Path) -> np.ndarray:
    with rasterio.open(mask_tif) as src:
        if src.count != 1:
            raise ValueError(f"Expected 1-band mask, got {src.count}: {mask_tif}")
        arr = src.read(1)
    return arr.astype(np.int32)


def remap_to_target6(oem_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(oem_ids, dtype=np.uint8)
    for k, v in OEM_ID_TO_TARGET6.items():
        out[oem_ids == k] = v
    out[(oem_ids < 0) | (oem_ids > 7)] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-root",
        default="data/openearthmap_filtered",
        help="Filtered OEM dataset (images/ + masks/)",
    )
    ap.add_argument(
        "--out-root",
        default="data/openearthmap_relabelled",
        help="Output directory for remapped OEM",
    )
    ap.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to populate images",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    in_images = in_root / "images"
    in_masks = in_root / "masks"
    out_images = out_root / "images"
    out_masks = out_root / "masks"

    ensure_dir(out_images)
    ensure_dir(out_masks)

    mask_files = sorted(in_masks.glob("*.tif"))
    if not mask_files:
        raise FileNotFoundError(f"No OEM masks found in {in_masks}")

    written = 0
    skipped = 0

    for msk in mask_files:
        stem = msk.stem
        img = in_images / f"{stem}.tif"
        if not img.exists():
            continue

        out_img = out_images / f"{stem}.tif"
        out_msk = out_masks / f"{stem}.png"

        if out_msk.exists() and not args.overwrite:
            skipped += 1
            continue

        link_or_copy(img, out_img, args.mode)

        oem_ids = read_oem_mask(msk)
        tgt = remap_to_target6(oem_ids)
        Image.fromarray(tgt, mode="L").save(out_msk)
        written += 1

    print("[oem_remap_labels]")
    print(f"  input root: {in_root}")
    print(f"  output root: {out_root.resolve()}")
    print(f"  wrote masks: {written}")
    print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()
