#!/usr/bin/env python3
"""
Prepare OpenEarthMap (raw-by-region) into a teacher-training split.

Input structure (example):
  data/openearthmap_raw/
    aachen/
      images/
      labels/
    beijing/
      images/
      labels/
    ...

Output:
  data/openearthmap_teacher/
    train/images, train/masks
    val/images,   val/masks

We create SYMLINKS (no big copies).
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
from convert_oem_masks import quick_mask_stats

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MASK_EXTS = (".png", ".tif", ".tiff")


def find_pairs(raw_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []

    for region_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        img_dir = region_dir / "images"
        lbl_dir = region_dir / "labels"  # OEM calls these labels

        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue

        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        lbls = {p.stem: p for p in lbl_dir.iterdir() if p.suffix.lower() in MASK_EXTS}

        for img_p in imgs:
            m = lbls.get(img_p.stem, None)
            if m is not None:
                pairs.append((img_p, m))

    return pairs


def ensure_dirs(out_root: Path):
    for split in ["train", "val"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "masks").mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, required=True,
                    help="Path to raw OEM root containing region folders.")
    ap.add_argument("--out_root", type=Path, default=Path("data/openearthmap_teacher"),
                    help="Output root for teacher split (symlinks).")
    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="Validation fraction (default 0.1).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for deterministic split.")
    args = ap.parse_args()

    raw_root: Path = args.raw_root
    out_root: Path = args.out_root

    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    pairs = find_pairs(raw_root)
    if len(pairs) == 0:
        raise RuntimeError(
            f"No (image,label) pairs found under {raw_root}. "
            "Expected region/images and region/labels."
        )

    print(f"[TeacherPrep] Found {len(pairs)} pairs under: {raw_root}")

    # deterministic shuffle + split
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n_val = int(round(len(pairs) * args.val_frac))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    print(f"[TeacherPrep] Split: train={len(train_pairs)}, val={len(val_pairs)} (val_frac={args.val_frac})")

    ensure_dirs(out_root)

    # symlink
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = out_root / split_name / "images"
        msk_out = out_root / split_name / "masks"

        for img_p, mask_p in split_pairs:
            safe_symlink(img_p, img_out / img_p.name)
            safe_symlink(mask_p, msk_out / mask_p.name)

    # quick sanity stats
    val_masks = [m for _, m in val_pairs]
    quick_mask_stats(val_masks)

    print(f"[TeacherPrep] Done. Output at: {out_root}")
    print("[TeacherPrep] Example:")
    print(f"  {out_root}/train/images")
    print(f"  {out_root}/train/masks")


if __name__ == "__main__":
    main()
