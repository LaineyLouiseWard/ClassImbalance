#!/usr/bin/env python3

"""
Split biodiversity images and masks into train/val/test sets.

Expects:
  in-root/
    images/
    masks/

Creates:
  out-root/
    train/images, train/masks
    val/images,   val/masks
    test/images,  test/masks

Pairs images and masks by filename stem and performs a reproducible
random split. Files can be copied or symlinked.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def ensure_empty(p: Path, overwrite: bool) -> None:
    if p.exists() and overwrite:
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default="data/biodiversity_raw", help="Input pool root with images/ and masks/")
    ap.add_argument("--out-root", default="data/biodiversity_split", help="Output root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--img-ext", default=".tif")
    ap.add_argument("--mask-ext", default=".png")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if abs((args.train_frac + args.val_frac + args.test_frac) - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.0")

    in_root = Path(args.in_root)
    images_dir = in_root / "images"
    masks_dir = in_root / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Expected in-root/images and in-root/masks")

    masks = {p.stem: p for p in masks_dir.glob(f"*{args.mask_ext}")}

    stems = []
    for img in images_dir.glob(f"*{args.img_ext}"):
        if img.stem in masks:
            stems.append(img.stem)

    if not stems:
        raise RuntimeError("No matched image/mask pairs found.")

    stems = sorted(stems)
    rnd = random.Random(args.seed)
    rnd.shuffle(stems)

    n = len(stems)
    n_train = int(round(n * args.train_frac))
    n_val = int(round(n * args.val_frac))
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Bad split sizes: train={n_train}, val={n_val}, test={n_test}")

    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    out_root = Path(args.out_root)
    for s in ["train", "val", "test"]:
        ensure_empty(out_root / s / "images", args.overwrite)
        ensure_empty(out_root / s / "masks", args.overwrite)

    def write_one(stem: str, split: str) -> None:
        src_img = images_dir / f"{stem}{args.img_ext}"
        src_msk = masks[stem]
        dst_img = out_root / split / "images" / src_img.name
        dst_msk = out_root / split / "masks" / src_msk.name

        if args.mode == "copy":
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_msk, dst_msk)
        else:
            dst_img.symlink_to(Path(shutil.os.path.relpath(src_img, start=dst_img.parent)))
            dst_msk.symlink_to(Path(shutil.os.path.relpath(src_msk, start=dst_msk.parent)))

    for split_name, split_stems in splits.items():
        for stem in split_stems:
            write_one(stem, split_name)

    print("[split_biodiversity_pool]")
    print(f"  total pairs: {n}")
    print(f"  train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"  seed: {args.seed}")
    print(f"  mode: {args.mode}")
    print(f"  out: {out_root.resolve()}")


if __name__ == "__main__":
    main()
