#!/usr/bin/env python3
"""
Replicate selected training tiles into a *_rep split (TIFF images + PNG masks).

Correct behaviour (matches paper + Chantelle):
- ALL original training samples are kept
- Minority-rich samples are replicated with *_repN suffixes
- Final training pool = originals + replicas

Expected layout:
  data_root/
    images/*.tif
    masks/*.png

Output:
  out_root/
    images/*.tif
    masks/*.png
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Set


def replicate_images(
    data_root: str,
    augmentation_list_file: str,
    out_root: str,
    replications: int = 1,
    overwrite: bool = False,
) -> None:
    data_root_p = Path(data_root)
    out_root_p = Path(out_root)

    src_images = data_root_p / "images"
    src_masks = data_root_p / "masks"
    if not src_images.exists() or not src_masks.exists():
        raise FileNotFoundError(f"Missing source dirs: {src_images} / {src_masks}")

    dst_images = out_root_p / "images"
    dst_masks = out_root_p / "masks"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    # ---- load augmentation list ----
    with open(augmentation_list_file, "r", encoding="utf-8") as f:
        aug_list = json.load(f)

    settlement_images: Set[str] = set(aug_list.get("settlement_images", []))
    seminatural_images: Set[str] = set(aug_list.get("seminatural_images", []))
    selected_ids = sorted(settlement_images.union(seminatural_images))

    print("[replicate_minority_classes]")
    print(f"  data_root: {data_root_p}")
    print(f"  out_root:  {out_root_p}")
    print(f"  selected unique IDs: {len(selected_ids)}")
    print(f"    - Only Settlement: {len(settlement_images - seminatural_images)}")
    print(f"    - Only SemiNatural: {len(seminatural_images - settlement_images)}")
    print(f"    - Both classes: {len(settlement_images & seminatural_images)}")
    print(f"  replications per selected image: {replications}")
    print(f"  overwrite: {overwrite}\n")

    copied_base = 0
    created_reps = 0
    missing = 0

    # ---- STEP 1: copy ALL originals ----
    for img_path in src_images.glob("*.tif"):
        mask_path = src_masks / f"{img_path.stem}.png"
        if not mask_path.exists():
            missing += 1
            continue

        shutil.copy2(img_path, dst_images / img_path.name)
        shutil.copy2(mask_path, dst_masks / mask_path.name)
        copied_base += 1

    # ---- STEP 2: replicate minority images ----
    for img_id in selected_ids:
        img_path = src_images / f"{img_id}.tif"
        mask_path = src_masks / f"{img_id}.png"
        if not img_path.exists() or not mask_path.exists():
            missing += 1
            continue

        for rep in range(1, replications + 1):
            shutil.copy2(img_path, dst_images / f"{img_id}_rep{rep}.tif")
            shutil.copy2(mask_path, dst_masks / f"{img_id}_rep{rep}.png")
            created_reps += 1

    # ---- log ----
    log = {
        "data_root": str(data_root_p),
        "out_root": str(out_root_p),
        "replications_per_selected_image": replications,
        "original_images": copied_base,
        "replicated_images": created_reps,
        "final_total": copied_base + created_reps,
        "missing_pairs": missing,
        "thresholds": aug_list.get("thresholds"),
    }

    log_path = out_root_p / "replication_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("✓ Done")
    print(f"  originals copied:  {copied_base}")
    print(f"  replicas created: {created_reps}")
    print(f"  final total:      {copied_base + created_reps}")
    print(f"  log: {log_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/biodiversity_split/train")
    ap.add_argument("--out-root", default="data/biodiversity_split/train_rep")
    ap.add_argument("--augmentation-list", default="train_augmentation_list.json")
    ap.add_argument("--replications", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    replicate_images(
        data_root=args.data_root,
        augmentation_list_file=args.augmentation_list,
        out_root=args.out_root,
        replications=args.replications,
        overwrite=args.overwrite,
    )
