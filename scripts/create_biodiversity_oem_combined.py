#!/usr/bin/env python3
from pathlib import Path
import shutil


def copy_tree(src_images: Path, src_masks: Path, dst_images: Path, dst_masks: Path, prefix: str | None = None) -> None:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    for img in src_images.glob("*.tif"):
        name = f"{prefix}_{img.name}" if prefix else img.name
        shutil.copy2(img, dst_images / name)

    for msk in src_masks.glob("*.png"):
        name = f"{prefix}_{msk.name}" if prefix else msk.name
        shutil.copy2(msk, dst_masks / name)


def main() -> None:
    out_root = Path("data/biodiversity_oem_combined")
    if out_root.exists():
        shutil.rmtree(out_root)

    # --- biodiversity_split (canonical split) ---
    bio_split = Path("data/biodiversity_split")

    # copy all biodiversity splits (train/val/test)
    for split in ["train", "val", "test"]:
        copy_tree(
            bio_split / f"{split}/images",
            bio_split / f"{split}/masks",
            out_root / f"{split}/images",
            out_root / f"{split}/masks",
        )

    # --- OEM (train only) ---
    oem_root = Path("data/openearthmap_relabelled")
    copy_tree(
        oem_root / "images",
        oem_root / "masks",
        out_root / "train/images",
        out_root / "train/masks",
        prefix="oem",
    )

    print("[create_biodiversity_oem_combined]")
    print("✓ train: biodiversity_split/train + OEM")
    print("✓ val:   biodiversity_split/val only")
    print("✓ test:  biodiversity_split/test only")


if __name__ == "__main__":
    main()
