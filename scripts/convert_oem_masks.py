#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import rasterio
from PIL import Image


def read_mask_tif(mask_tif: Path) -> np.ndarray:
    """Read a single-band mask TIFF into int64 numpy."""
    mask_tif = Path(mask_tif)
    with rasterio.open(mask_tif) as src:
        if src.count != 1:
            raise ValueError(f"Expected 1-band mask, got {src.count}: {mask_tif}")
        arr = src.read(1)
    return arr.astype(np.int64, copy=False)


def read_mask_png(mask_png: Path) -> np.ndarray:
    """Read a PNG mask into int64 numpy."""
    mask_png = Path(mask_png)
    return np.array(Image.open(mask_png)).astype(np.int64, copy=False)


def read_mask_any(mask_path: Path) -> np.ndarray:
    """Read TIFF/PNG mask into int64 numpy."""
    mask_path = Path(mask_path)
    suf = mask_path.suffix.lower()
    if suf in (".tif", ".tiff"):
        return read_mask_tif(mask_path)
    if suf == ".png":
        return read_mask_png(mask_path)
    raise ValueError(f"Unsupported mask extension: {mask_path}")


def remap_ids(src_ids: np.ndarray, id_map: Dict[int, int], default_value: int = 0) -> np.ndarray:
    """
    Remap integer class IDs using a lookup table.
    Unmapped IDs become default_value.
    """
    src_ids = np.asarray(src_ids, dtype=np.int64)
    if src_ids.size == 0:
        return src_ids.astype(np.uint8)

    max_id = int(src_ids.max())
    if max_id < 0:
        raise ValueError("Mask contains negative IDs; remap_ids expects non-negative IDs.")

    lut = np.full((max_id + 1,), default_value, dtype=np.int64)
    for k, v in id_map.items():
        k = int(k)
        if 0 <= k <= max_id:
            lut[k] = int(v)

    out = lut[src_ids]
    return out.astype(np.uint8, copy=False)


def save_mask_png(mask: np.ndarray, out_png: Path) -> None:
    """Save a 2D uint8 mask as PNG (mode L)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    img = Image.fromarray(mask.astype(np.uint8, copy=False), mode="L")
    img.save(out_png)


def convert_oem_mask_tif_to_png(
    mask_tif: Path,
    out_png: Path,
    *,
    id_map: Dict[int, int],
    default_value: int = 0,
) -> None:
    """TIFF -> (optional remap) -> PNG."""
    src = read_mask_tif(mask_tif)
    tgt = remap_ids(src, id_map=id_map, default_value=default_value)
    save_mask_png(tgt, out_png)


def quick_mask_stats(mask_paths: List[Path], max_checks: int = 200) -> None:
    """Quick sanity stats that works for both PNG and TIFF masks."""
    sample = mask_paths[:max_checks]
    if not sample:
        print("[MaskStats] No masks to check.")
        return

    uniques = set()
    maxv = -1
    has_255 = 0

    for p in sample:
        arr = read_mask_any(p)
        uniques.update(np.unique(arr).tolist())
        maxv = max(maxv, int(arr.max()))
        if (arr == 255).any():
            has_255 += 1

    uniques_sorted = sorted(int(u) for u in uniques)
    print(f"[MaskStats] Checked {len(sample)} masks for stats.")
    print(f"[MaskStats] Unique labels (sample): {uniques_sorted[:40]}{' ...' if len(uniques_sorted)>40 else ''}")
    print(f"[MaskStats] Max label (sample): {maxv}")
    print(f"[MaskStats] Masks containing 255 (sample): {has_255}")
