#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from geoseg.datasets.biodiversity_dataset import (
    BiodiversityTrainDataset,
    train_aug_minority,   # whatever is currently defined as train_aug_minority
)

# ---- label ids (your scheme) ----
BG = 0
FOREST = 1
GRASS = 2
CROP = 3
SETTLE = 4
SEMI = 5


def erode_binary(bin_mask: np.ndarray, r: int = 3) -> np.ndarray:
    """Binary erosion (prefers cv2). If unavailable, returns original."""
    if r <= 0:
        return bin_mask
    try:
        import cv2  # type: ignore
        k = 2 * r + 1
        kernel = np.ones((k, k), np.uint8)
        return cv2.erode(bin_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        return bin_mask


def mask_stats(mask: np.ndarray, erode_r: int = 3) -> dict:
    """
    mask: HxW int in [0..5]
    Returns fractions + interior counts for class 4/5.
    """
    valid = mask  # bg is not ignored here; we're just describing crops
    total = valid.size

    binc = np.bincount(valid.reshape(-1), minlength=6)
    frac = binc / float(total)

    # dominant class fraction (helps detect "mostly one class" crops)
    dom = float(frac.max())

    # interior pixels for settlement & seminatural
    stats = {"dom_frac": dom}
    for cls, name in [(SETTLE, "settle"), (SEMI, "semi")]:
        m = (valid == cls)
        stats[f"{name}_px"] = int(m.sum())
        interior = erode_binary(m, r=erode_r)
        stats[f"{name}_interior_px"] = int(interior.sum())
        # boundary-ish ratio: interior/total-of-class (low => mostly boundary/specks)
        stats[f"{name}_interior_ratio"] = float(
            (interior.sum() / m.sum()) if m.sum() > 0 else 0.0
        )
        stats[f"{name}_frac"] = float(frac[cls])

    return stats


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/biodiversity_split/train_rep")
    ap.add_argument("--n", type=int, default=300, help="number of samples to test")
    ap.add_argument("--erode-r", type=int, default=3)
    args = ap.parse_args()

    ds = BiodiversityTrainDataset(data_root=args.data_root, transform=train_aug_minority)

    n = min(args.n, len(ds))
    rows = []
    for i in range(n):
        item = ds[i]
        mask_t = item["gt_semantic_seg"]  # torch.LongTensor HxW
        mask = mask_t.cpu().numpy().astype(np.int64)
        rows.append(mask_stats(mask, erode_r=args.erode_r))

    # aggregate
    def mean(key): return float(np.mean([r[key] for r in rows]))
    def pct(cond): return 100.0 * float(np.mean([1.0 if cond(r) else 0.0 for r in rows]))

    print(f"Checked n={n} samples from {args.data_root}")
    print("---- Averages ----")
    print(f"dominant-class fraction (mean): {mean('dom_frac'):.3f}")
    print(f"settlement frac (mean): {mean('settle_frac'):.4f} | px: {mean('settle_px'):.1f}")
    print(f"seminatural frac (mean): {mean('semi_frac'):.4f} | px: {mean('semi_px'):.1f}")
    print(f"settlement interior px (mean): {mean('settle_interior_px'):.1f} | interior ratio (mean): {mean('settle_interior_ratio'):.3f}")
    print(f"seminatural interior px (mean): {mean('semi_interior_px'):.1f} | interior ratio (mean): {mean('semi_interior_ratio'):.3f}")

    print("---- “Bad crop” rates ----")
    # tweak these thresholds if you like; they’re just quick diagnostics
    print(f"dominant-class >= 0.75: {pct(lambda r: r['dom_frac'] >= 0.75):.1f}%")
    print(f"seminatural present but interior ~0 (semi_px>0 & semi_interior_px<20): {pct(lambda r: r['semi_px'] > 0 and r['semi_interior_px'] < 20):.1f}%")
    print(f"settlement present but interior ~0 (settle_px>0 & settle_interior_px<10): {pct(lambda r: r['settle_px'] > 0 and r['settle_interior_px'] < 10):.1f}%")


if __name__ == "__main__":
    main()
