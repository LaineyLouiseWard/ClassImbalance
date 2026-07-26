#!/usr/bin/env python3
"""
Per-class support for every split, in the units that decide whether a number is estimable.

WHY. A per-class IoU is only as trustworthy as the amount of independent ground it rests on, and the
three obvious ways to express that disagree badly:

  * SHARE of foreground pixels. A proportion, not a support:
    1.9% of a 307-tile validation set is roughly four times the ground of 1.9% of a 79-tile one, yet
    the floor treats them identically.
  * PIXEL count. Large enough to look reassuring in almost every case, and almost meaningless here.
    The tiles are chipped on a 50% stride, so neighbouring tiles share ground; a claim resting on
    "168k pixels" turned out to rest on three tiles, and 63% of one fold's cropland pixels sat in
    three mutually overlapping tiles.
  * DISTINCT GROUND BLOCKS. Tiles binned onto a grid at the autocorrelation scale, so two tiles in
    the same block are not independent evidence. This is the honest denominator.

All three are reported, because the gap between them is the diagnosis. A class with millions of
pixels in two blocks is not estimable, and saying so requires showing both numbers side by side.

WHAT THIS DOES NOT DO. It reports; it does not classify. The reader decides whether 8 independent
blocks is enough to quote a per-class number. An earlier version applied invented thresholds and
labelled classes ok / weak / UNESTIMABLE; those labels are gone (D17).

Run:
    PYTHONPATH=. python scripts/analysis/report_class_support.py --split-root data/split_f1
    PYTHONPATH=. python scripts/analysis/report_class_support.py --split-root data/split_f1 --markdown
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

FOREGROUND = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}
SPLITS = ("train", "val", "test", "external_test")

# Grid cell for counting independent ground: two tiles in the same cell are within one
# autocorrelation length and are not independent evidence about a class.
#
# 950 m is NOT the inland site's measured range, which this comment claimed until 2026-07-26.
# Measured and committed under artifacts/correlogram/: inland composition 750 m (on 900 of 1,952
# tiles), inland spectral 1,350 m; 950 m is ireland2's composition range. It is used here because
# support is a class-COMPOSITION criterion and 950 m is above the inland composition scale, so it
# counts fewer independent units and cannot flatter the support.
# Sensitivity: PYTHONPATH=. python scripts/analysis/block_size_sensitivity.py
BLOCK_M = 950.0

# NO VERDICT THRESHOLDS. An earlier version of this script classified each class as ok / weak /
# UNESTIMABLE against MIN_BLOCKS=5, MIN_TILES=8 and a "weak" band at twice those. All three numbers were
# invented here, and labelling a result with a self-chosen bar is the same mistake that produced (and
# then withdrew) the rho pre-registration. The support counts are the useful output; the reader judges
# them. See D17.


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def tile_geometry(split_root: Path, split: str) -> dict:
    """{tile_id: (block_x, block_y)} at BLOCK_M, per CRS group so metres and degrees never mix."""
    rows = []
    for f in sorted((split_root / split / "images").glob("*.tif")):
        with rasterio.open(f) as d:
            b = d.bounds
            geo = bool(d.crs and d.crs.is_geographic)
            rows.append((f.stem, (b.left + b.right) / 2, (b.bottom + b.top) / 2, geo, str(d.crs)))
    out = {}
    by_crs = defaultdict(list)
    for r in rows:
        by_crs[r[4]].append(r)
    for crs, rs in by_crs.items():
        lat = float(np.mean([r[2] for r in rs]))
        if rs[0][3]:
            sx = BLOCK_M / (111320.0 * math.cos(math.radians(lat)))
            sy = BLOCK_M / 111132.0
        else:
            sx = sy = BLOCK_M
        for tid, cx, cy, _, _ in rs:
            # CRS in the key so two sites in different systems can never share a block id.
            out[tid] = (crs, int(math.floor(cx / sx)), int(math.floor(cy / sy)))
    return out


def support(split_root: Path, split: str) -> dict:
    """{class_index: {pixels, tiles, blocks, share}} for one split."""
    blocks = tile_geometry(split_root, split)
    mask_dir = split_root / split / "masks"
    px = defaultdict(int)
    tiles = defaultdict(set)
    blks = defaultdict(set)
    for p in sorted(mask_dir.glob("*.png")):
        m = np.array(Image.open(p).convert("L"))
        cnt = np.bincount(m.reshape(-1), minlength=6)
        for k in FOREGROUND:
            if cnt[k] > 0:
                px[k] += int(cnt[k])
                tiles[k].add(p.stem)
                if p.stem in blocks:
                    blks[k].add(blocks[p.stem])
    fg = sum(px.values()) or 1
    return {k: {"pixels": px[k], "tiles": len(tiles[k]), "blocks": len(blks[k]),
                "share": px[k] / fg} for k in FOREGROUND}




def report(split_root: Path, root: Path, markdown: bool) -> dict:
    name = split_root.name
    out = {"split_root": str(split_root.relative_to(root)), "block_m": BLOCK_M, "splits": {}}
    for split in SPLITS:
        if not (split_root / split / "masks").is_dir():
            continue
        sup = support(split_root, split)
        out["splits"][split] = {FOREGROUND[k]: v for k, v in sup.items()}

        if markdown:
            print(f"\n**{name} / {split}**\n")
            print("| class | share | pixels | tiles | independent blocks |")
            print("|---|---|---|---|---|")
            for k, v in sup.items():
                print(f"| {FOREGROUND[k]} | {100*v['share']:.2f}% | {v['pixels']:,} | "
                      f"{v['tiles']} | {v['blocks']} |")
        else:
            print(f"\n{name} / {split}")
            print(f"  {'class':<12} {'share':>7} {'pixels':>12} {'tiles':>6} {'blocks':>7}")
            for k, v in sup.items():
                print(f"  {FOREGROUND[k]:<12} {100*v['share']:6.2f}% {v['pixels']:12,} "
                      f"{v['tiles']:6d} {v['blocks']:7d}")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="emit the supplementary-table form")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = find_repo_root()
    roots = ([root / f"data/split_f{k}" for k in (1, 2, 3)] if args.all_folds
             else [root / (args.split_root or "data/split_f1")])
    blobs = [report(r, root, args.markdown) for r in roots]

    out = Path(args.out) if args.out else root / "artifacts/class_support.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blobs, indent=2))
    print(f"\nwrote {out.relative_to(root)}")


if __name__ == "__main__":
    main()
