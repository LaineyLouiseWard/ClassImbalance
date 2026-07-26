#!/usr/bin/env python3
"""
Per-class support for every split, in the units that decide whether a number is estimable.

WHY. A per-class IoU is only as trustworthy as the amount of independent ground it rests on, and the
three obvious ways to express that disagree badly:

  * SHARE of foreground pixels. What MIN_CLASS_SHARE constrains. It is a proportion, not a support:
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

WHERE THE FLOOR CANNOT REACH. MIN_CLASS_SHARE is a rejection criterion on a search over cut
placements. It works by discarding candidates. external_test is not in that search space: the two
upland sites are held out whole and are identical across every fold, so there is no candidate to
reject and no alternative to select. Settlement at 1.47% of upland foreground is a fact about that
landscape, not a property of a split choice, and no amount of searching produces a different upland.
The floor is therefore inapplicable there rather than waived -- but its PURPOSE still applies, so it
is enforced by reporting instead: this script marks classes too thin to interpret, and those are to
be reported as unestimable with their support stated, never as an estimate.

Run:
    PYTHONPATH=. python scripts/analysis/report_class_support.py --split-root data/split_f1
    PYTHONPATH=. python scripts/analysis/report_class_support.py --all-folds --markdown
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

# Grid cell for counting independent ground. 950 m is the correlogram range measured on the full tile
# pool, so two tiles in the same cell are within one autocorrelation length and are not independent
# evidence about a class. Deliberately the conservative end of the measured range (750 m shipped
# subsample, 950 m full pool): a larger cell counts fewer independent units, so it cannot flatter
# the support.
BLOCK_M = 950.0

# A class is unestimable in a split below EITHER threshold. Blocks is the binding one in practice.
MIN_BLOCKS = 5
MIN_TILES = 8


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


def verdict(s: dict) -> str:
    if s["blocks"] < MIN_BLOCKS or s["tiles"] < MIN_TILES:
        return "UNESTIMABLE"
    if s["blocks"] < 2 * MIN_BLOCKS:
        return "weak"
    return "ok"


def report(split_root: Path, root: Path, markdown: bool) -> dict:
    name = split_root.name
    out = {"split_root": str(split_root.relative_to(root)), "block_m": BLOCK_M,
           "min_blocks": MIN_BLOCKS, "min_tiles": MIN_TILES, "splits": {}}
    for split in SPLITS:
        if not (split_root / split / "masks").is_dir():
            continue
        sup = support(split_root, split)
        out["splits"][split] = {FOREGROUND[k]: {**v, "verdict": verdict(v)} for k, v in sup.items()}

        if markdown:
            print(f"\n**{name} / {split}**\n")
            print("| class | share | pixels | tiles | blocks | |")
            print("|---|---|---|---|---|---|")
            for k, v in sup.items():
                print(f"| {FOREGROUND[k]} | {100*v['share']:.2f}% | {v['pixels']:,} | "
                      f"{v['tiles']} | {v['blocks']} | {verdict(v)} |")
        else:
            print(f"\n{name} / {split}")
            print(f"  {'class':<12} {'share':>7} {'pixels':>12} {'tiles':>6} {'blocks':>7}")
            for k, v in sup.items():
                flag = verdict(v)
                mark = "  <-- report as unestimable" if flag == "UNESTIMABLE" else (
                    "  (weak)" if flag == "weak" else "")
                print(f"  {FOREGROUND[k]:<12} {100*v['share']:6.2f}% {v['pixels']:12,} "
                      f"{v['tiles']:6d} {v['blocks']:7d}{mark}")

    bad = [(sp, c) for sp, d in out["splits"].items() for c, v in d.items()
           if v["verdict"] == "UNESTIMABLE"]
    if bad:
        print(f"\n  {len(bad)} class x split combinations are UNESTIMABLE and must not be reported "
              f"as an estimate:")
        for sp, c in bad:
            v = out["splits"][sp][c]
            print(f"    {sp}/{c}: {v['blocks']} independent blocks, {v['tiles']} tiles "
                  f"({v['pixels']:,} px — pixel count is not support here)")
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
