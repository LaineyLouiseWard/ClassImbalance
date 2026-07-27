#!/usr/bin/env python
"""
Pick, per held-out split, a set of tiles that do not overlap each other.

WHY. Tiles are 256 m footprints on a 128 m grid, so `cm += tile_cm` in
`evaluation/compute_metrics.py` counts a mid-strip pixel about four times and a strip-edge pixel
once: realised multiplicity 2.66x val, 2.85x Test A, 3.05x ireland1, 3.26x ireland2. Each test set
is therefore a differently-weighted average over its own ground, AND the two test sets are weighted
differently from each other -- which matters because the Test A vs Test B contrast is the headline.

This is the whole fix: score the subset. No averaging, no weighting, no new estimator -- the same
`compute_metrics.py`, given fewer filenames. Precedent: Cira et al. 2024, Remote Sensing 16(16) 2954
cut their test area "with no overlap" before computing metrics. The two aggregation alternatives
were measured and rejected (docs/CORRECTIONS_PAPER_PT2.md §3): averaging is a 3-4 member overlap
ensemble, and nearest-tile-centre changes the share of scored ground taken from tile margins by
different amounts per site.

Measured cost, deduplicated labelled ground: Test A keeps 5.20 of 5.76 km2 (90%), Test B 2.44 of
2.62 km2 (93%). Per-class retention is uniform (28-31% Test A, 23-29% uplands), so it thins space
without favouring a class.

Greedy in raster order, which is deterministic and reproducible. It is not guaranteed maximal, but
on this lattice it returns 90 / 18 / 33 tiles against theoretical maxima of 90 / 20 / 45 free of
the dropped-buffer holes -- close enough that a smarter packing is not worth the extra machinery.

Output: artifacts/scoring_subset_<SPLIT_TAG>.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import rasterio

SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(10):
        if (p / "data").is_dir() and (p / "scripts").is_dir():
            return p
        if p == p.parent:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


def site_of(tile_id: str) -> str:
    return str(tile_id).split("_")[0]


def lattice(root: Path, ids: list[str]) -> dict:
    """Integer (col,row) per tile, from its own georeferencing.

    Indices come from the tile's own resolution, never from rounded coordinates: the upland sites
    are in degrees, where rounding to a few decimals silently merges tiles hundreds of metres apart.
    """
    b = {}
    for t in ids:
        with rasterio.open(root / "data/biodiversity_raw/images" / f"{t}.tif") as s:
            b[t] = (s.bounds.left, s.bounds.bottom, s.res[0], s.res[1], s.width, s.height)
    rx, ry = b[ids[0]][2], b[ids[0]][3]
    sx, sy = (b[ids[0]][4] / 2) * rx, (b[ids[0]][5] / 2) * ry     # 50% stride
    x0 = min(v[0] for v in b.values())
    y0 = min(v[1] for v in b.values())
    idx = {}
    for t, (l, bt, *_rest) in b.items():
        i, j = (l - x0) / sx, (bt - y0) / sy
        if abs(i - round(i)) > 1e-3 or abs(j - round(j)) > 1e-3:
            raise SystemExit(f"{t}: not on the 50%-stride lattice (col={i:.4f}, row={j:.4f})")
        idx[t] = (round(i), round(j))
    return idx


def pack(idx: dict) -> list[str]:
    """Greedy non-overlapping selection. A tile occupies the 2x2 lattice cells it spans."""
    sel, used = [], set()
    for t, (i, j) in sorted(idx.items(), key=lambda kv: (kv[1][1], kv[1][0])):
        cells = {(i + a, j + c) for a in range(2) for c in range(2)}
        if not (cells & used):
            sel.append(t)
            used |= cells
    return sorted(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"artifacts/spatial_split_manifest_{SPLIT_TAG}.json")
    ap.add_argument("--out", default=f"artifacts/scoring_subset_{SPLIT_TAG}.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    out = root / args.out
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to regenerate")

    assign = json.loads((root / args.manifest).read_text())["assignment"]
    result = {}
    for split in ("val", "test", "external_test"):
        ids = sorted(t for t, v in assign.items() if v == split)
        by_site = defaultdict(list)
        for t in ids:
            by_site[site_of(t)].append(t)
        keep, per_site = [], {}
        for site, sids in sorted(by_site.items()):
            sel = pack(lattice(root, sorted(sids)))
            # Every kept pair must be disjoint. Cheap, and it is the only property that matters.
            idx = lattice(root, sorted(sids))
            occupied = set()
            for t in sel:
                i, j = idx[t]
                cells = {(i + a, j + c) for a in range(2) for c in range(2)}
                if cells & occupied:
                    raise SystemExit(f"{site}/{t}: selected tiles overlap")
                occupied |= cells
            per_site[site] = {"n_all": len(sids), "n_kept": len(sel)}
            keep += sel
            print(f"{split:14s} {site:13s} {len(sids):4d} -> {len(sel):3d}")
        result[split] = {"tiles": sorted(keep), "per_site": per_site}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"split_tag": SPLIT_TAG, "manifest": args.manifest,
         "rule": "greedy raster-order packing; a tile occupies the 2x2 cells of the 50%-stride "
                 "lattice that it spans, so no two kept tiles share ground",
         "why": "so every ground pixel is scored exactly once; see docs/CORRECTIONS_PAPER_PT2.md",
         "splits": result}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
