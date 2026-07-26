#!/usr/bin/env python3
"""Grid phase and effective sample size for the 950 m block unit.

Two properties of `utils.spatial_blocks` that are invisible unless printed.

PHASE. The grid is anchored at the CRS origin -- for the inland site, the UTM 29N false easting of
500,000 m. That origin is arbitrary with respect to the landscape, so the block count is one member
of a family indexed by the offset. It matters twice: every bootstrap interval scales roughly as
1/sqrt(n_blocks), and the block count feeds the class-support criterion that admitted the split.

EFFECTIVE N. The blocks are badly unequal -- Test A's sixteen hold between 43 and 2 tiles -- so
resampling them as sixteen exchangeable draws overstates the support. Kish n_eff is reported beside
the nominal count, both by tiles per block and by foreground pixels per block.

Run:
    PYTHONPATH=. python scripts/analysis/block_phase_sweep.py
    PYTHONPATH=. python scripts/analysis/block_phase_sweep.py --self-test
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

from scripts.data_prep.build_spatial_split import (
    FOREGROUND, class_block_support, class_counts, read_pool, site_of,
)

BLOCK_M = 950.0
N_PHASES = 10          # 0.1-cell steps, both axes together


def blocks_at_phase(pool: dict, block_m: float, frac: float) -> dict:
    """{tile_id: block key} with the grid origin shifted by `frac` of a cell on both axes.

    Same construction as build_spatial_split.support_blocks, with the offset made explicit
    instead of being fixed at zero by the CRS origin.
    """
    out = {}
    by_site = defaultdict(list)
    for t in pool:
        by_site[site_of(t)].append(t)
    for site, ids in by_site.items():
        cy = float(np.mean([(pool[t][2] + pool[t][4]) / 2 for t in ids]))
        geo = max(abs(pool[t][1]) for t in ids) <= 180
        sx = block_m / (111320.0 * math.cos(math.radians(cy))) if geo else block_m
        sy = block_m / 111132.0 if geo else block_m
        for t in ids:
            x = ((pool[t][1] + pool[t][3]) / 2) / sx + frac
            y = ((pool[t][2] + pool[t][4]) / 2) / sy + frac
            out[t] = (site, int(math.floor(x)), int(math.floor(y)))
    return out


def kish(weights) -> float:
    w = np.asarray(list(weights), dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.size else float("nan")


def fg_pixels(counts: dict, tid: str) -> int:
    return int(sum(counts[tid][c] for c in FOREGROUND))


def self_test() -> int:
    """A shift of exactly one whole cell must reproduce the shipped partition; a shift of half a
    cell must not. Without this the sweep could be reporting the same grid ten times."""
    pool = {"biodiversity_0001": ("x", 0.0, 0.0, 100.0, 100.0),
            "biodiversity_0002": ("x", 900.0, 0.0, 1000.0, 100.0),
            "biodiversity_0003": ("x", 1800.0, 0.0, 1900.0, 100.0)}
    base = blocks_at_phase(pool, BLOCK_M, 0.0)
    same = blocks_at_phase(pool, BLOCK_M, 1.0)
    half = blocks_at_phase(pool, BLOCK_M, 0.5)
    relabel_ok = ({frozenset(k for k, v in base.items() if v == b) for b in set(base.values())}
                  == {frozenset(k for k, v in same.items() if v == b) for b in set(same.values())})
    moved = half != base
    print(f"  a whole-cell shift gives the same partition   [{'ok' if relabel_ok else 'FAIL'}]")
    print(f"  a half-cell shift moves at least one tile     [{'ok' if moved else 'FAIL'}]")
    ok = relabel_ok and moved
    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default="data/split_f1")
    ap.add_argument("--manifest", default="artifacts/spatial_split_manifest_f1.json")
    ap.add_argument("--out", default="artifacts/block_phase_sweep.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    repo = Path(__file__).resolve().parents[2]
    split_root = repo / args.split_root
    cache = repo / "artifacts" / "_cache"
    manifest = json.loads((repo / args.manifest).read_text())
    assign = dict(manifest["assignment"])
    floors = {"train": manifest["min_class_blocks"], "val": manifest["min_class_blocks"],
              "test": manifest["min_test_class_blocks"]}

    pool = read_pool(split_root, cache / "tile_bounds_pool.json")
    counts = class_counts(split_root, pool, cache / "tile_class_counts.json")
    # read_pool covers train/val/test only -- the adequacy criterion is defined on those three. The
    # uplands are still bootstrapped over, so their bounds are needed for the effective-n table.
    # Counted here rather than through class_counts, which is cached on a key covering the three
    # internal splits only; adding entries under that key would pollute the split builder's cache.
    ext = split_root / "external_test" / "images"
    if ext.is_dir():
        for f in sorted(ext.glob("*.tif")):
            with rasterio.open(f) as d:
                b = d.bounds
                pool[f.stem] = ("external_test", b.left, b.bottom, b.right, b.top)
            m = np.array(Image.open(split_root / "external_test" / "masks" / f"{f.stem}.png"
                                    ).convert("L"))
            counts[f.stem] = {c: int((m == c).sum()) for c in FOREGROUND}
    splits = ["train", "val", "test", "external_test"]

    # ── effective n at the shipped phase ────────────────────────────────────
    print(f"Kish effective n at the shipped phase, {BLOCK_M:.0f} m blocks\n")
    print(f"  {'split':14s} {'tiles':>6s} {'blocks':>7s} {'n_eff(tiles)':>13s} {'n_eff(fg px)':>13s}"
          f"   tiles per block")
    eff = {}
    b0 = blocks_at_phase(pool, BLOCK_M, 0.0)
    for s in splits:
        ids = [t for t, v in assign.items() if v == s]
        per_block_tiles = defaultdict(int)
        per_block_px = defaultdict(int)
        for t in ids:
            per_block_tiles[b0[t]] += 1
            per_block_px[b0[t]] += fg_pixels(counts, t)
        tpb = sorted(per_block_tiles.values(), reverse=True)
        eff[s] = {"n_tiles": len(ids), "n_blocks": len(per_block_tiles),
                  "n_eff_tiles": kish(per_block_tiles.values()),
                  "n_eff_fg_px": kish(per_block_px.values()), "tiles_per_block": tpb}
        print(f"  {s:14s} {len(ids):6d} {len(per_block_tiles):7d} "
              f"{eff[s]['n_eff_tiles']:13.2f} {eff[s]['n_eff_fg_px']:13.2f}   {tpb}")

    # ── phase sweep ─────────────────────────────────────────────────────────
    print(f"\nPhase sweep: {N_PHASES} offsets in 0.1-cell steps "
          f"({BLOCK_M / N_PHASES:.0f} m), both axes together\n")
    print(f"  {'offset':>7s} " + " ".join(f"{s:>14s}" for s in splits) + "   adequacy")
    rows = []
    for k in range(N_PHASES):
        frac = k / N_PHASES
        b = blocks_at_phase(pool, BLOCK_M, frac)
        nb = {s: len({b[t] for t, v in assign.items() if v == s}) for s in splits}
        sup = class_block_support(assign, counts, b)
        mins = {s: min(sup[s].values()) for s in floors if s in sup and sup[s]}
        passes = all(mins.get(s, 0) >= n for s, n in floors.items())
        rows.append({"offset_cells": frac, "offset_m": frac * BLOCK_M, "n_blocks": nb,
                     "min_class_blocks": mins, "passes": passes})
        print(f"  {frac:7.1f} " + " ".join(f"{nb[s]:14d}" for s in splits)
              + f"   {'passes' if passes else 'FAILS '}"
              + ("   <- shipped" if k == 0 else ""))

    n_pass = sum(r["passes"] for r in rows)
    print(f"\nThe shipped split clears its own adequacy criterion at {n_pass} of {N_PHASES} phases.")
    for s in splits:
        v = [r["n_blocks"][s] for r in rows]
        print(f"  {s:14s} block count over the sweep: min {min(v)}  shipped {v[0]}  max {max(v)}")
    print("\nInterval width scales roughly as 1/sqrt(n_blocks), so the same data at an equally "
          f"\narbitrary offset would give a CI up to {math.sqrt(max(r['n_blocks']['test'] for r in rows) / min(r['n_blocks']['test'] for r in rows)):.2f}x wider on Test A.")

    p = repo / args.out
    p.write_text(json.dumps({"block_m": BLOCK_M, "floors": floors, "effective_n": eff,
                             "n_phases": N_PHASES, "n_phases_passing": n_pass,
                             "phases": rows}, indent=2))
    print(f"\nwrote {p.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
