#!/usr/bin/env python3
"""Recompute the block-support adequacy criterion at several block sizes, on the shipped manifest.

SUPPORT_BLOCK_M is not only the bootstrap unit. It is the block size in MIN_CLASS_BLOCKS, which is
the criterion that ADMITTED this split, so the split's admissibility depends on which correlogram
number is used. This prints the dependence rather than asserting it.

Run:
    PYTHONPATH=. python scripts/analysis/block_size_sensitivity.py
    PYTHONPATH=. python scripts/analysis/block_size_sensitivity.py --self-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.data_prep.build_spatial_split import (
    FOREGROUND, SPLITS, class_block_support, class_counts, read_pool, support_blocks,
)

# The four block sizes worth reporting, and where each comes from. Measured 2026-07-26 by
# scripts/analysis/spatial_correlogram.py; the outputs are committed under artifacts/correlogram/.
BLOCK_SIZES = [
    (650.0, "ireland1 composition range, and the val|test buffer width"),
    (750.0, "INLAND composition range (900 of 1,952 tiles) -- the criterion's own scale"),
    (950.0, "shipped SUPPORT_BLOCK_M; measured range at ireland2, not inland"),
    (1350.0, "INLAND spectral range -- imagery similarity, a different question"),
]


def floors_from(manifest: dict) -> dict:
    """The per-split block minima THIS split was admitted under, read from its own manifest.

    Not a bar invented here (D17). These two numbers are the design constraint that selected the
    shipped placement, so the only honest question this script can ask is what the gate that
    actually ran would have said at a different block size.
    """
    n = manifest["min_class_blocks"]
    return {"train": n, "val": n, "test": manifest["min_test_class_blocks"]}


def support_at(block_m: float, assign: dict, counts: dict, pool: dict) -> dict:
    return class_block_support(assign, counts, support_blocks(pool, block_m))


def verdict(sup: dict, floors: dict) -> tuple[bool, dict]:
    """Does this support table clear the floors? Returns (passes, per-split minimum)."""
    mins = {s: min(sup[s].values()) for s in SPLITS if s in sup and sup[s]}
    return all(mins.get(s, 0) >= n for s, n in floors.items()), mins


def self_test() -> int:
    """A support table that must FAIL, and one that must PASS. The verdict is only worth printing
    if it can come out either way, and the 1350 m row below is the case that decides it."""
    ok = True
    floors = {"train": 5, "val": 5, "test": 8}
    good = {s: {c: floors[s] for c in FOREGROUND} for s in ("train", "val", "test")}
    passes, _ = verdict(good, floors)
    print(f"  exactly at every floor -> {'PASSES' if passes else 'FAILS'}  [{'ok' if passes else 'FAIL'}]")
    ok &= passes
    for split in ("train", "val", "test"):
        bad = {s: dict(v) for s, v in good.items()}
        bad[split][FOREGROUND[3]] = floors[split] - 1
        passes, _ = verdict(bad, floors)
        print(f"  one class one block below the {split:5s} floor -> "
              f"{'PASSES' if passes else 'FAILS'}  [{'ok' if not passes else 'FAIL'}]")
        ok &= not passes
    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default="data/split_f1")
    ap.add_argument("--manifest", default="artifacts/spatial_split_manifest_f1.json")
    ap.add_argument("--out", default="artifacts/block_size_sensitivity.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    repo = Path(__file__).resolve().parents[2]
    split_root = repo / args.split_root
    cache = repo / "artifacts" / "_cache"

    manifest = json.loads((repo / args.manifest).read_text())
    assign = dict(manifest["assignment"])            # {tile_id: split}
    floors = floors_from(manifest)
    pool = read_pool(split_root, cache / "tile_bounds_pool.json")
    counts = class_counts(split_root, pool, cache / "tile_class_counts.json")

    print(f"{args.manifest}: "
          + "  ".join(f"{s}={sum(1 for v in assign.values() if v == s)}"
                      for s in sorted(set(assign.values()))))
    print(f"\ndesign constraint this split was admitted under: every foreground class in "
          f"train >= {floors['train']}, val >= {floors['val']}, test >= {floors['test']} "
          f"independent blocks\n")
    print(f"{'block':>7}  {'train':>5} {'val':>4} {'test':>4}   verdict   provenance")

    out = {"manifest": args.manifest, "floors": floors, "by_block_m": {}}
    for block_m, why in BLOCK_SIZES:
        sup = support_at(block_m, assign, counts, pool)
        passes, mins = verdict(sup, floors)
        print(f"{block_m:6.0f}m  {mins.get('train', 0):5d} {mins.get('val', 0):4d} "
              f"{mins.get('test', 0):4d}   {'PASSES' if passes else 'FAILS ':7s}  {why}")
        out["by_block_m"][f"{block_m:.0f}"] = {
            "provenance": why, "passes": passes, "min_blocks_per_split": mins,
            "class_block_support": {s: {FOREGROUND[c]: n for c, n in sup[s].items()} for s in sup}}

    p = repo / args.out
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
