#!/usr/bin/env python3
"""
Preflight gate: refuse to train if any held-out tile has reached the training side.

The original campaign leaked because the tiles are chipped on a 50% stride and the split was
random by tile (notes/TILE_OVERLAP_LEAKAGE_2026-07-25.md). The split is now spatially blocked, but
several artefacts are DERIVED from the training set and silently outlive a re-split:

  - data/biodiversity_oem_combined/train  (the stage2a pre-training pool -- if stale, the transfer
    arm pre-trains on tiles that are now test, contaminating the cell carrying the positive result)
  - artifacts/sampler_weights_clsbal*.tsv
  - artifacts/train_augmentation_list.json
  - artifacts/teacher_oem_gt_confusion.npz  (grounded on confusion over the training set)

This asserts, for a given split root:
  1. the three split directories are disjoint by tile id;
  2. no val/test tile id appears in the OEM combined pre-training pool;
  3. no val/test tile id appears in any training-side artefact;
  4. no val/test tile shares ground with a training tile (independent geometry re-read).

Exit code 0 = safe to train. Non-zero = stop.

Run:
    PYTHONPATH=. python scripts/data_prep/assert_no_split_leakage.py \
        --split-root data/biodiversity_split_spatial_a1 \
        --oem-root data/biodiversity_oem_combined_a1 \
        --sampler-tsv artifacts/sampler_weights_clsbal_a1.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

INTERNAL = ("train", "val", "test")
# The uplands are held out whole as a fourth split. It was originally omitted from this gate, so its
# 191 tiles went unchecked: never confirmed disjoint from train, never confirmed non-empty, and
# skipped by every bleed check. An empty external_test would have passed the gate and then produced
# an evaluation over nothing at all.
EXTERNAL = "external_test"


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def ids_in(d: Path, ext: str) -> set:
    return {f.stem for f in d.glob(f"*.{ext}")} if d.is_dir() else set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--oem-root", default=None,
                    help="combined Biodiversity+OEM pre-training root (stage2a pool)")
    ap.add_argument("--sampler-tsv", default=None)
    ap.add_argument("--augmentation-list", default=None,
                    help="minority-rich tile list for this split. Defaults to $AUG_LIST, then the "
                         "untagged legacy path. The legacy path is shared across splits, so "
                         "checking it while training on a tagged one would test the wrong file.")
    ap.add_argument("--confusion-npz", default=None,
                    help="teacher->GT confusion for this split. Recorded for provenance: it carries "
                         "no tile ids, so it cannot be checked for held-out bleed directly, but a "
                         "matrix whose pixel count disagrees with the training set is stale.")
    ap.add_argument("--no-external-test", action="store_true",
                    help="declare that this split intentionally has no held-out external site. "
                         "Without it, a missing external_test/ is a failure rather than a silent "
                         "omission.")
    args = ap.parse_args()

    root = find_repo_root()
    split_root = root / args.split_root
    failures, checks = [], []

    splits = list(INTERNAL)
    split_ids = {s: ids_in(split_root / s / "images", "tif") for s in INTERNAL}
    for s in INTERNAL:
        if not split_ids[s]:
            failures.append(f"split '{s}' is empty at {split_root / s}")

    ext_ids = ids_in(split_root / EXTERNAL / "images", "tif")
    if ext_ids:
        split_ids[EXTERNAL] = ext_ids
        splits.append(EXTERNAL)
    elif not args.no_external_test:
        failures.append(f"no tiles at {split_root / EXTERNAL} — the held-out external site is the "
                        f"only test ground fully independent of the training site. Pass "
                        f"--no-external-test if this split is deliberately without one.")

    # Every held-out split, so the artefact bleed checks below cover the external site too.
    held_out = set().union(*(split_ids[s] for s in splits if s != "train"))
    checks.append(f"split sizes: " + "  ".join(f"{s}={len(split_ids[s])}" for s in splits))

    # 1. disjoint by id, over every pair
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = split_ids[a] & split_ids[b]
            if shared:
                failures.append(f"{len(shared)} tile ids in BOTH {a} and {b}, "
                                f"e.g. {sorted(shared)[:3]}")
    checks.append(f"split directories are disjoint by tile id ({len(splits)} splits)")

    # 2. OEM combined pre-training pool
    if args.oem_root:
        oem_train = ids_in(root / args.oem_root / "train" / "images", "tif")
        if not oem_train:
            failures.append(f"OEM combined train pool is empty at {args.oem_root}")
        bleed = oem_train & held_out
        if bleed:
            failures.append(f"{len(bleed)} held-out tiles are in the OEM pre-training pool, "
                            f"e.g. {sorted(bleed)[:3]}")
        missing = split_ids["train"] - oem_train
        if missing:
            failures.append(f"OEM pool is missing {len(missing)} current training tiles "
                            f"(stale pool from an older split?), e.g. {sorted(missing)[:3]}")
        checks.append(f"OEM pre-training pool: {len(oem_train)} tiles, no held-out bleed")

    # 3. training-side artefacts
    if args.sampler_tsv:
        p = root / args.sampler_tsv
        if not p.exists():
            failures.append(f"sampler TSV not found: {args.sampler_tsv}")
        else:
            # No header row: build_clsbal_sampler.py writes "<tile_id>\t<weight>" from line one.
            tsv_ids = {ln.split("\t")[0].strip() for ln in p.read_text().splitlines()
                       if ln.strip() and "\t" in ln}
            bleed = tsv_ids & held_out
            if bleed:
                failures.append(f"{len(bleed)} held-out tiles in {args.sampler_tsv}, "
                                f"e.g. {sorted(bleed)[:3]}")
            missing = split_ids["train"] - tsv_ids
            if missing:
                failures.append(f"sampler TSV missing {len(missing)} current training tiles "
                                f"(stale from an older split), e.g. {sorted(missing)[:3]}")
            checks.append(f"sampler TSV: {len(tsv_ids)} ids, matches training set")

    # 3b. teacher->GT confusion staleness. It holds no tile ids, so held-out bleed cannot be checked
    # directly -- but its total pixel count is exactly n_train_tiles * 512 * 512, which pins the
    # training set it was fitted on. This is how the stale 1,846-tile matrix was caught.
    conf_path = args.confusion_npz or os.environ.get("TEACHER_CONFUSION_NPZ")
    if conf_path:
        p = root / conf_path
        if not p.exists():
            failures.append(f"teacher confusion not found: {conf_path} — A7 has not run for this "
                            f"split, so the OEM relabel rests on another split's mapping")
        else:
            total = int(np.load(p, allow_pickle=True)["hard"].sum())
            tiles = total / (512 * 512)
            if abs(tiles - len(split_ids["train"])) > 0.5:
                failures.append(f"{conf_path} was fitted on {tiles:.0f} tiles but this split trains "
                                f"on {len(split_ids['train'])} — stale mapping")
            checks.append(f"teacher confusion: fitted on {tiles:.0f} tiles, matches training set")

    aug = root / (args.augmentation_list or os.environ.get("AUG_LIST")
                  or "artifacts/train_augmentation_list.json")
    if aug.exists():
        blob = json.loads(aug.read_text())
        # Keys are settlement_images / seminatural_images; a plain "ids" list is never written.
        aug_ids = (set(blob) if isinstance(blob, list)
                   else set(blob.get("settlement_images", [])) | set(blob.get("seminatural_images", [])))
        bleed = aug_ids & held_out
        if bleed:
            failures.append(f"{len(bleed)} held-out tiles in {aug.name}, "
                            f"e.g. {sorted(bleed)[:3]}")
        checks.append(f"augmentation list ({aug.name}): {len(aug_ids)} ids, no held-out bleed")

    # 4. geometry, re-read from the rasters. Grouped by CRS because the inland site is in UTM metres
    # and the uplands in WGS84 degrees; comparing footprints across the two would be meaningless. The
    # sites are tens of km apart, so no genuine overlap is being hidden by that grouping.
    by_crs = defaultdict(list)
    for s in splits:
        for f in sorted((split_root / s / "images").glob("*.tif")):
            with rasterio.open(f) as d:
                b = d.bounds
                by_crs[str(d.crs)].append((f.stem, s, b.left, b.bottom, b.right, b.top))
    n_overlap = 0
    example = None
    for crs, rows in by_crs.items():
        arr = np.array([r[2:] for r in rows], float)
        L, B, R, T = arr.T
        tol = 1e-6 * float(np.min(R - L))
        for i, (ti, si, *_) in enumerate(rows):
            ox = np.minimum(R[i], R) - np.maximum(L[i], L)
            oy = np.minimum(T[i], T) - np.maximum(B[i], B)
            hit = (ox > tol) & (oy > tol)
            hit[i] = False
            for j in np.where(hit)[0]:
                if rows[j][1] != si:
                    n_overlap += 1
                    example = example or (ti, si, rows[j][0], rows[j][1])
    if n_overlap:
        failures.append(f"{n_overlap} cross-split footprint overlaps, e.g. {example}")
    checks.append(f"geometry: {sum(len(v) for v in by_crs.values())} tiles across "
                  f"{len(by_crs)} CRS, {n_overlap} cross-split overlaps")

    for c in checks:
        print(f"  ok   {c}")
    if failures:
        print("\nLEAKAGE PREFLIGHT FAILED:")
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("\nLEAKAGE PREFLIGHT PASSED — safe to train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
