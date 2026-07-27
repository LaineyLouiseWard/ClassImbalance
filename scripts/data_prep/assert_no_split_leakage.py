#!/usr/bin/env python3
"""
Preflight gate: refuse to train if any held-out tile has reached the training side.

The original campaign leaked because the tiles are chipped on a 50% stride and the split was
random by tile (notes/rebuild_2026-07/decisions/TILE_OVERLAP_LEAKAGE_2026-07-25.md). The split is now spatially blocked, but
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

# Metres per degree: ONE definition, in geoseg/geo.py. These two literals appeared in seven
# files with no derivation, and terrain_separability used the LONGITUDE constant for the
# latitude direction.
from geoseg.geo import M_PER_DEG_LAT, M_PER_DEG_LON_EQ  # noqa: E402,F401

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


def _external_placement(by_crs: dict) -> tuple:
    """(km to the nearest internal tile, is any external centroid inside the internal extent).

    The internal splits are in UTM and the uplands in WGS84, so the per-CRS distance loop never
    compares them and Test B's independence went unverified entirely. Both quantities are computed
    in WGS84 so the two coordinate systems can be put side by side.
    """
    from rasterio.warp import transform as _warp

    internal, external = [], []
    for crs, rows in by_crs.items():
        for tid, s, l, bo, r, to in rows:
            xs, ys = _warp(crs, "EPSG:4326", [(l + r) / 2], [(bo + to) / 2])
            (external if s == EXTERNAL else internal).append((xs[0], ys[0]))
    if not internal or not external:
        return float("inf"), False
    A, B = np.array(internal), np.array(external)
    inside = bool(((B[:, 0] >= A[:, 0].min()) & (B[:, 0] <= A[:, 0].max())
                   & (B[:, 1] >= A[:, 1].min()) & (B[:, 1] <= A[:, 1].max())).any())
    Ar, Br = np.radians(A), np.radians(B)
    dlon = Ar[:, None, 0] - Br[None, :, 0]
    dlat = Ar[:, None, 1] - Br[None, :, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(Ar[:, None, 1]) * np.cos(Br[None, :, 1]) * np.sin(dlon / 2) ** 2
    return float((6371.0 * 2 * np.arcsin(np.sqrt(h))).min()), inside


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--oem-root", default=None,
                    help="combined Biodiversity+OEM pre-training root (stage2a pool)")
    ap.add_argument("--sampler-tsv", default=None)
    ap.add_argument("--augmentation-list", default=None,
                    help="minority-rich tile list for this split; defaults to $AUG_LIST. There is "
                         "deliberately no untagged fallback: that path is shared across splits, so "
                         "checking it while training on a tagged one tests the wrong file.")
    ap.add_argument("--confusion-npz", default=None,
                    help="teacher->GT confusion for this split. Recorded for provenance: it carries "
                         "no tile ids, so it cannot be checked for held-out bleed directly, but a "
                         "matrix whose pixel count disagrees with the training set is stale.")
    # Defaults are the shipped three-region design: train | 256 m | val | 650 m | test.
    ap.add_argument("--min-sep-train-val", type=float, default=256.0)
    ap.add_argument("--min-sep-val-test", type=float, default=650.0)
    ap.add_argument("--min-sep-train-test", type=float, default=650.0)
    ap.add_argument("--no-external-test", action="store_true",
                    help="declare that this split intentionally has no held-out external site. "
                         "Without it, a missing external_test/ is a failure rather than a silent "
                         "omission.")
    args = ap.parse_args()

    root = find_repo_root()
    split_root = root / args.split_root
    failures, checks, skipped = [], [], []

    splits = list(INTERNAL)
    split_ids = {s: ids_in(split_root / s / "images", "tif") for s in INTERNAL}
    for s in INTERNAL:
        if not split_ids[s]:
            failures.append(f"split '{s}' is empty at {split_root / s}")
    # Every tile must have BOTH an image and a mask. This read images/ only, so a split that had
    # lost its masks passed the gate and printed the image count as though nothing were wrong; the
    # loss would surface later as a training set quietly smaller than the one reported.
    for s in INTERNAL + (EXTERNAL,):
        imgs = ids_in(split_root / s / "images", "tif")
        msks = ids_in(split_root / s / "masks", "png")
        if not imgs and not msks:
            continue
        if imgs ^ msks:
            only_i, only_m = sorted(imgs - msks), sorted(msks - imgs)
            failures.append(
                f"{s}: {len(only_i)} tiles have an image and no mask (e.g. {only_i[:3]}), "
                f"{len(only_m)} have a mask and no image (e.g. {only_m[:3]})")
        else:
            checks.append(f"{s}: image/mask pairing exact over {len(imgs)} tiles")

    ext_ids = ids_in(split_root / EXTERNAL / "images", "tif")
    if ext_ids:
        split_ids[EXTERNAL] = ext_ids
        splits.append(EXTERNAL)
    elif not args.no_external_test:
        failures.append(f"no tiles at {split_root / EXTERNAL} — the held-out external site is the "
                        f"only test ground fully independent of the training site. Pass "
                        f"--no-external-test if this split is deliberately without one.")
    else:
        skipped.append("external test site (--no-external-test declared): every bleed and "
                       "separation check below covers three splits, not four")

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
        if oem_train and not bleed and not missing:
            checks.append(f"OEM pre-training pool: {len(oem_train)} tiles, no held-out bleed")
    else:
        skipped.append("OEM pre-training pool (--oem-root not supplied)")

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
            # Symmetric: extra ids are as wrong as missing ones. A TSV carrying the 413
            # buffer-dropped tiles printed "matches training set" over 1,485 ids.
            extra = tsv_ids - split_ids["train"]
            if extra:
                failures.append(f"sampler TSV has {len(extra)} ids that are not current training "
                                f"tiles, e.g. {sorted(extra)[:3]}")
            if not bleed and not missing and not extra:
                checks.append(f"sampler TSV: {len(tsv_ids)} ids, exactly the training set")
    else:
        skipped.append("sampler TSV (--sampler-tsv not supplied)")

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
            else:
                checks.append(f"teacher confusion: fitted on {tiles:.0f} tiles, "
                              f"matches training set")
    else:
        skipped.append("teacher confusion (--confusion-npz / $TEACHER_CONFUSION_NPZ not supplied)")

    # The untagged legacy default was removed on 2026-07-26. It made this check unskippable, which
    # is why a from-scratch run stopped here: A1b runs BEFORE A2 builds the list, so the exported
    # $AUG_LIST names a file that does not exist yet. Naming a list that is missing is still a hard
    # failure; naming no list at all is a skip, printed below so it cannot pass unnoticed.
    aug_arg = args.augmentation_list or os.environ.get("AUG_LIST")
    if not aug_arg:
        skipped.append("augmentation list (--augmentation-list / $AUG_LIST not supplied)")
    elif not (root / aug_arg).exists():
        failures.append(f"augmentation list not found: {aug_arg}")
    else:
        aug = root / aug_arg
        blob = json.loads(aug.read_text())
        # Keys are settlement_images / seminatural_images; a plain "ids" list is never written.
        aug_ids = (set(blob) if isinstance(blob, list)
                   else set(blob.get("settlement_images", [])) | set(blob.get("seminatural_images", [])))
        # An empty id set means the keys were renamed, not that the list is clean. Without this the
        # check printed "ok" over a file it had failed to read -- the same shape as the 2026-07-25 B2
        # defect one layer up.
        if not aug_ids:
            failures.append(f"{aug_arg} yielded zero ids: expected a list, or the keys "
                            f"settlement_images / seminatural_images. Keys present: "
                            f"{sorted(blob)[:6] if isinstance(blob, dict) else type(blob).__name__}")
        bleed = aug_ids & held_out
        if bleed:
            failures.append(f"{len(bleed)} held-out tiles in {aug_arg}, "
                            f"e.g. {sorted(bleed)[:3]}")
        # Every id must BE a current training tile. Intersecting against held_out alone is not
        # enough: an adversarial pass got a list of 302 ids past this check -- 40 of them held out --
        # simply by writing the ids with a `.tif` suffix, because a set of strings that matches
        # nothing intersects held_out in nothing and reads as clean. Anything unrecognised is a
        # failure, which covers suffix drift, whitespace padding, and the 413 buffer-dropped tiles
        # that belong to no split at all.
        stray = aug_ids - split_ids["train"]
        if stray:
            failures.append(f"{len(stray)} ids in {aug_arg} are not current training tiles "
                            f"(held-out, buffer-dropped, or a different id format), "
                            f"e.g. {sorted(stray)[:3]}")
        if not bleed and not stray and aug_ids:
            checks.append(f"augmentation list ({aug_arg}): {len(aug_ids)} ids, all current "
                          f"training tiles")

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
    # 4b. Minimum separation between splits, from the SAME independently re-read bounds. Until
    # 2026-07-26 nothing anywhere enforced the buffer on the shipped path: splits 1 m apart, and
    # splits exactly abutting, both returned PASSED. Zero overlap is a much weaker property than the
    # buffer the design claims, and the buffer is the whole argument for the split being independent.
    req = {("train", "val"): args.min_sep_train_val,
           ("val", "test"): args.min_sep_val_test,
           ("train", "test"): args.min_sep_train_test,
}
    sep_lines = []
    for (a, b), need in req.items():
        if a not in split_ids or b not in split_ids:
            continue
        best = float("inf")
        comparable = False
        for crs, rows in by_crs.items():
            ra = [r for r in rows if r[1] == a]
            rb = [r for r in rows if r[1] == b]
            if not ra or not rb:
                continue
            comparable = True
            # Degrees -> metres at the group's own latitude; the inland site is already in metres.
            lat = np.mean([(r[3] + r[5]) / 2 for r in rows])
            geo = abs(lat) <= 90 and max(abs(r[2]) for r in rows) <= 180
            mx = M_PER_DEG_LON_EQ * np.cos(np.radians(lat)) if geo else 1.0
            my = M_PER_DEG_LAT if geo else 1.0
            A = np.array([r[2:] for r in ra], float)
            for r in rb:
                _, _, l, bo, rt, to = r
                dx = np.maximum(np.maximum(l - A[:, 2], A[:, 0] - rt), 0.0) * mx
                dy = np.maximum(np.maximum(bo - A[:, 3], A[:, 1] - to), 0.0) * my
                best = min(best, float(np.hypot(dx, dy).min()))
        if not np.isfinite(best):
            skipped.append(f"{a}|{b} separation: not measurable, one split is empty")
            continue
        if best + 1e-6 < need:
            failures.append(f"{a}|{b} separation is {best:.0f} m, below the required {need:.0f} m")
        else:
            sep_lines.append(f"{a}|{b} {best:.0f} m (>= {need:.0f})")
    if sep_lines:
        checks.append("split separation: " + ",  ".join(sep_lines))

    # The external site is held out as a SEPARATE PLACE, not by a buffer, so a metre threshold would
    # be a number invented to look like a check. The question that actually matters is binary: does
    # the held-out site sit inside the ground the model trained on? Answered by containment, and
    # the measured distance is reported beside it for the reader rather than tested against a bar.
    if EXTERNAL in split_ids:
        km, inside = _external_placement(by_crs)
        if inside:
            failures.append(f"an {EXTERNAL} tile centroid falls inside the bounding box of the "
                            f"internal splits — the held-out site is not a separate place")
        elif np.isfinite(km):
            checks.append(f"{EXTERNAL} is a separate site: nearest centroid {km:.0f} km from any "
                          f"train/val/test tile, outside their extent")

    # Only report this as a passing check when it PASSED. Appending it unconditionally printed
    # "ok   geometry: ... 10 cross-split overlaps" beside the FAIL line that the same number
    # raised -- the exit code was right, but a gate that prints "ok" next to the number that
    # failed it is a gate people learn to skim.
    if n_overlap:
        failures.append(f"{n_overlap} cross-split footprint overlaps, e.g. {example}")
    else:
        checks.append(f"geometry: {sum(len(v) for v in by_crs.values())} tiles across "
                      f"{len(by_crs)} CRS, no cross-split overlaps")

    for c in checks:
        print(f"  ok   {c}")
    # A check that did not run is reported, not merely absent. A1b legitimately runs this gate over
    # the split geometry alone, and the difference between "geometry only" and "everything" was
    # previously visible only by counting the lines that were not printed.
    for s in skipped:
        print(f"  SKIP {s}")
    if failures:
        print("\nLEAKAGE PREFLIGHT FAILED:")
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("\nLEAKAGE PREFLIGHT PASSED — safe to train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
