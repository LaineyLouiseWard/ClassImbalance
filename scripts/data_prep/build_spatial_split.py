#!/usr/bin/env python3
"""
Build a spatially blocked train/val/test split with a leakage buffer.

WHY: the Biodiversity tiles are chipped on a 50% stride (256 x 256 m footprints on a 128 m grid),
so every tile overlaps its eight neighbours. The original split
(`scripts/data_prep/split_biodiversity_dataset.py`) assigns tiles at random, which puts ~93% of each
held-out tile's ground area inside a training tile -- identical pixels and identical labels. See
notes/rebuild_2026-07/decisions/TILE_OVERLAP_LEAKAGE_2026-07-25.md.

WHAT THIS DOES:
  1. Repools every tile from the existing split directories and reads its footprint from the
     GeoTIFF transform. The three sites use two coordinate systems, so geometry is only ever
     compared within a site.
  2. Cuts ONE site into train | val | test strips along a single axis, searching cut placements for
     the one with the least composition drift and the closest match to the target proportions, and
     rejecting any that leaves a class with too little independent ground. Sites too small to cut are
     held out whole as external_test.
  3. Applies a buffer band centred on each cut, so the minimum separation between splits is the
     buffer width. Symmetric, so no split is trimmed preferentially.
  4. Verifies zero cross-split footprint overlap, and reports retained counts and per-split class
     composition.

Blocks are laid within each site rather than holding whole sites out because the two designs answer
different questions, not because leave-one-site-out is infeasible. Semi-natural grassland is ~74% of
foreground pixels at ireland2 and ~27% at ireland1 against ~5% inland, but the inland site is so much
larger that it still holds 57.8% of the dataset's semi-natural pixel MASS (25.5M of 44.1M) across 614
tiles, so LOSO leaves the priority class well represented in training. Cutting within a site estimates accuracy on unlabelled ground inside a
surveyed landscape; holding the uplands out whole estimates transfer to an unsurveyed region. Both
are reported. Do not conflate prevalence with mass when justifying either.

The block-grid search and the leave-one-site-out variant were DELETED on 2026-07-26. Three separate
constraints -- the class floor, the composition-drift penalty and the target-proportion term -- were
each found stranded on that path, enforced in code the shipped design never reached. Measured
directly, the grid also lost 44.3% of tiles to buffers against 18% for two cuts, so it was never
going to be adopted. Keeping it only produced defects one at a time.

Output: artifacts/spatial_split_manifest.json (tile -> split or 'dropped', plus the audit).
Materialising the directories is a separate step (--materialise).

Run:
    PYTHONPATH=. python scripts/data_prep/build_spatial_split.py
    PYTHONPATH=. python scripts/data_prep/build_spatial_split.py --materialise --mode symlink
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# Metres per degree: ONE definition, in geoseg/geo.py. These two literals appeared in seven
# files with no derivation, and terrain_separability used the LONGITUDE constant for the
# latitude direction.
from geoseg.geo import M_PER_DEG_LAT, M_PER_DEG_LON_EQ  # noqa: E402,F401

SPLITS = ("train", "val", "test")
# A fourth destination, outside the blocked design. Sites whose extent is smaller than twice their
# autocorrelation range cannot be split internally at all (Roberts et al. 2017), so they are held
# out whole as a fixed external test set: one model, two questions -- accuracy on new ground inside
# a surveyed area (test) and accuracy on terrain never labelled (external_test).
EXTERNAL = "external_test"
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}
FOREGROUND = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}

# Adequacy is enforced on INDEPENDENT BLOCK SUPPORT (MIN_CLASS_BLOCKS below), not on class share.
# A share floor was tried and retired on 2026-07-26: it does not track estimability in either
# direction. Measured on the folds it produced, validation semi-natural at 1.91% share had 11
# independent blocks and was perfectly usable, while validation cropland at 7.00% share and 2.6M
# pixels had 3 blocks and was not. Keeping both criteria also let them drift apart -- the constant
# read 0.015 while every shipped manifest recorded 0.01 and one fold sat at 0.01137.
# Every site must contribute at least this many tiles to every split. One or two tiles satisfies a
# bare presence test while being far too few to estimate anything from.
MIN_SITE_TILES = 4


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        # `artifacts` and `scripts`, NOT `data`: data/ is gitignored, so keying on it made every
        # one of these scripts raise "repo root not found" in a fresh clone -- including the ledger
        # a reviewer would run to check the paper's numbers.
        if (parent / "artifacts").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def site_of(tile_id: str) -> str:
    """Site == coordinate-system group; footprints are only comparable within one."""
    return tile_id.split("_")[0]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _source_key(split_root: Path, sub: str, ext: str) -> str:
    """Cheap fingerprint of the source files, so a stale cache cannot silently be reused."""
    files = [f for s in SPLITS for f in sorted((split_root / s / sub).glob(f"*.{ext}"))]
    return f"{len(files)}:{max((f.stat().st_mtime_ns for f in files), default=0)}"


def read_pool(split_root: Path, cache: Path | None) -> dict:
    """{tile_id: (orig_split, left, bottom, right, top)} over every tile in the current split."""
    key = _source_key(split_root, "images", "tif")
    if cache is not None and cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("key") == key:
            return {k: tuple(v) for k, v in blob["pool"].items()}
    pool = {}
    for split in SPLITS:
        for f in sorted((split_root / split / "images").glob("*.tif")):
            with rasterio.open(f) as d:
                b = d.bounds
                pool[f.stem] = (split, b.left, b.bottom, b.right, b.top)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "pool": pool}))
    return pool


# ---------------------------------------------------------------------------
# Class composition
# ---------------------------------------------------------------------------
def class_counts(split_root: Path, pool: dict, cache: Path | None) -> dict:
    """{tile_id: {class: pixel count}} from the ground-truth masks."""
    key = _source_key(split_root, "masks", "png")
    if cache is not None and cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("key") == key:
            return {k: {int(c): n for c, n in v.items()} for k, v in blob["counts"].items()}
    out = {}
    for i, (tid, (split, *_)) in enumerate(sorted(pool.items()), 1):
        m = np.array(Image.open(split_root / split / "masks" / f"{tid}.png").convert("L"))
        out[tid] = {c: int((m == c).sum()) for c in FOREGROUND}
        if i % 250 == 0:
            print(f"  counted {i}/{len(pool)} masks", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "counts": out}))
    return out



# Grid cell for counting INDEPENDENT ground. Two tiles in the same cell are within one correlation
# length and are not independent evidence about a class.
#
# WHERE 950 COMES FROM, corrected 2026-07-26. It is NOT the inland site's measured range, which
# earlier versions of this comment claimed. Measured, committed under artifacts/correlogram/:
# inland composition range 750 m (on 900 of 1,952 tiles), inland spectral range 1,350 m; 950 m is
# ireland2's composition range. Block support is a class-COMPOSITION criterion, so the composition
# scale is the one that applies, and 950 m is CONSERVATIVE against it -- a larger cell counts fewer
# independent units, so it cannot flatter the support. The shipped split clears the floors at 650,
# 750 and 950 m and fails at 1,350 m, which is the spectral range and answers a different question
# (imagery similarity, not class-composition independence).
# Reproduce the table: PYTHONPATH=. python scripts/analysis/block_size_sensitivity.py
SUPPORT_BLOCK_M = 950.0
# Minimum independent blocks a class needs in a split before a per-class number from it means
# anything. This REPLACES the share floor as the binding criterion, because a share does not track
# estimability: measured on the 2026-07-26 folds, validation semi-natural at 1.91% share had 11
# blocks and was fine, while validation cropland at 7.00% share and 2.6M pixels had 3 blocks and was
# not. The share floor passes the second and rejects the first, i.e. exactly backwards.
MIN_CLASS_BLOCKS = 5
# Floor on the training strip. Not tight -- the 80/10/10 ranking term controls the size in
# practice; this only rejects degenerate placements.
MIN_TRAIN_TILES = 500


def support_blocks(pool: dict, block_m: float = SUPPORT_BLOCK_M) -> dict:
    """{tile_id: block key} on a grid at the autocorrelation scale, per site so CRSs never mix."""
    out = {}
    by_site = defaultdict(list)
    for t in pool:
        by_site[site_of(t)].append(t)
    for site, ids in by_site.items():
        cy = float(np.mean([(pool[t][2] + pool[t][4]) / 2 for t in ids]))
        # Degrees if the coordinates are small enough to be lon/lat; metres otherwise.
        geo = max(abs(pool[t][1]) for t in ids) <= 180
        sx = block_m / (M_PER_DEG_LON_EQ * math.cos(math.radians(cy))) if geo else block_m
        sy = block_m / M_PER_DEG_LAT if geo else block_m
        for t in ids:
            x = ((pool[t][1] + pool[t][3]) / 2) / sx
            y = ((pool[t][2] + pool[t][4]) / 2) / sy
            out[t] = (site, int(math.floor(x)), int(math.floor(y)))
    return out


def class_block_support(assign: dict, counts: dict, blocks: dict) -> dict:
    """{split: {class: number of distinct independent blocks holding any pixel of that class}}."""
    seen = {s: defaultdict(set) for s in SPLITS}
    for tid, s in assign.items():
        if s not in seen:
            continue
        for c in FOREGROUND:
            if counts[tid][c] > 0:
                seen[s][c].add(blocks[tid])
    return {s: {c: len(seen[s][c]) for c in FOREGROUND} for s in SPLITS}


def split_class_shares(assign: dict, counts: dict) -> dict:
    """Per-split share of foreground pixels held by each class."""
    tot = {s: Counter() for s in SPLITS}
    for tid, s in assign.items():
        if s in tot:
            tot[s].update(counts[tid])
    shares = {}
    for s in SPLITS:
        n = sum(tot[s].values()) or 1
        shares[s] = {FOREGROUND[c]: tot[s][c] / n for c in FOREGROUND}
    return shares


# ---------------------------------------------------------------------------
# Assignment search
# ---------------------------------------------------------------------------



def _bounds_from_rasters(ids, root: Path, split_root: Path, orig: dict) -> dict:
    """{tile: (left, bottom, right, top)} re-read from the GeoTIFFs, never from the cache.

    pairwise_separation_m used to read `pool`, which is the mtime-keyed cache AND the same dict that
    placed the cuts, so its docstring's independence claim was false: a doctored cache with a valid
    fingerprint reported 20,384 m for a real 128 m gap and passed.
    """
    out = {}
    for t in ids:
        with rasterio.open(split_root / orig[t] / "images" / f"{t}.tif") as d:
            b = d.bounds
            out[t] = (b.left, b.bottom, b.right, b.top)
    return out


def pairwise_separation_m(kept: dict, root: Path, split_root: Path, orig: dict, site: str,
                          geographic: bool = False, lat: float = 0.0) -> dict:
    """{(split_a, split_b): minimum footprint-to-footprint distance in metres}.

    Per PAIR, not a global minimum: this design separates its boundaries by different amounts on
    purpose (train|val by the pixel-identity distance, val|test by the full buffer), so a single
    global minimum is meaningless and comparing it to the largest requirement fails spuriously.

    Recomputed from raw GeoTIFF bounds, independently of the band arithmetic that produced the split,
    so it cannot pass by construction.
    """
    # Re-read from the GeoTIFFs. This used to take the cached `pool` -- the SAME dict that placed the
    # cuts -- so the docstring's independence claim was false and a cache with a reconstructed
    # fingerprint reported 41,536 m for a real 1,664 m gap and exited 0.
    bnds = _bounds_from_rasters(list(kept), root, split_root, orig)
    ids = [t for t in kept if site_of(t) == site and kept[t] in SPLITS]
    arr = np.array([bnds[t] for t in ids], float)   # (left, bottom, right, top)
    lab = np.array([kept[t] for t in ids])
    L, B, R, T = arr.T
    mx, my = ((M_PER_DEG_LON_EQ * math.cos(math.radians(lat)), M_PER_DEG_LAT) if geographic else (1.0, 1.0))
    out = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ia, ib = np.where(lab == a)[0], np.where(lab == b)[0]
        if len(ia) == 0 or len(ib) == 0:
            out[(a, b)] = float("nan")
            continue
        best = float("inf")
        for i in ia:
            dx = np.maximum(np.maximum(L[ib] - R[i], L[i] - R[ib]), 0.0) * mx
            dy = np.maximum(np.maximum(B[ib] - T[i], B[i] - T[ib]), 0.0) * my
            best = min(best, float(np.min(np.hypot(dx, dy))))
        out[(a, b)] = best
    return out


def build_three_region(pool: dict, counts: dict, site: str, external: tuple,
                       buffer_test_m: float, buffer_val_m: float, seed: int, restarts: int,
                       priors: list,
                       min_val_tiles: int | None = None,
                       min_test_tiles: int | None = None,
                       min_class_blocks: int = MIN_CLASS_BLOCKS,
                       min_test_class_blocks: int | None = None,
                       blocks: dict | None = None) -> dict:
    """THE SHIPPED DESIGN. Cut one site into train | val | test strips; hold other sites out whole.

    Geometry, along one axis of the splittable site:

        train ........ | <- buffer_val_m -> | val | <- buffer_test_m -> | test

    Two design decisions, both deliberate:

    1. **Two straight cuts, not a block grid.** Each cut costs one buffered boundary. A 3x3 grid puts
       boundaries on every side of every block, which at this site's size (about six autocorrelation
       lengths across) destroys ~49% of the tiles where two cuts cost ~19%.

    2. **Asymmetric buffers.** The val|test boundary gets the full separation; the train|val boundary
       gets only the pixel-identity distance. Validation selects checkpoints and confirms convergence
       -- it is never reported as an accuracy estimate, and whatever optimism it carries is
       common-mode across all factorial cells, so it cannot manufacture a between-cell effect. Only
       the test set carries the full guarantee. This is also what makes three regions fit at all: at
       650 m on BOTH boundaries a ~680 m validation strip loses almost its whole width, which is what
       silently emptied the val and test strips on an earlier attempt.

    A buffer is a band of width B centred on a cut, so the guarantee delivered is *minimum separation
    B between the splits either side* -- not "B of ground removed", which is twice as expensive and
    was the source of that earlier failure.

    Search is over cut PLACEMENT, which is the single-cut analogue of prevalence-stratified blocks
    (Elith et al. 2008, via Roberts et al. 2017): with no blocks to stratify, where the cuts fall is
    the only lever on class composition. Candidates are scored on their WORST per-class share in any
    split, because a class thin in a small test set makes its IoU noise -- and foreground mIoU weights
    classes equally and drives checkpoint selection.
    """
    ids = [t for t in pool if site_of(t) == site]
    rng = np.random.default_rng(seed)
    best = None
    n_below_blocks = 0
    blocks = blocks if blocks is not None else support_blocks(pool)

    for _ in range(restarts):
        axis = "x" if rng.random() < 0.5 else "y"
        train_low = rng.random() < 0.5          # wide train strip at the low end of the axis?
        # Strip widths as fractions of the site's extent along the chosen axis. Ranges widened
        # 2026-07-26: with f_val clipped at 0.20 the search space contained NO placement satisfying
        # spatially clustered and only a wide validation strip captures enough of both. Every
        # placement that clears the floor needs f_val around 0.22, so the old clip excluded the
        # feasible region entirely rather than the floor being unreachable.
        f_test = float(np.clip(rng.normal(0.16, 0.045), 0.08, 0.28))
        f_val = float(np.clip(rng.normal(0.16, 0.05), 0.07, 0.30))

        coord = {t: ((pool[t][1] + pool[t][3]) / 2 if axis == "x"
                     else (pool[t][2] + pool[t][4]) / 2) for t in ids}
        v = np.array([coord[t] for t in ids])
        if train_low:
            c_tv, c_vt = float(np.quantile(v, 1 - f_test - f_val)), float(np.quantile(v, 1 - f_test))
            roles = ("train", "val", "test")
        else:
            c_vt, c_tv = float(np.quantile(v, f_test)), float(np.quantile(v, f_test + f_val))
            roles = ("test", "val", "train")

        assign = {t: EXTERNAL for t in pool if site_of(t) in external}
        for t in ids:
            assign[t] = roles[0] if coord[t] <= min(c_tv, c_vt) else (
                roles[2] if coord[t] > max(c_tv, c_vt) else roles[1])

        # drop tiles whose footprint intersects either band
        dropped = set()
        for t in ids:
            lo, hi = ((pool[t][1], pool[t][3]) if axis == "x" else (pool[t][2], pool[t][4]))
            if (lo < c_tv + buffer_val_m / 2 and hi > c_tv - buffer_val_m / 2) or \
               (lo < c_vt + buffer_test_m / 2 and hi > c_vt - buffer_test_m / 2):
                dropped.add(t)
        kept = {t: sp for t, sp in assign.items() if t not in dropped}

        got = Counter(kept.values())
        # Val and test carry different guarantees, so they get different minima -- the same asymmetry
        # as the buffers and the drift term. Test carries the reported estimate and must be large
        # enough to estimate per-class IoU from. Validation only selects checkpoints and confirms
        # convergence, and is never reported, so it needs enough tiles for that choice to be stable
        # rather than enough to publish. A single shared minimum at 150 admitted only two distinct
        # folds; separating them admits three without weakening the test sets at all.
        need = {"train": MIN_TRAIN_TILES,
                "val": min_val_tiles if min_val_tiles is not None else MIN_TRAIN_TILES,
                "test": min_test_tiles if min_test_tiles is not None else MIN_TRAIN_TILES}
        if any(got.get(sp, 0) < need[sp] for sp in SPLITS):
            continue
        # Each fold must hold out substantially different ground from every earlier fold. Bounded
        # overlap, not disjointness: with straight cuts only TWO fully disjoint test strips exist
        # along one axis, and a third fold must cut the other axis, whose strip necessarily contains
        # tiles from both of the first two. Demanding disjointness therefore caps the design at two
        # folds. Two strips on perpendicular axes intersect only in a corner -- for ~16% strips that
        # is ~2.6% of the site -- so a small cap admits four distinct test regions (one per edge)
        # while still bounding how much they share. The realised overlap is recorded in the manifest.
        # Folds must hold out EXACTLY disjoint test ground. The bounded-overlap allowance was
        # retired on 2026-07-26 with the third fold: only two fully disjoint test strips exist along
        # one axis, so a third fold had to cut the perpendicular axis and share corners (13.9% and
        # 14.7% realised). That allowance was the single hardest item in the design to defend, and it
        # existed solely to admit that third fold.
        test_tiles = frozenset(t for t, sp in kept.items() if sp == "test")
        overlaps = [len(test_tiles & prior) / max(1, min(len(test_tiles), len(prior)))
                    for prior in priors]
        if overlaps and max(overlaps) > 0.0:
            continue

        shares = split_class_shares(kept, counts)
        worst = min(shares[sp][c] for sp in SPLITS for c in shares[sp])


        # The binding criterion. A class needs enough INDEPENDENT ground in a split, not enough
        # pixels and not a high enough share: the 50% stride means neighbouring tiles repeat the same
        # ground, so pixel counts and shares both overstate support, and they overstate it unevenly.
        # The TEST split gets a stricter minimum than train and val, for the same reason it gets the
        # wider buffer and the higher tile minimum: it is the only split whose per-class numbers are
        # reported. Without this the search returns a placement whose test cropland sits on 5 blocks
        # and 20 tiles -- clearing the floor, but marked "weak" by report_class_support.py and so not
        # really reportable. Requiring 8 finds a placement with 10 blocks and 73 tiles instead.
        need_blocks = {"train": min_class_blocks, "val": min_class_blocks,
                       "test": min_test_class_blocks if min_test_class_blocks is not None
                               else min_class_blocks}
        sup = class_block_support(kept, counts, blocks)
        if any(sup[sp][c] < need_blocks[sp] for sp in SPLITS for c in FOREGROUND):
            n_below_blocks += 1
            continue

        # Composition drift against the whole-site prior. Also confined to the dead path until now,
        # which let f2 ship with 13.84% cropland in validation against 2.13% in its own test set -- a
        # 6.5x ratio, so its checkpoint was selected against a prior neither its training nor its test
        # set shared. Ranked ahead of worst-share: once the floor is a hard reject, every surviving
        # candidate is adequate, and what remains to optimise is comparability between the splits.
        pooled = split_class_shares({t: "train" for t in kept}, counts)["train"]
        # Drift is measured on ALL THREE splits against the whole-site prior.
        # Exempting validation was wrong. The "optimism is common-mode across all four cells" argument
        # is sound WITHIN a fold, and is why the headline contrast is safe -- but it does not cover
        # comparison BETWEEN folds, which is the only thing multiple folds are bought for. Realised
        # cost of the exemption: validation cropland at 20.70% (f1) and 14.91% (f3) against 4.17% and
        # 1.59% in those folds' own test sets, i.e. 5x and 9x. That is larger than the 13.84%-vs-2.13%
        # case the exemption's own docstring cited as the failure it had fixed.
        drift = sum(abs(shares[sp][c] - pooled[c]) for sp in SPLITS for c in pooled)

        # Deviation from the 80/10/10 target. This is the THIRD term that was confined to score() on
        # the dead block-grid path, and omitting it is not cosmetic: ranking on drift alone favours fat
        # validation and test strips, and the first three-fold attempt at this floor produced training
        # sets of 1209, 841 and 894 tiles. A 44% spread in training volume across folds makes the folds
        # incomparable as blocks and throws away most of the data for no scientific reason.
        n_int = sum(1 for s in kept.values() if s in SPLITS) or 1
        dev = sum(abs(got.get(sp, 0) / n_int - TARGET[sp]) for sp in SPLITS)

        # Summed on the same footing as score() did. Both are L1 deviations on comparable scales, and
        # the floor is already a hard reject, so every candidate reaching here is adequate.
        key = (round(dev + drift, 4), -round(worst, 4), len(dropped))
        if best is None or key < best[0]:
            best = (key, {"axis": axis, "roles": list(roles), "f_test": f_test, "f_val": f_val,
                          "test_overlap_with_priors": [round(o, 4) for o in overlaps],
                          "cut_train_val": c_tv, "cut_val_test": c_vt,
                          "buffer_val_m": buffer_val_m, "buffer_test_m": buffer_test_m,
                          "composition_drift": round(drift, 4),
                          "proportion_deviation": round(dev, 4)},
                    kept, dropped, worst, shares)

    if best is None:
        raise SystemExit(
            f"no viable cut placement in {restarts} tries ({n_below_blocks} rejected for a class with under {min_class_blocks} independent "
            f"{SUPPORT_BLOCK_M:.0f} m blocks in some split; the rest "
            f"left under {MIN_TRAIN_TILES} tiles in a split or duplicated an earlier fold's test ground). "
            f"Raise --restarts, widen the strip fractions, or lower --min-split-tiles.")
    _, geom, kept, dropped, worst, shares = best
    return {"assignment": kept, "dropped": dropped, "geometry": geom,
            "worst_class_share": worst, "class_shares": shares,
            "class_block_support": class_block_support(kept, counts, blocks),
            "min_class_blocks": min_class_blocks, "support_block_m": SUPPORT_BLOCK_M}


def materialise(kept: dict, pool: dict, root: Path, split_root: Path, out_root: str, mode: str):
    """Write out_root/{train,val,test}/{images,masks} for the kept assignment."""
    out = root / out_root
    if out.exists():
        shutil.rmtree(out)
    for s in set(kept.values()):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "masks").mkdir(parents=True, exist_ok=True)
    for tid, s in kept.items():
        orig = pool[tid][0]
        for sub, ext in (("images", "tif"), ("masks", "png")):
            src = split_root / orig / sub / f"{tid}.{ext}"
            dst = out / s / sub / f"{tid}.{ext}"
            if mode == "symlink":
                dst.symlink_to(os.path.relpath(src, start=dst.parent))
            else:
                shutil.copy2(src, dst)


def verify_independent(kept: dict, root: Path, split_root: Path, orig_split: dict) -> list:
    """Re-derive the geometry from the GeoTIFFs and list every cross-split overlap.

    Deliberately shares no code, cache or data structure with the buffer step. Checking `kept`
    against the same `nbr` graph that built it is tautological -- it cannot fail even if the graph
    is wrong -- and this guarantee is the one thing the whole re-run depends on. Tiles are grouped
    by CRS, not by site name, so two sites in the same CRS would still be compared.
    """
    by_crs = defaultdict(list)
    for tid in kept:
        p = split_root / orig_split[tid] / "images" / f"{tid}.tif"
        with rasterio.open(p) as d:
            b = d.bounds
            by_crs[str(d.crs)].append((tid, b.left, b.bottom, b.right, b.top))

    bad = []
    for crs, rows in by_crs.items():
        ids = [r[0] for r in rows]
        arr = np.array([r[1:] for r in rows], float)
        L, B, R, T = arr.T
        # Tolerance relative to footprint size: one absolute epsilon cannot serve both a metre CRS
        # (bounds ~5.8e6, 1 ULP ~9.3e-10) and a degree CRS (bounds ~1e1, footprints ~4.7e-3).
        tol = 1e-6 * float(np.min(R - L))
        for i, ti in enumerate(ids):
            ox = np.minimum(R[i], R) - np.maximum(L[i], L)
            oy = np.minimum(T[i], T) - np.maximum(B[i], B)
            hit = (ox > tol) & (oy > tol)
            hit[i] = False
            for j in np.where(hit)[0]:
                if kept[ids[j]] != kept[ti]:
                    bad.append((ti, kept[ti], ids[j], kept[ids[j]]))
    return bad


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restarts", type=int, default=400,
                    help="seeded random block assignments to search over")
    ap.add_argument("--distinct-from", nargs="*", default=None, metavar="MANIFEST",
                    help="reject candidates whose test blocks match any of these manifests at any "
                         "site, so the assignments are genuinely different replicates")
    ap.add_argument("--from-manifest", default=None,
                    help="skip the search and materialise exactly the assignment in this manifest. "
                         "Use to reproduce a split on another machine (e.g. the cluster) without "
                         "re-deriving it, so the two are guaranteed identical.")
    ap.add_argument("--three-region", metavar="SITE", default=None,
                    help="SHIPPED DESIGN: cut this site into train|val|test strips with two straight "
                         "cuts, holding --external-sites out whole. Searches cut placement for the "
                         "candidate with the least composition drift, rejecting any that leaves a class "
                         "with under --min-class-blocks independent blocks in a split.")
    ap.add_argument("--buffer-test-m", type=float, default=650.0,
                    help="minimum separation between the validation and test strips, in metres. "
                         "650 m = 2.5x the 256 m tile footprint, so no tile can share a pixel across "
                         "the boundary. It is NOT above the measured autocorrelation range: the "
                         "inland site measures 750 m for composition and 1,350 m for spectral "
                         "similarity (artifacts/correlogram/, 900 of 1,952 tiles). The buffer keys "
                         "on effect size, not range -- peak Mantel r is 0.044 and stable across "
                         "every subsample, seed and permutation count tried -- and is anchored on "
                         "the pixel-identity geometry. What reaches the reader is the REALISED "
                         "separation, 1,664 m train-test, which is the guarantee the test number "
                         "rests on.")
    ap.add_argument("--buffer-val-m", type=float, default=256.0,
                    help="minimum separation between the train and validation strips, in metres. "
                         "256 m is the exact pixel-identity distance: tiles are 512 px at 0.5 m on a "
                         "128 m stride, so beyond 256 m no two tiles share a pixel. Validation only "
                         "selects checkpoints and is never reported, so it needs no more than this.")
    ap.add_argument("--min-val-tiles", type=int, default=100,
                    help="minimum validation tiles. Lower than --min-test-tiles by design: "
                         "validation selects checkpoints and is never reported.")
    ap.add_argument("--min-test-tiles", type=int, default=150,
                    help="minimum test tiles. This split carries the reported estimate.")
    ap.add_argument("--min-test-class-blocks", type=int, default=8,
                    help="stricter block minimum for the TEST split, whose per-class numbers are the "
                         "ones reported. Defaults above --min-class-blocks deliberately.")
    ap.add_argument("--min-class-blocks", type=int, default=MIN_CLASS_BLOCKS,
                    help=f"reject a candidate if any foreground class has fewer than this many "
                         f"distinct {SUPPORT_BLOCK_M:.0f} m blocks in train, val or test. This is the "
                         f"binding adequacy criterion; the share floor is a weak safety net.")
    ap.add_argument("--external-sites", nargs="*", default=None, metavar="SITE",
                    help="hold these sites out WHOLE as a fixed external test set, and block the "
                         "remaining sites three ways. Use when a site's extent is under twice its "
                         "autocorrelation range, so no independent internal split exists.")
    ap.add_argument("--out", default="artifacts/spatial_split_manifest.json")
    ap.add_argument("--materialise", action="store_true",
                    help="write data/biodiversity_split_spatial/{train,val,test}")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--out-root", default="data/biodiversity_split_spatial")
    args = ap.parse_args()

    root = find_repo_root()
    split_root = root / "data/biodiversity_split"
    cache_dir = root / "artifacts/_cache"

    print("reading tile footprints ...", flush=True)
    pool = read_pool(split_root, cache_dir / "tile_bounds_pool.json")
    print(f"  {len(pool)} tiles across {len(set(map(site_of, pool)))} sites")

    if args.from_manifest:
        src = json.loads((root / args.from_manifest).read_text())
        kept = dict(src["assignment"])
        missing = set(kept) - set(pool)
        if missing:
            raise SystemExit(f"manifest names {len(missing)} tiles absent from {split_root}, "
                             f"e.g. {sorted(missing)[:3]}")
        print(f"replaying {args.from_manifest}: {len(kept)} tiles "
              f"({dict(Counter(kept.values()))})", flush=True)
        print("verifying (independent re-read of the GeoTIFF geometry) ...", flush=True)
        bad = verify_independent(kept, root, split_root, {t: pool[t][0] for t in kept})
        if bad:
            raise SystemExit(f"FAILED: {len(bad)} cross-split overlaps, e.g. {bad[:3]}")

        # Replay used to check overlap ONLY, so a manifest with 384 m separation against a 650 m
        # requirement, or with zero validation tiles, replayed clean. Zero overlap is far weaker than
        # the buffer the design claims. Everything below is recomputed from the raw geometry and
        # checked against what the manifest itself declares, so a doctored manifest fails.
        got_counts = Counter(kept.values())
        for sp in SPLITS:
            if got_counts.get(sp, 0) == 0:
                raise SystemExit(f"FAILED: manifest has zero {sp} tiles")
        declared = src.get("counts") or {}
        for sp, n in declared.items():
            if got_counts.get(sp, 0) != n:
                raise SystemExit(f"FAILED: manifest declares {n} {sp} tiles but its assignment holds "
                                 f"{got_counts.get(sp, 0)}")
        # Recompute adequacy from the MASKS and check it against what the manifest declares. Without
        # this the class-share and block-support criteria bind only at derivation time: RUNBOOK.sh
        # exits unless a committed manifest exists and then only ever calls --from-manifest, so a
        # hand-edited manifest with every validation class collapsed into one 950 m block replays and
        # materialises without objection. `class_counts` is called AFTER the return below, so the
        # masks were never even opened on this path.
        declared_sup = src.get("class_block_support")
        if declared_sup:
            rcounts = class_counts(split_root, pool, cache_dir / "tile_class_counts.json")
            rblocks = support_blocks(pool, src.get("support_block_m", SUPPORT_BLOCK_M))
            actual = class_block_support(kept, rcounts, rblocks)
            need_b = {"train": src.get("min_class_blocks", MIN_CLASS_BLOCKS),
                      "val": src.get("min_class_blocks", MIN_CLASS_BLOCKS),
                      "test": src.get("min_test_class_blocks",
                                      src.get("min_class_blocks", MIN_CLASS_BLOCKS))}
            for sp in SPLITS:
                for c in FOREGROUND:
                    got_b, want_b = actual[sp][c], need_b[sp]
                    if got_b < want_b:
                        raise SystemExit(f"FAILED: {sp}/{FOREGROUND[c]} occupies {got_b} independent "
                                         f"blocks, below the {want_b} this manifest requires")
                    if declared_sup.get(sp, {}).get(str(c), got_b) != got_b:
                        raise SystemExit(
                            f"FAILED: manifest declares {declared_sup[sp][str(c)]} blocks for "
                            f"{sp}/{FOREGROUND[c]} but the masks give {got_b}")
            print(f"  VERIFIED block support recomputed from masks: min "
                  f"{min(actual[sp][c] for sp in SPLITS for c in FOREGROUND)} blocks")
        else:
            # Recompute rather than warn. A manifest that omits the key used to waive the very
            # criterion that admitted the split -- a val set collapsed to one block replayed clean,
            # materialised, and then passed the leakage gate too. Adequacy is now enforced on the
            # shipped replay path, which is the only path anyone runs.
            print("  manifest declares no class_block_support; recomputing it from the masks")
            rcounts = class_counts(split_root, pool, cache_dir / "tile_class_counts.json")
            rblocks = support_blocks(pool, src.get("support_block_m", SUPPORT_BLOCK_M))
            actual = class_block_support(kept, rcounts, rblocks)
            need_b = {"train": src.get("min_class_blocks", MIN_CLASS_BLOCKS),
                      "val": src.get("min_class_blocks", MIN_CLASS_BLOCKS),
                      "test": src.get("min_test_class_blocks",
                                      src.get("min_class_blocks", MIN_CLASS_BLOCKS))}
            bad = [f"{s}/{FOREGROUND[c]}={n}<{need_b[s]}" for s in need_b
                   for c, n in actual.get(s, {}).items() if n < need_b[s]]
            if bad:
                raise SystemExit(
                    "  FAIL recomputed class_block_support is below the criterion this design "
                    f"requires: {', '.join(bad)}")
            print(f"  VERIFIED recomputed block support: min "
                  f"{min(n for s in need_b for n in actual.get(s, {}).values())} blocks")

        need_m = {tuple(k.split("-")): v for k, v in (src.get("required_separation_m") or {}).items()}
        if need_m:
            sep = pairwise_separation_m(kept, root, split_root, {t: pool[t][0] for t in kept}, src.get("site_split", "biodiversity"))
            for pair, need in need_m.items():
                if pair in sep and sep[pair] + 1e-6 < need:
                    raise SystemExit(f"FAILED: {pair[0]}-{pair[1]} separation {sep[pair]:.0f} m is "
                                     f"below the {need:.0f} m this manifest requires")
            print("  VERIFIED separation: " + ",  ".join(
                f"{a}-{b} {sep[(a, b)]:.0f} m (>= {need_m[(a, b)]:.0f})"
                for (a, b) in need_m if (a, b) in sep))
        else:
            print("  WARNING: manifest declares no required_separation_m, so replay cannot check it")
        print("VERIFIED: zero cross-split footprint overlap")
        if args.materialise:
            materialise(kept, pool, root, split_root, args.out_root, args.mode)
            print(f"materialised {args.out_root} ({args.mode})")
        return

    print("counting mask classes ...", flush=True)
    counts = class_counts(split_root, pool, cache_dir / "tile_class_counts.json")


    external = tuple(args.external_sites) if args.external_sites else ()
    if external:
        unknown = set(external) - {site_of(t) for t in pool}
        if unknown:
            raise SystemExit(f"unknown site(s) for --external-sites: {sorted(unknown)}")
        print(f"EXTERNAL TEST SITES (held out whole, outside the blocked design): {external}",
              flush=True)

    if args.three_region:
        priors = []
        for m in (args.distinct_from or []):
            prior = json.loads((root / m).read_text())["assignment"]
            priors.append(frozenset(t for t, sp in prior.items() if sp == "test"))
            print(f"  requiring different test ground from {m} ({len(priors[-1])} tiles)", flush=True)

        print(f"searching {args.restarts} cut placements "
              f"(val|test buffer {args.buffer_test_m:.0f} m, train|val {args.buffer_val_m:.0f} m) ...",
              flush=True)
        out = build_three_region(pool, counts, args.three_region, external,
                                 args.buffer_test_m, args.buffer_val_m, args.seed,
                                 args.restarts, priors,
                                 min_val_tiles=args.min_val_tiles,
                                 min_test_tiles=args.min_test_tiles,
                                 min_class_blocks=args.min_class_blocks,
                                 min_test_class_blocks=args.min_test_class_blocks)
        kept, dropped, geom, shares = (out["assignment"], out["dropped"],
                                       out["geometry"], out["class_shares"])
        got = Counter(kept.values())

        # Independent check: recompute the realised separation from the raw GeoTIFF geometry, not
        # from the band arithmetic that produced the split. Must meet the requested guarantee.
        print("verifying separation from the raw geometry ...", flush=True)
        sep = pairwise_separation_m(kept, root, split_root, {t: pool[t][0] for t in kept}, args.three_region)
        required = {("train", "val"): args.buffer_val_m,
                    ("train", "test"): args.buffer_test_m,
                    ("val", "test"): args.buffer_test_m}
        for pair, need in required.items():
            got_sep = sep[pair]
            if not (got_sep + 1e-6 >= need):
                raise SystemExit(f"FAILED: {pair[0]}-{pair[1]} separation is {got_sep:.0f} m, "
                                 f"below the required {need:.0f} m")
        bad = verify_independent(kept, root, split_root, {t: pool[t][0] for t in kept})
        if bad:
            raise SystemExit(f"FAILED: {len(bad)} cross-split footprint overlaps, e.g. {bad[:3]}")

        print(f"\nkept {len(kept)}/{len(pool)} tiles  (dropped {len(dropped)} to the buffers, "
              f"{100*len(dropped)/len(pool):.1f}%)")
        for sp in list(SPLITS) + [EXTERNAL]:
            if got.get(sp):
                line = ("  " + "  ".join(f"{c}={100*shares[sp][c]:.1f}%" for c in shares[sp])
                        if sp in SPLITS else "  (held out whole; generalisation test set)")
                print(f"  {sp:13s} {got[sp]:5d} tiles{line}")
        print(f"  cut geometry: axis={geom['axis']} roles={geom['roles']}")
        print(f"  worst per-class share in any split: {100*out['worst_class_share']:.1f}%")
        for pair, need in required.items():
            print(f"  VERIFIED {pair[0]:5s}-{pair[1]:5s} separation {sep[pair]:7.0f} m "
                  f"(required {need:.0f} m)")
        print("  VERIFIED: zero cross-split footprint overlap")

        manifest = {
            "generated_by": "scripts/data_prep/build_spatial_split.py --three-region",
            "design": "three-region single-axis cuts + whole-site external test",
            "seed": args.seed, "restarts": args.restarts,
            "site_split": args.three_region, "external_sites": list(external),
            "buffer_test_m": args.buffer_test_m, "buffer_val_m": args.buffer_val_m,
            "realised_separation_m": {f"{a}-{b}": v for (a, b), v in sep.items()},
            "required_separation_m": {f"{a}-{b}": v for (a, b), v in required.items()},
            "pixel_identity_distance_m": 256.0,
            "cut_geometry": geom,

            "min_class_blocks": out["min_class_blocks"],
            "min_test_class_blocks": args.min_test_class_blocks,
            "support_block_m": out["support_block_m"],
            "class_block_support": out["class_block_support"],
            "composition_drift": geom.get("composition_drift"),
            "n_pool": len(pool), "n_kept": len(kept), "n_dropped_buffer": len(dropped),
            "counts": {sp: got.get(sp, 0) for sp in list(SPLITS) + [EXTERNAL]},
            "class_shares": shares, "worst_class_share": out["worst_class_share"],
            "cross_split_overlaps": 0,
            "verification": "separation and overlap recomputed from GeoTIFF bounds, independently "
                            "of the band arithmetic that produced the split",
            "assignment": kept, "dropped_buffer": sorted(dropped),
        }
        (root / args.out).parent.mkdir(parents=True, exist_ok=True)
        (root / args.out).write_text(json.dumps(manifest, indent=2))
        print(f"wrote {args.out}")
        if args.materialise:
            materialise(kept, pool, root, split_root, args.out_root, args.mode)
            print(f"materialised {args.out_root} ({args.mode})")
        return


if __name__ == "__main__":
    main()
