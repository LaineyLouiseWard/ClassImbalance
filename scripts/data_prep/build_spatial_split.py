#!/usr/bin/env python3
"""
Build a spatially blocked train/val/test split with a leakage buffer.

WHY: the Biodiversity tiles are chipped on a 50% stride (256 x 256 m footprints on a 128 m grid),
so every tile overlaps its eight neighbours. The original split
(`scripts/data_prep/split_biodiversity_dataset.py`) assigns tiles at random, which puts ~93% of each
held-out tile's ground area inside a training tile -- identical pixels and identical labels. See
notes/TILE_OVERLAP_LEAKAGE_2026-07-25.md.

WHAT THIS DOES:
  1. Repools every tile from the existing split directories and reads its footprint from the
     GeoTIFF transform. The three sites use two coordinate systems, so geometry is only ever
     compared within a site.
  2. Cuts ONE site into train | val | test strips along a single axis, searching cut placements for
     the one with the least composition drift and the closest match to the target proportions, and
     rejecting any that leaves a foreground class below MIN_CLASS_SHARE. Sites too small to cut are
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

SPLITS = ("train", "val", "test")
# A fourth destination, outside the blocked design. Sites whose extent is smaller than twice their
# autocorrelation range cannot be split internally at all (Roberts et al. 2017), so they are held
# out whole as a fixed external test set: one model, two questions -- accuracy on new ground inside
# a surveyed area (test) and accuracy on terrain never labelled (external_test).
EXTERNAL = "external_test"
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}
FOREGROUND = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}

# A split must hold at least this share of its foreground pixels in each class, or the candidate
# assignment is REJECTED (train/val/test only -- see the note in build_three_region on why
# external_test is outside the search space rather than exempted from the floor).
#
# History, because the value is a judgement and its cost was measured rather than assumed.
# 0.002 was far too weak: it admitted an assignment whose validation set held semi-natural at 1.3%
# of foreground, which then drove checkpoint selection on noise in the priority class. Raised to 0.03
# on 2026-07-25 -- but only inside score(), on the block-grid path, which the shipped three-region
# branch returns before reaching. So nothing was enforced, and all three folds shipped in breach
# (worst 1.55%, f3's validation set at 1.87% semi-natural).
#
# Enforcing it by rejection on 2026-07-26 exposed a hard trade-off. Cropland and settlement are
# spatially clustered inland, so with straight cuts the floor trades directly against the number of
# DISTINCT held-out regions. Measured over 12,000 cut placements per fold, with the adequacy minima
# and the drift/proportion ranking all in force (an earlier table taken WITHOUT them was optimistic
# and is not repeated here -- it suggested three folds at 2%, which do not exist):
#
#     floor    distinct folds    test sizes         binding class
#     2.0%           2           166, 405           test Cropland
#     1.8%           2           166, 405           test Cropland
#     1.5%           3           166, 405, 185      test Cropland 1.8%
#
# A third fold was confirmed infeasible at 2.0% across five seeds and 60,000 placements, so this is
# structural, not a search failure.
#
# 0.015 is chosen deliberately. It dominates what actually shipped on every axis -- worst share
# 1.55% -> 1.8%, test sets 103-125 -> 166-405 tiles, validation 79-176 -> 143-327 -- while keeping the
# three held-out regions that the generalisation claim and the within-fold blocked analysis both
# depend on. A 2% floor would buy 0.2 pp on the thinnest class at the cost of a whole region.
#
# The classes still thin here are not silently accepted. Per-class support (pixels, tiles and
# distinct ground blocks) is reported beside every per-class IoU, and classes too thin to interpret
# are marked unestimable rather than given an estimate. That is the same discipline applied to
# external_test, where the floor is inapplicable rather than waived: the uplands are held out whole
# and identical across folds, so there is no candidate to reject and no alternative to select.
MIN_CLASS_SHARE = 0.015
# Every site must contribute at least this many tiles to every split. One or two tiles satisfies a
# bare presence test while being far too few to estimate anything from.
MIN_SITE_TILES = 4


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
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

def assign_by_cuts(pool: dict, site: str, axis: str, fractions: tuple, order: tuple) -> dict:
    """Cut one site into three strips along `axis` and give each strip a role.

    Two straight cuts, three regions. Preferred over a block grid because each cut costs one
    buffered boundary, and boundaries are the scarce resource: the inland site is only ~6
    autocorrelation lengths across, so a 3x3 grid (which needs boundaries on all sides of every
    block) destroys 49% of the tiles where two cuts destroy 19%.

    `fractions` are quantiles of the tile centroids along the axis, so the strips hold roughly the
    intended share of tiles regardless of how the site is shaped or where its gaps are.
    """
    ids = [t for t in pool if site_of(t) == site]
    coord = {t: ((pool[t][1] + pool[t][3]) / 2 if axis == "x" else (pool[t][2] + pool[t][4]) / 2)
             for t in ids}
    vals = np.array([coord[t] for t in ids])
    c1, c2 = (float(np.quantile(vals, f)) for f in fractions)
    out = {}
    for t in ids:
        v = coord[t]
        out[t] = order[0] if v <= c1 else (order[1] if v <= c2 else order[2])
    return out


def apply_cut_buffer(assign: dict, pool: dict, site: str, axis: str,
                     cuts: tuple, buffer_m: float, geographic: bool, lat: float) -> tuple:
    """Drop tiles whose footprint intersects a band of width `buffer_m` centred on each cut.

    The guarantee this delivers is the one that matters: **the minimum ground separation between any
    surviving tile on one side of a cut and any on the other is at least `buffer_m`.**

    Deliberately NOT the pairwise rule used for the block designs (`apply_buffer`), which drops every
    tile within `buffer_m` of a differently-assigned tile and therefore clears `buffer_m` from BOTH
    sides -- a 2 x buffer band. With two cuts that removes 2,600 m of a 6,784 m axis and erases any
    strip narrower than 1,300 m, which is what silently emptied the val and test strips on the first
    attempt. A centred band of width `buffer_m` achieves the stated separation at half the cost.
    """
    half = buffer_m / 2.0
    if geographic:
        half_x = half / (111320.0 * math.cos(math.radians(lat)))
        half_y = half / 111132.0
    else:
        half_x = half_y = half
    pad = half_x if axis == "x" else half_y
    dropped = set()
    for t in [k for k in assign if site_of(k) == site]:
        lo, hi = (pool[t][1], pool[t][3]) if axis == "x" else (pool[t][2], pool[t][4])
        for c in cuts:
            if lo < c + pad and hi > c - pad:      # footprint intersects the band
                dropped.add(t)
                break
    return {t: s for t, s in assign.items() if t not in dropped}, dropped


def pairwise_separation_m(kept: dict, pool: dict, site: str,
                          geographic: bool = False, lat: float = 0.0) -> dict:
    """{(split_a, split_b): minimum footprint-to-footprint distance in metres}.

    Per PAIR, not a global minimum: this design separates its boundaries by different amounts on
    purpose (train|val by the pixel-identity distance, val|test by the full buffer), so a single
    global minimum is meaningless and comparing it to the largest requirement fails spuriously.

    Recomputed from raw GeoTIFF bounds, independently of the band arithmetic that produced the split,
    so it cannot pass by construction.
    """
    ids = [t for t in kept if site_of(t) == site and kept[t] in SPLITS]
    arr = np.array([pool[t][1:5] for t in ids], float)
    lab = np.array([kept[t] for t in ids])
    L, B, R, T = arr.T
    mx, my = ((111320.0 * math.cos(math.radians(lat)), 111132.0) if geographic else (1.0, 1.0))
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
                       priors: list, min_tiles: int, max_test_overlap: float = 0.25,
                       min_class_share: float = MIN_CLASS_SHARE,
                       min_val_tiles: int | None = None,
                       min_test_tiles: int | None = None) -> dict:
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
    n_below_floor = 0

    for _ in range(restarts):
        axis = "x" if rng.random() < 0.5 else "y"
        train_low = rng.random() < 0.5          # wide train strip at the low end of the axis?
        # Strip widths as fractions of the site's extent along the chosen axis. Ranges widened
        # 2026-07-26: with f_val clipped at 0.20 the search space contained NO placement satisfying
        # MIN_CLASS_SHARE on all three splits, because the inland site's cropland and settlement are
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
        need = {"train": min_tiles,
                "val": min_val_tiles if min_val_tiles is not None else min_tiles,
                "test": min_test_tiles if min_test_tiles is not None else min_tiles}
        if any(got.get(sp, 0) < need[sp] for sp in SPLITS):
            continue
        # Each fold must hold out substantially different ground from every earlier fold. Bounded
        # overlap, not disjointness: with straight cuts only TWO fully disjoint test strips exist
        # along one axis, and a third fold must cut the other axis, whose strip necessarily contains
        # tiles from both of the first two. Demanding disjointness therefore caps the design at two
        # folds. Two strips on perpendicular axes intersect only in a corner -- for ~16% strips that
        # is ~2.6% of the site -- so a small cap admits four distinct test regions (one per edge)
        # while still bounding how much they share. The realised overlap is recorded in the manifest.
        test_tiles = frozenset(t for t, sp in kept.items() if sp == "test")
        overlaps = [len(test_tiles & prior) / max(1, min(len(test_tiles), len(prior)))
                    for prior in priors]
        if overlaps and max(overlaps) > max_test_overlap:
            continue

        shares = split_class_shares(kept, counts)
        worst = min(shares[sp][c] for sp in SPLITS for c in shares[sp])

        # REJECT below the floor. Until 2026-07-26 this path only *ranked* by worst share and never
        # rejected, because the floor lived in score(), which the three-region branch returns before
        # reaching. All three shipped folds breached it: f3's validation set held semi-natural at
        # 1.87%, in the priority class, in the split that selects checkpoints -- the exact failure the
        # comment on MIN_CLASS_SHARE says was fixed.
        #
        # SPLITS is (train, val, test), so this deliberately does not touch external_test. The floor is
        # a rejection criterion on a search, and the uplands are not in the search space: they are held
        # out whole and identical across every fold, so there is no candidate to reject and no
        # alternative to select. Settlement at 1.47% of upland foreground is a fact about that
        # landscape, not a property of a split choice. The floor's purpose still applies to Test B, but
        # it can only be honoured there by reporting per-class support and marking the thin classes
        # unestimable -- see report_test_b_support.py.
        if worst < min_class_share:
            n_below_floor += 1
            continue

        # Composition drift against the whole-site prior. Also confined to the dead path until now,
        # which let f2 ship with 13.84% cropland in validation against 2.13% in its own test set -- a
        # 6.5x ratio, so its checkpoint was selected against a prior neither its training nor its test
        # set shared. Ranked ahead of worst-share: once the floor is a hard reject, every surviving
        # candidate is adequate, and what remains to optimise is comparability between the splits.
        pooled = split_class_shares({t: "train" for t in kept}, counts)["train"]
        # Drift is measured on TEST against the whole-site prior, and on train, but NOT on val.
        # The three splits do different jobs and weighting them equally optimises the wrong thing.
        # Test carries the reported estimate, so it must look like the landscape the paper claims to
        # generalise within. Train must too, or the model is fitted to a skewed prior. Validation only
        # selects checkpoints and confirms convergence: it is never reported as an accuracy, and
        # whatever optimism or skew it carries is common-mode across all four factorial cells, so it
        # cannot manufacture a between-cell effect. Its requirements are the class floor (so the
        # priority class is not selected on noise -- the original 1.3% semi-natural failure) and enough
        # tiles to be stable, both already hard constraints above. Forcing its composition as well is
        # what left every candidate either thin in test or 44% short of training data.
        drift = sum(abs(shares[sp][c] - pooled[c]) for sp in ("train", "test") for c in pooled)

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
            f"no viable cut placement in {restarts} tries ({n_below_floor} rejected for holding a "
            f"foreground class below min_class_share={min_class_share:.2%} in some split; the rest "
            f"left under {min_tiles} tiles in a split or duplicated an earlier fold's test ground). "
            f"Raise --restarts, widen the strip fractions, or lower --min-split-tiles.")
    _, geom, kept, dropped, worst, shares = best
    return {"assignment": kept, "dropped": dropped, "geometry": geom,
            "worst_class_share": worst, "class_shares": shares}


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
                         "candidate with the best worst-case per-class share.")
    ap.add_argument("--buffer-test-m", type=float, default=650.0,
                    help="minimum separation between the validation and test strips, in metres. "
                         "650 m = 2.5x the 256 m tile footprint (so no tile can share a pixel across "
                         "the boundary) and above the measured autocorrelation range in tile class "
                         "composition. This is the guarantee the reported test number rests on.")
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
    ap.add_argument("--min-class-share", type=float, default=MIN_CLASS_SHARE,
                    help="reject any candidate holding a foreground class below this share of "
                         "foreground pixels in train, val or test. Not applied to external_test, "
                         "which is held out whole and identical across folds, so there is no "
                         "candidate to reject; its thin classes are handled by reporting support "
                         "instead (see report_test_b_support.py).")
    ap.add_argument("--max-test-overlap", type=float, default=0.25,
                    help="largest share of test tiles a fold may share with any earlier fold. "
                         "Straight cuts admit only two fully disjoint test strips per axis, so a "
                         "third fold must cut the other axis and unavoidably shares a corner.")
    ap.add_argument("--min-split-tiles", type=int, default=60,
                    help="reject a cut placement leaving fewer than this in any split")
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

    priors = []
    for m in (args.distinct_from or []):
        prior_assign = json.loads((root / m).read_text())["assignment"]
        priors.append(test_blocks_by_site(prior_assign, blocks))
        print(f"  requiring different held-out ground from {m}: "
              f"{ {s: sorted(v) for s, v in priors[-1].items()} }", flush=True)

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
                                 args.restarts, priors, args.min_split_tiles,
                                 args.max_test_overlap, args.min_class_share,
                                 args.min_val_tiles, args.min_test_tiles)
        kept, dropped, geom, shares = (out["assignment"], out["dropped"],
                                       out["geometry"], out["class_shares"])
        got = Counter(kept.values())

        # Independent check: recompute the realised separation from the raw GeoTIFF geometry, not
        # from the band arithmetic that produced the split. Must meet the requested guarantee.
        print("verifying separation from the raw geometry ...", flush=True)
        sep = pairwise_separation_m(kept, pool, args.three_region)
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
            "max_test_overlap_allowed": args.max_test_overlap,
            # Both thresholds the search actually applied. min_class_share is a rejection criterion on
            # train/val/test only; external_test is outside the search space (held out whole, identical
            # across folds), so it is neither ranked nor rejected on composition.
            "min_class_share": args.min_class_share,
            "min_class_share_applies_to": list(SPLITS),
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
