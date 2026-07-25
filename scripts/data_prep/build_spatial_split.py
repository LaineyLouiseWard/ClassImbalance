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
  2. Lays a coarse block grid over each site and assigns whole BLOCKS to train/val/test, searching
     seeded random assignments for the one that best matches the target tile proportions while
     keeping every foreground class present in every split.
  3. Applies a buffer: any tile whose footprint overlaps a tile assigned to a DIFFERENT split is
     dropped from the dataset entirely. Symmetric, so no split is trimmed preferentially.
  4. Verifies zero cross-split footprint overlap, and reports retained counts and per-split class
     composition.

Blocks are laid within each site rather than holding whole sites out because the two designs answer
different questions, not because leave-one-site-out is infeasible. Semi-natural grassland is ~74% of
foreground pixels at ireland2 and ~27% at ireland1 against ~5% inland, but the inland site is so much
larger that it still holds 57.8% of the dataset's semi-natural pixel MASS (25.5M of 44.1M) across 614
tiles, so LOSO leaves the priority class well represented in training. It is run here as a secondary
transfer experiment (--loso). Blocking within sites estimates accuracy on unlabelled ground inside a
surveyed landscape; LOSO estimates transfer to an unsurveyed region. Do not conflate prevalence with
mass when justifying either.

Output: artifacts/spatial_split_manifest.json (tile -> split or 'dropped', plus the audit).
Materialising the directories is a separate step (--materialise).

Run:
    PYTHONPATH=. python scripts/data_prep/build_spatial_split.py
    PYTHONPATH=. python scripts/data_prep/build_spatial_split.py --materialise --mode symlink
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

SPLITS = ("train", "val", "test")
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}
FOREGROUND = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}

# Block grid per site (columns x rows of blocks over that site's tile grid). Blocks must stay
# comfortably larger than the buffer rim: the buffer drops roughly one tile cell from each block
# edge, so a block only 2x2 cells across is erased entirely. The inland tile grid is 47x52 cells,
# ireland1 is 8x10 and ireland2 is 9x19, which caps how finely the small sites can be cut.
BLOCK_GRID = {
    "biodiversity": (4, 4),
    "ireland1": (2, 2),
    "ireland2": (2, 3),
}
# A split must hold at least this share of its foreground pixels in each class, or the candidate
# assignment is rejected. Guards against a split losing semi-natural or cropland entirely.
MIN_CLASS_SHARE = 0.002
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


def block_index(pool: dict, site: str, ncols: int, nrows: int) -> dict:
    """Map each tile of `site` to a block id (bx, by) over that site's bounding box."""
    ids = [k for k in pool if site_of(k) == site]
    L = np.array([pool[k][1] for k in ids])
    B = np.array([pool[k][2] for k in ids])
    x0, x1 = L.min(), L.max()
    y0, y1 = B.min(), B.max()
    # nextafter so the maximum origin lands inside the last block rather than one past it
    bx = np.minimum((((L - x0) / (np.nextafter(x1 - x0, np.inf))) * ncols).astype(int), ncols - 1)
    by = np.minimum((((B - y0) / (np.nextafter(y1 - y0, np.inf))) * nrows).astype(int), nrows - 1)
    return {k: (int(bx[i]), int(by[i])) for i, k in enumerate(ids)}


def overlapping_pairs(pool: dict, ids: list[str], buffer_widths: float = 0.0) -> dict:
    """{tile: set(tiles within the exclusion distance)}, computed within each site.

    buffer_widths extends the test beyond bare pixel sharing, in units of one tile width (256 m
    here). Zero removes only identity leakage; a positive value also excludes tiles that share no
    pixels but sit inside the spatial-autocorrelation range of a differently-assigned tile.
    Expressed in tile widths so one setting is meaningful in both a metre and a degree CRS.
    """
    nbr = defaultdict(set)
    by_site = defaultdict(list)
    for k in ids:
        by_site[site_of(k)].append(k)
    for site, keys in by_site.items():
        arr = np.array([pool[k][1:5] for k in keys], float)
        L, B, R, T = arr.T
        w = float(np.min(R - L))
        # Relative to footprint size: an absolute epsilon cannot serve both a metre CRS and a
        # degree CRS. Erring large here only widens the buffer, which is the safe direction.
        tol = 1e-6 * w
        pad = buffer_widths * w
        for i, ki in enumerate(keys):
            ox = np.minimum(R[i], R) - np.maximum(L[i], L) + pad
            oy = np.minimum(T[i], T) - np.maximum(B[i], B) + pad
            hit = (ox > tol) & (oy > tol)
            hit[i] = False
            for j in np.where(hit)[0]:
                nbr[ki].add(keys[j])
    return nbr


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
def propose_loso(pool: dict, blocks: dict, rng: np.random.Generator, holdout: tuple) -> dict:
    """Leave-one-site-out: whole sites become test; the rest is blocked into train/val.

    Secondary transfer experiment. Asks whether the class concepts carry to a landscape type the
    model never trained on, which is the question ODOS faces when scaling to new regions. The
    inland site holds 57.8% of the dataset's semi-natural pixels, so holding both upland sites out
    still leaves the priority class well represented in training.
    """
    assign = {}
    for site in BLOCK_GRID:
        site_tiles = [k for k in pool if site_of(k) == site]
        if site in holdout:
            for k in site_tiles:
                assign[k] = "test"
            continue
        per_block = defaultdict(list)
        for k in site_tiles:
            per_block[blocks[k]].append(k)
        bids = list(per_block)
        rng.shuffle(bids)
        want = {"train": 0.9 * len(site_tiles), "val": 0.1 * len(site_tiles)}
        got = {"train": 0, "val": 0}
        for bid in bids:
            s = max(want, key=lambda x: (want[x] - got[x]) / want[x])
            for k in per_block[bid]:
                assign[k] = s
            got[s] += len(per_block[bid])
    return assign


def propose(pool: dict, blocks: dict, rng: np.random.Generator) -> dict:
    """Assign whole blocks to splits, per site, roughly in TARGET proportion by tile count."""
    assign = {}
    for site in BLOCK_GRID:
        site_tiles = [k for k in pool if site_of(k) == site]
        per_block = defaultdict(list)
        for k in site_tiles:
            per_block[blocks[k]].append(k)
        bids = list(per_block)
        rng.shuffle(bids)
        n_site = len(site_tiles)
        want = {s: TARGET[s] * n_site for s in SPLITS}
        got = {s: 0 for s in SPLITS}
        # Award each block to the split with the largest RELATIVE shortfall. An absolute-deficit
        # rule would hand every early block to train (its deficit starts at 0.8n) and starve the
        # held-out splits: with the small sites that left test empty of coastal tiles entirely.
        for bid in bids:
            s = max(SPLITS, key=lambda x: (want[x] - got[x]) / want[x])
            for k in per_block[bid]:
                assign[k] = s
            got[s] += len(per_block[bid])
    return assign


def score(assign: dict, pool: dict, counts: dict, loso: tuple = ()) -> tuple[float, bool]:
    """Deviation from target proportions; and whether the split is scientifically usable.

    Usable means every foreground class clears MIN_CLASS_SHARE in every split AND every site
    contributes tiles to every split. Without the site check a valid-looking assignment can leave
    the test set entirely inland, which strips the priority semi-natural class out of it.
    """
    n = len(assign)
    got = Counter(assign.values())
    dev = sum(abs(got[s] / n - TARGET[s]) for s in SPLITS)

    shares = split_class_shares(assign, counts)
    pooled = split_class_shares({t: "train" for t in assign}, counts)["train"]
    # Penalise class composition drifting between splits: a val set holding 0.9% cropland against a
    # test set holding 12% makes the two incomparable and the per-class intervals meaningless.
    imbalance = sum(abs(shares[s][c] - pooled[c]) for s in SPLITS for c in pooled)

    class_ok = all(shares[s][FOREGROUND[c]] >= MIN_CLASS_SHARE for s in SPLITS for c in FOREGROUND)
    per_site = Counter((site_of(t), s) for t, s in assign.items())
    if loso:
        # Held-out sites are test-only and the remaining sites never reach test, both by design.
        site_ok = all(per_site[(site, s)] >= MIN_SITE_TILES
                      for site in BLOCK_GRID if site not in loso for s in ("train", "val"))
    else:
        site_ok = all(per_site[(site, s)] >= MIN_SITE_TILES for site in BLOCK_GRID for s in SPLITS)
    return dev + imbalance, (class_ok and site_ok)


def apply_buffer(assign: dict, nbr: dict) -> tuple[dict, set]:
    """Drop every tile that shares ground with a tile in a different split."""
    dropped = {t for t, s in assign.items()
               if any(assign.get(o) not in (None, s) for o in nbr.get(t, ()))}
    return {t: s for t, s in assign.items() if t not in dropped}, dropped


def materialise(kept: dict, pool: dict, root: Path, split_root: Path, out_root: str, mode: str):
    """Write out_root/{train,val,test}/{images,masks} for the kept assignment."""
    out = root / out_root
    if out.exists():
        shutil.rmtree(out)
    for s in SPLITS:
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
    ap.add_argument("--from-manifest", default=None,
                    help="skip the search and materialise exactly the assignment in this manifest. "
                         "Use to reproduce a split on another machine (e.g. the cluster) without "
                         "re-deriving it, so the two are guaranteed identical.")
    ap.add_argument("--loso", nargs="*", default=None, metavar="SITE",
                    help="leave-one-site-out mode: hold these whole sites out as test "
                         "(e.g. --loso ireland1 ireland2). Secondary transfer experiment.")
    ap.add_argument("--buffer-widths", type=float, default=0.0,
                    help="exclusion distance between splits, in tile widths (1.0 = 256 m). "
                         "0 removes only pixel-sharing; >0 also buffers spatial autocorrelation.")
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

    blocks = {}
    for site, (nc, nr) in BLOCK_GRID.items():
        blocks.update(block_index(pool, site, nc, nr))
    print(f"  block grid: { {s: f'{c}x{r}' for s, (c, r) in BLOCK_GRID.items()} }")

    print(f"precomputing exclusion graph (buffer = {args.buffer_widths} tile widths) ...", flush=True)
    nbr = overlapping_pairs(pool, list(pool), args.buffer_widths)

    loso = tuple(args.loso) if args.loso else ()
    if loso:
        unknown = set(loso) - set(BLOCK_GRID)
        if unknown:
            raise SystemExit(f"unknown site(s) for --loso: {sorted(unknown)}")
        print(f"LEAVE-ONE-SITE-OUT: holding out {loso} entirely as test", flush=True)

    print(f"searching {args.restarts} seeded block assignments ...", flush=True)
    best = None
    rng = np.random.default_rng(args.seed)
    for _ in range(args.restarts):
        cand = (propose_loso(pool, blocks, rng, loso) if loso
                else propose(pool, blocks, rng))
        kept, dropped = apply_buffer(cand, nbr)
        if not kept:
            continue
        dev, ok = score(kept, pool, counts, loso)
        # Strictly lexicographic: usable assignments first, then fewest tiles lost to the buffer,
        # then closest to the target proportions. Summing the last two would let a candidate trade
        # a worse drop rate against a better proportion match.
        key = (not ok, round(len(dropped) / len(pool), 4), dev)
        if best is None or key < best[0]:
            best = (key, cand, kept, dropped, dev, ok)

    if best is None:
        raise SystemExit("no candidate assignment retained any tiles; check BLOCK_GRID")
    _, assign, kept, dropped, dev, ok = best
    if not ok:
        raise SystemExit(
            "no candidate kept every foreground class above MIN_CLASS_SHARE in every split, "
            "with every site represented in every split. Refine BLOCK_GRID or raise --restarts."
        )

    print("verifying (independent re-read of the GeoTIFF geometry) ...", flush=True)
    orig_split = {t: pool[t][0] for t in kept}
    bad = verify_independent(kept, root, split_root, orig_split)
    if bad:
        raise SystemExit(f"FAILED: {len(bad)} cross-split overlaps remain, e.g. {bad[:3]}")

    got = Counter(kept.values())
    shares = split_class_shares(kept, counts)
    print(f"\nkept {len(kept)}/{len(pool)} tiles  (dropped {len(dropped)} as buffer, "
          f"{100*len(dropped)/len(pool):.1f}%)")
    for s in SPLITS:
        print(f"  {s:5s} {got[s]:5d} tiles ({100*got[s]/len(kept):4.1f}%)  " +
              "  ".join(f"{c}={100*shares[s][c]:.1f}%" for c in FOREGROUND.values()))
    print(f"  per-site kept: "
          f"{ {site: Counter(kept[t] for t in kept if site_of(t) == site) for site in BLOCK_GRID} }")
    print("VERIFIED: zero cross-split footprint overlap")

    manifest = {
        "generated_by": "scripts/data_prep/build_spatial_split.py",
        "seed": args.seed, "restarts": args.restarts,
        "buffer_widths": args.buffer_widths,
        "min_site_tiles": MIN_SITE_TILES,
        "block_grid": {s: list(v) for s, v in BLOCK_GRID.items()},
        "target_proportions": TARGET,
        "min_class_share": MIN_CLASS_SHARE,
        "n_pool": len(pool), "n_kept": len(kept), "n_dropped_buffer": len(dropped),
        "counts": {s: got[s] for s in SPLITS},
        "per_site_counts": {site: dict(Counter(kept[t] for t in kept if site_of(t) == site))
                            for site in BLOCK_GRID},
        "class_shares": shares,
        "class_and_site_guard_passed": ok,
        "proportion_deviation": dev,
        "cross_split_overlaps": len(bad),
        "verification": "independent re-read of GeoTIFF bounds, grouped by CRS",
        "assignment": kept,
        "dropped_buffer": sorted(dropped),
        "dropped_buffer_intended_split": {t: assign[t] for t in sorted(dropped)},
        "block_of_tile": {t: list(blocks[t]) for t in pool},
    }
    (root / args.out).parent.mkdir(parents=True, exist_ok=True)
    (root / args.out).write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")

    if args.materialise:
        materialise(kept, pool, root, split_root, args.out_root, args.mode)
        print(f"materialised {args.out_root} ({args.mode})")


if __name__ == "__main__":
    main()
