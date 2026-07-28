#!/usr/bin/env python3
"""
Accuracy as a function of distance from the nearest training tile.

WHY. The buffer is a single chosen width (650 m), and Wadoux et al. (2021) object that the radius in
any spatial-CV design is subjective. Roberts et al. (2017) answer that objection directly: rather
than defend one radius, run the evaluation across a range of separations and report the curve --

  "cross-validations could be run several times with spatial or other structured blocks defined in a
   variety of sizes and/or orientations. This approach produces a range of validation statistics ...
   it may be possible to define a limit ... at which the model no longer produces useful
   predictions. Such extrapolation limits, or 'forecast horizons'..."

This computes that curve post hoc from a single trained model, at no extra training cost: every test
tile is labelled by its distance to the nearest training tile, and accuracy is reported per distance
stratum. Three uses:

  1. It shows the 650 m buffer is adequate rather than asserting it -- if accuracy is flat beyond
     650 m, nothing is left to buffer away.
  2. It is the honest answer to "why 650 and not 1,100": the curve is reported across radii, so the
     conclusion does not depend on the choice.
  3. It places the external upland test set on the same axis as the inland test strip, turning two
     separate numbers into one gradient of increasing distance from labelled ground.

The unit of analysis is the TILE, not the pixel. Overlapping chips and dense pixel scoring make
pseudoreplication the default failure in this dataset: a previous claim of "168k pixels, therefore
not noise" turned out to rest on three tiles, so the tile count is printed beside every
stratum. No interval is reported: this is a description of a fully enumerated test set.

Run:
    PYTHONPATH=. python scripts/analysis/accuracy_vs_separation.py \\
        --split-root data/split_f1 --metrics-dir evaluation/evaluation_results/f1
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

# Metres per degree: ONE definition, in geoseg/geo.py. These two literals appeared in seven
# files with no derivation, and terrain_separability used the LONGITUDE constant for the
# latitude direction.
from geoseg.geo import M_PER_DEG_LAT, M_PER_DEG_LON_EQ  # noqa: E402,F401

SPLITS_HELD_OUT = ("test", "external_test")
FOREGROUND = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}
# Strata in metres. The first edge is the buffer itself; beyond that, whether accuracy is flat is
# the question the figure exists to answer.
EDGES_M = [0.0, 650.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, float("inf")]


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        # `artifacts` and `scripts`, NOT `data`: data/ is gitignored, so keying on it made every
        # one of these scripts raise "repo root not found" in a fresh clone -- including the ledger
        # a reviewer would run to check the paper's numbers.
        if (parent / "artifacts").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def site_of(tile_id: str) -> str:
    return tile_id.split("_")[0]


def read_bounds(split_root: Path) -> dict:
    """{tile_id: (split, left, bottom, right, top, is_geographic)} over a materialised split."""
    out = {}
    for split in ("train", "val", "test", "external_test"):
        d = split_root / split / "images"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.tif")):
            with rasterio.open(f) as src:
                b = src.bounds
                out[f.stem] = (split, b.left, b.bottom, b.right, b.top,
                               bool(src.crs and src.crs.is_geographic))
    return out


def distance_to_training_m(bounds: dict) -> dict:
    """{held-out tile: metres to the nearest TRAINING tile footprint}.

    Distances are only meaningful within a site: the three sites sit in two coordinate systems and
    are tens of km apart, so a cross-site distance would be both wrong in units and meaningless in
    interpretation. A held-out tile whose site contributes no training tiles -- which is exactly the
    upland external test set -- is reported as infinite separation. That is the correct reading: its
    nearest labelled ground is another site entirely, and it belongs at the far end of the axis.
    """
    train_by_site = defaultdict(list)
    for tid, (split, *_rest) in bounds.items():
        if split == "train":
            train_by_site[site_of(tid)].append(tid)

    out = {}
    for tid, (split, L, B, R, T, geo) in bounds.items():
        if split not in SPLITS_HELD_OUT:
            continue
        peers = train_by_site.get(site_of(tid), [])
        if not peers:
            out[tid] = float("inf")
            continue
        if geo:
            lat = (B + T) / 2.0
            mx, my = M_PER_DEG_LON_EQ * math.cos(math.radians(lat)), M_PER_DEG_LAT
        else:
            mx = my = 1.0
        best = float("inf")
        for jid in peers:
            _, jl, jb, jr, jt, _ = bounds[jid]
            dx = max(jl - R, L - jr, 0.0) * mx
            dy = max(jb - T, B - jt, 0.0) * my
            best = min(best, math.hypot(dx, dy))
        out[tid] = best
    return out


def load_per_tile_iou(metrics_dir: Path) -> dict:
    """{tile_id: {class_index: (intersection, union)}} from a per-tile metrics dump.

    Expects `per_tile_iou.json` as written by the evaluation stage. Kept separate from the geometry
    so this script never re-runs inference: it is a stratification of numbers already computed.
    """
    p = metrics_dir / "per_tile_iou.json"
    if not p.exists():
        raise SystemExit(
            f"{p} not found. Run the evaluation with per-tile IoU accumulation first; this script "
            f"only stratifies existing per-tile numbers, it does not run inference.")
    raw = json.loads(p.read_text())
    return {t: {int(k): tuple(v) for k, v in d.items()} for t, d in raw.items()}


def macro_iou(tiles: list, per_tile: dict) -> float:
    """Micro-pooled per-class IoU, then unweighted mean over foreground classes.

    Pooling intersections and unions across tiles before dividing is the same convention the paper
    uses elsewhere; averaging per-tile IoUs would weight a tile holding twelve pixels of cropland
    the same as one holding half a million.
    """
    inter = defaultdict(float)
    union = defaultdict(float)
    for t in tiles:
        for k, (i, u) in per_tile.get(t, {}).items():
            inter[k] += i
            union[k] += u
    vals = [inter[k] / union[k] for k in FOREGROUND if union.get(k, 0) > 0]
    return float(np.mean(vals)) if vals else float("nan")


# bootstrap_ci REMOVED 2026-07-26 with the block bootstrap. This script reports macro IoU against
# distance from training ground; the distance strata are a DESCRIPTION of a fully enumerated test
# set, not a sample of one, so a resampling interval on them answers a question nobody asked.
# Uncertainty in this study is per-seed and paired (aggregate_seeds.py).
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True, help="a materialised split, e.g. data/split_f1")
    ap.add_argument("--metrics-dir", required=True, nargs="+",
                    help="one or more directories holding per_tile_iou.json. Test A and Test B are "
                         "written to DIFFERENT directories by compute_metrics, so passing one "
                         "silently drops the other stratum -- which is the whole point of this "
                         "script. Pass both.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = find_repo_root()
    rng = np.random.default_rng(args.seed)
    bounds = read_bounds(root / args.split_root)
    dist = distance_to_training_m(bounds)
    per_tile = {}
    for d in args.metrics_dir:
        got = load_per_tile_iou(root / d)
        overlap = set(got) & set(per_tile)
        if overlap:
            raise SystemExit(f"{d} repeats {len(overlap)} tiles already loaded, e.g. "
                             f"{sorted(overlap)[:3]} -- two directories for the same split?")
        per_tile.update(got)

    covered = {s for s, *_ in (bounds[t] for t in per_tile if t in bounds)}
    missing = [s for s in SPLITS_HELD_OUT if s not in covered]
    if missing:
        raise SystemExit(
            f"no scored tiles for {missing} in {args.metrics_dir}. This script exists to put the "
            f"inland strip and the upland sites on ONE axis; reporting it over a subset is the "
            f"defect it was written to expose. Pass the metrics directory for each split.")
    from scripts.analysis.utils import spatial_blocks
    blocks = {}
    for sp in ("test", "external_test"):
        if (root / args.split_root / sp / "images").is_dir():
            blocks.update(spatial_blocks(root / args.split_root, sp, 950.0))

    print(f"{len(dist)} held-out tiles from {args.split_root}")
    strata, rows = [], []
    for lo, hi in zip(EDGES_M[:-1], EDGES_M[1:]):
        sel = [t for t, d in dist.items() if lo <= d < hi]
        if not sel:
            continue
        v = macro_iou(sel, per_tile)
        by_split = defaultdict(int)
        for t in sel:
            by_split[bounds[t][0]] += 1
        label = ("inf" if hi == float("inf") else f"{hi:.0f}")
        rows.append({"lo_m": lo, "hi_m": (None if hi == float("inf") else hi),
                     "n_tiles": len(sel), "macro_fg_iou": v,
                     "by_split": dict(by_split)})
        strata.append((f"{lo:.0f}-{label}", len(sel), v, dict(by_split)))

    print(f"\n{'separation (m)':>16} {'tiles':>6} {'macro fg IoU':>13}  composition")
    for name, n, v, comp in strata:
        print(f"{name:>16} {n:6d} {100*v:12.2f}%  {comp}")

    print("\nReading: if IoU is flat beyond the 650 m buffer, nothing is left for a wider buffer to")
    print("remove. A decline continuing well past it is a forecast horizon, not a leakage artefact.")
    print("The infinite stratum is the upland external test set: its nearest labelled ground is a")
    print("different site, which is why it sits at the far end of the axis.")

    out = Path(args.out) if args.out else (root / args.metrics_dir[0] / "accuracy_vs_separation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split_root": args.split_root, "edges_m": EDGES_M,
                               "interval": None,
                               "strata": rows}, indent=2))
    # relative_to raises for a path outside the repo, AFTER the result is written:
    # exit 1 on a success, from a cosmetic print.
    try:
        _shown = out.relative_to(root)
    except ValueError:
        _shown = out
    print(f"\nwrote {_shown}")


if __name__ == "__main__":
    main()
