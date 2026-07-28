#!/usr/bin/env python3
"""
Per-class, per-seed error rate near a boundary and deep inside parcels.

`boundary_trimap_iou.py` gives per-class rates from the SEED ENSEMBLE and per-seed rates POOLED over
classes. Neither supports the claim this script exists to test: that the two curation interventions
change semi-natural grassland's INTERIOR error rate. That claim needs per class AND per seed, so the
spread across the four cells can be compared against the spread across the ten seeds. Without the
comparison, four cell means differing by a few points say nothing -- the same arithmetic on the
boundary rates showed the cell spread there was entirely seed noise.

Two distance sets, both on ground-truth boundaries, both in metres via the per-site anisotropic
sampling:

    near     distance <  8 m     (the registered band)
    interior distance >= 32 m    (deep interior: four band widths out)

Tiles with no ground-truth boundary are excluded, as in boundary_rate_ratio.py -- their pixels are
all at infinite distance and would land wholly in the interior set, which is exactly the bias that
inflated Test B before 2026-07-28.

Run:
    PYTHONPATH=. python scripts/analysis/interior_error_by_class.py --self-test
    PYTHONPATH=. python scripts/analysis/interior_error_by_class.py \\
        --softmax-root <dir> --split test --cell stage1_baseline --out-dir analysis/interior
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")
FG = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}
NEAR_M = 8.0
INTERIOR_M = 32.0


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def accumulate_bins(mask, dist, pred, bin_n, bin_e, edges):
    """Add one tile's per-class counts into (6, nbin) accumulators over the distance bins.

    The decay curve and any single-threshold figure MUST come from the same accumulation, or the
    headline and the curve end up in different units -- which is exactly what happened when the
    curve was taken from the seed ensemble and the headline from the per-seed mean.
    """
    fg = mask != 0
    err = fg & (pred != mask)
    bidx = np.digitize(dist, edges[1:-1])
    for k in FG:
        sel = mask == k
        if not sel.any():
            continue
        bin_n[k] += np.bincount(bidx[sel], minlength=len(edges) - 1)
        e = sel & err
        if e.any():
            bin_e[k] += np.bincount(bidx[e], minlength=len(edges) - 1)


def accumulate(mask, dist, pred, near_n, near_e, int_n, int_e):
    """Add one tile's counts, per class, into the four (6,) accumulators."""
    fg = mask != 0
    err = fg & (pred != mask)
    near = fg & (dist < NEAR_M)
    inter = fg & (dist >= INTERIOR_M)
    for k in FG:
        sel = mask == k
        near_n[k] += int((sel & near).sum())
        near_e[k] += int((sel & near & err).sum())
        int_n[k] += int((sel & inter).sum())
        int_e[k] += int((sel & inter & err).sum())


def self_test() -> int:
    """Plant per-class error that is entirely interior for one class and entirely near-boundary for
    another, then confirm the accumulator separates them. A statistic that cannot tell those two
    cases apart is the whole point of failure here."""
    ok = True
    # 32 m of interior at 0.5 m/px is 64 px, so every class in the fixture must be more than 128 px
    # across or its interior set is empty and the checks below pass vacuously. The first version of
    # this test used a 200 px raster with a 60 px slab and reported zero interior support for both
    # classes -- it failed, which is why the gate is written this way.
    H = W = 600
    mask = np.full((H, W), 2, np.uint8)          # grassland background
    mask[:, :200] = 1                            # forest slab, 100 m wide
    mask[250:450, 250:450] = 5                   # semi-natural block, 100 m square
    from scripts.analysis.seed_disagreement import boundary_distance
    dist = boundary_distance(mask, "biodiversity_0001")

    pred = mask.copy()
    # class 1 wrong ONLY near a boundary; class 5 wrong ONLY deep inside.
    pred[(mask == 1) & (dist < NEAR_M)] = 2
    pred[(mask == 5) & (dist >= INTERIOR_M)] = 2

    n0 = {k: 0 for k in FG}
    near_n, near_e, int_n, int_e = (dict(n0), dict(n0), dict(n0), dict(n0))
    accumulate(mask, dist, pred, near_n, near_e, int_n, int_e)

    f_near = near_e[1] / max(near_n[1], 1)
    f_int = int_e[1] / max(int_n[1], 1)
    s_near = near_e[5] / max(near_n[5], 1)
    s_int = int_e[5] / max(int_n[5], 1)
    checks = [("forest near rate is 1.0", abs(f_near - 1.0) < 1e-9),
              ("forest interior rate is 0.0", f_int == 0.0),
              ("semi-natural near rate is 0.0", s_near == 0.0),
              ("semi-natural interior rate is 1.0", abs(s_int - 1.0) < 1e-9)]
    for name, good in checks:
        ok &= good
        print(f"  {name}  [{'ok' if good else 'FAIL'}]")

    # The interior set must be non-empty for BOTH classes, else the checks above pass vacuously.
    good = int_n[1] > 1000 and int_n[5] > 100
    ok &= good
    print(f"  interior support forest {int_n[1]}, semi-natural {int_n[5]} (both must be non-trivial)"
          f"  [{'ok' if good else 'FAIL'}]")

    # accumulate_bins had NO test at all: dropping its per-class restriction made every class's
    # curve identical to the whole-tile total and the self-test still passed. The bin edges include
    # 8.0 and 32.0 exactly, so the binned counts must reproduce the two-band summary term for term.
    from scripts.analysis.seed_disagreement import DIST_BIN_EDGES_M as EDGES
    nb = len(EDGES) - 1
    bn = {k: np.zeros(nb, np.int64) for k in FG}
    be = {k: np.zeros(nb, np.int64) for k in FG}
    accumulate_bins(mask, dist, pred, bn, be, EDGES)
    i8 = int(np.where(EDGES == NEAR_M)[0][0])
    i32 = int(np.where(EDGES == INTERIOR_M)[0][0])
    agree = all(bn[k][:i8].sum() == near_n[k] and be[k][:i8].sum() == near_e[k]
                and bn[k][i32:].sum() == int_n[k] and be[k][i32:].sum() == int_e[k] for k in FG)
    ok &= agree
    print(f"  binned counts reproduce the two-band summary for all five classes  "
          f"[{'ok' if agree else 'FAIL'}]")

    # And a tile with no boundary must be skipped, not counted as all-interior.
    flat = np.full((64, 64), 2, np.uint8)
    d2 = boundary_distance(flat, "biodiversity_0001")
    good = not np.isfinite(d2[flat != 0]).any()
    ok &= good
    print(f"  single-class tile has no finite boundary distance  [{'ok' if good else 'FAIL'}]")

    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--softmax-root")
    ap.add_argument("--split", default="test", choices=["test", "external_test"])
    ap.add_argument("--cell")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--dedup", action="store_true",
                    help="restrict to the non-overlapping scoring subset")
    ap.add_argument("--out-dir", default="analysis/interior")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    import sys
    sys.path.insert(0, str(root))
    if args.self_test:
        return self_test()
    if not (args.softmax_root and args.cell):
        raise SystemExit("--softmax-root and --cell are required")

    from scripts.analysis.seed_disagreement import boundary_distance, load_seed_stack

    mask_dir = root / f"data/split_{SPLIT_TAG}/{args.split}/masks"
    keep = None
    if args.dedup:
        sub = json.loads((root / f"artifacts/scoring_subset_{SPLIT_TAG}.json").read_text())
        keep = set(sub["splits"][args.split]["tiles"])
    tiles = sorted(p for p in mask_dir.glob("*.png") if keep is None or p.stem in keep)

    from scripts.analysis.seed_disagreement import DIST_BIN_EDGES_M as EDGES
    nb = len(EDGES) - 1
    z = {k: 0 for k in FG}
    acc = {s: (dict(z), dict(z), dict(z), dict(z)) for s in args.seeds}
    bacc = {s: ({k: np.zeros(nb, np.int64) for k in FG},
                {k: np.zeros(nb, np.int64) for k in FG}) for s in args.seeds}
    n_scored = n_skipped = 0
    for p in tiles:
        m = np.array(Image.open(p).convert("L"))
        fg = m != 0
        if not fg.any():
            continue
        d = boundary_distance(m, p.stem)
        if not np.isfinite(d[fg]).any():
            n_skipped += 1
            continue
        n_scored += 1
        stack = load_seed_stack(args.softmax_root, args.seeds, args.cell, p.stem)
        for i, s in enumerate(args.seeds):
            pr = stack[i].argmax(axis=0)
            accumulate(m, d, pr, *acc[s])
            accumulate_bins(m, d, pr, *bacc[s], EDGES)

    if n_scored == 0:
        raise SystemExit(f"no scorable tiles in {mask_dir}: every tile lacks a GT boundary")

    out = {"split": args.split, "cell": args.cell, "seeds": args.seeds,
           "dedup": bool(args.dedup), "near_m": NEAR_M, "interior_m": INTERIOR_M,
           "n_tiles_scored": n_scored, "n_tiles_boundary_free_skipped": n_skipped,
           "edges_m": EDGES.tolist(), "per_seed": {}, "per_seed_bins": {}}
    for s in args.seeds:
        bn, be = bacc[s]
        out["per_seed_bins"][str(s)] = {
            FG[k]: {"n": bn[k].tolist(), "err": be[k].tolist()} for k in FG}
        nn, ne, tn, te = acc[s]
        out["per_seed"][str(s)] = {
            FG[k]: {"near_rate": ne[k] / max(nn[k], 1), "near_n": nn[k],
                    "interior_rate": te[k] / max(tn[k], 1), "interior_n": tn[k]}
            for k in FG}

    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"interior_{args.split}_{args.cell}{'_dedup' if args.dedup else ''}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"{args.cell} {args.split}: {n_scored} tiles scored, {n_skipped} boundary-free skipped")
    for k in FG:
        r = [out["per_seed"][str(s)][FG[k]]["interior_rate"] for s in args.seeds]
        n = out["per_seed"][str(args.seeds[0])][FG[k]]["interior_n"]
        print(f"  {FG[k]:12s} interior rate {100*np.mean(r):6.2f}% +/- {100*np.std(r, ddof=1):5.2f} "
              f"over {len(r)} seeds, n={n:,}")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
