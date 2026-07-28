#!/usr/bin/env python3
"""
Where the errors of one class pair sit, and how big they are. Per seed, Test A.

Two measurements that between them decide whether "whole parcels are misassigned" is a measurement
or an interpretation:

  A. DISTANCE. For each ordered pair (A, B), the share of A-called-B error pixels lying more than
     8 m and more than 32 m from ANY ground-truth class boundary. Not from B's boundary -- from any
     boundary, which is the quantity a reader assumes.

  B. COMPONENT SIZE. Connected components (8-connectivity) of the A-called-B error mask, and how the
     error mass is distributed over them. If the mass sits in a few large components, whole-parcel
     misassignment is measured. If it sits in many small ones, it is not, and the distance result in
     A follows almost automatically from fields being large.

THE TILE-EDGE PROBLEM, and why the guarded figure is the one to quote.
`boundary_distance` runs on ONE tile. A pixel 3 m from a boundary that happens to lie in the next
tile is scored as 3 m from nothing, so its distance is over-stated and it can be counted as "beyond
8 m" when it is not. The bias is one-directional: it can only inflate the beyond-band shares.

Rather than estimate the bias, this excludes it. A pixel more than W metres from the TILE EDGE
cannot have its "beyond W m" status changed by anything outside the tile, because any outside
boundary is at least W away. So each band is reported twice: over all error pixels, and over the
guarded subset. Quote the guarded one.

Run:
    PYTHONPATH=. python scripts/analysis/pair_error_geometry.py --self-test
    PYTHONPATH=. python scripts/analysis/pair_error_geometry.py \\
        --softmax-root <dir> --split test --cell stage1_baseline
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")
FG = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}
BANDS = (8.0, 32.0)
STRUCT = np.ones((3, 3), bool)          # 8-connectivity


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def edge_distance(shape, sampling) -> np.ndarray:
    """Metres from each pixel to the nearest tile edge, on the same anisotropic sampling."""
    h, w = shape
    sy, sx = sampling
    yy = np.minimum(np.arange(h), h - 1 - np.arange(h))[:, None] * sy
    xx = np.minimum(np.arange(w), w - 1 - np.arange(w))[None, :] * sx
    return np.minimum(yy, xx)


def pair_counts(e, d, ed, bands, widest):
    """Per-band raw / per-band-guarded / common-guarded counts for one error mask on one tile.

    Extracted from main() on 2026-07-28 because it lived there and `self_test` never called `main`,
    so deleting the guard entirely (`ed > w` -> `ed > 0`) left the self-test passing. On the 90
    deduped test tiles that mutation moves Grassland->Seminatural's beyond-8 m share from 0.00% to
    45.42% on a synthetic shifted prediction whose true share is zero.
    """
    out = {"n": int(e.sum()), "common_n": int((e & (ed > widest)).sum())}
    common = e & (ed > widest)
    for w in bands:
        g = e & (ed > w)
        out[f"beyond{w:g}"] = int((e & (d >= w)).sum())
        out[f"guard_n{w:g}"] = int(g.sum())
        out[f"guard_beyond{w:g}"] = int((g & (d >= w)).sum())
        out[f"common_beyond{w:g}"] = int((common & (d >= w)).sum())
    return out


def component_sizes(err_mask: np.ndarray, px_area_m2: float) -> np.ndarray:
    """Areas in m2 of the 8-connected components of an error mask, largest first."""
    if not err_mask.any():
        return np.zeros(0)
    lab, n = ndimage.label(err_mask, structure=STRUCT)
    if n == 0:
        return np.zeros(0)
    counts = np.bincount(lab.ravel())[1:]
    return np.sort(counts.astype(float) * px_area_m2)[::-1]


def self_test() -> int:
    ok = True
    # (1) The edge guard must actually bite. One 100x100 tile at 0.5 m/px is 50 m across, so the
    # pixels more than 32 m from an edge form an empty set -- there is no 32 m-guarded interior in a
    # tile this small, and the guarded count must be zero rather than silently equal to the raw one.
    ed = edge_distance((100, 100), (0.5, 0.5))
    good = ed.max() == 24.5 and (ed > 32.0).sum() == 0
    ok &= good
    print(f"  50 m tile: max edge distance {ed.max()} m, pixels beyond 32 m from an edge "
          f"{(ed > 32.0).sum()}  [{'ok' if good else 'FAIL'}]")
    # At 0.5 m/px, 8 m is 16 PIXELS, so row i qualifies when min(i, 99-i) > 16, i.e. i in [17, 82]
    # -- 66 rows, and the guarded region is the inner 66x66 block. (The first version of this gate
    # expected 84x84, having subtracted 8 in metres from a count in pixels. The gate failed and the
    # expectation was wrong, not the code; this is the same units slip the boundary band has its own
    # test for.)
    good = (ed > 8.0).sum() == 66 * 66
    ok &= good
    print(f"  ... pixels beyond 8 m from an edge {(ed > 8.0).sum()} (expect {66*66})  "
          f"[{'ok' if good else 'FAIL'}]")

    # (2) Component sizes: three squares of known area, one of them touching only diagonally, which
    # 8-connectivity must merge and 4-connectivity would not.
    m = np.zeros((60, 60), bool)
    m[5:15, 5:15] = True                 # 10x10 = 100 px
    m[30:40, 30:40] = True               # 10x10, and
    m[40:45, 40:45] = True               # 5x5 touching it corner-to-corner -> one component of 125
    sizes = component_sizes(m, 0.25)     # 0.5 m px -> 0.25 m2
    good = sizes.size == 2 and np.allclose(sizes, [125 * 0.25, 100 * 0.25])
    ok &= good
    print(f"  component areas {sizes.tolist()} m2 (expect [31.25, 25.0]; the diagonal pair merges "
          f"under 8-connectivity)  [{'ok' if good else 'FAIL'}]")

    # (3) The whole point: a few large components vs many small ones must be distinguishable by the
    # share of mass in the largest components.
    big = np.zeros((200, 200), bool); big[20:120, 20:120] = True
    scattered = np.zeros((200, 200), bool); scattered[::4, ::4] = True
    sb, ss = component_sizes(big, 0.25), component_sizes(scattered, 0.25)
    share_b = sb[:5].sum() / sb.sum()
    share_s = ss[:5].sum() / ss.sum()
    good = share_b > 0.99 and share_s < 0.01
    ok &= good
    print(f"  mass in 5 largest components: one block {100*share_b:.1f}%, scattered "
          f"{100*share_s:.2f}%  [{'ok' if good else 'FAIL'}]")

    # (4) THE GUARD ITSELF, on a two-tile scene. The left tile is one class throughout, so scored
    # alone every pixel in it is infinitely far from a boundary. The scene has a boundary at the
    # seam, so pixels near the right edge of that tile are in truth CLOSE to one. An unguarded
    # count therefore reports them as beyond 8 m; the guard must exclude every one of them.
    W = 8.0
    scene = np.ones((64, 128), np.uint8); scene[:, 64:] = 2
    left = scene[:, :64]                                   # the tile as the pipeline sees it
    d_alone = np.full(left.shape, np.inf)                  # single-class tile: no in-tile boundary
    # EDT measures distance to the ZEROS of its input, so to get "distance to the class-2 region"
    # the input must be `!= 2`, not `!= 1`. Writing it the other way round made d_scene zero
    # everywhere and the gate reported 900 spurious flips -- a fault in the fixture, caught by the
    # fixture, which is the behaviour wanted.
    d_scene = ndimage.distance_transform_edt(
        np.pad(left, ((0, 0), (0, 1)), constant_values=2) != 2, sampling=(0.5, 0.5))[:, :64]
    ed = edge_distance(left.shape, (0.5, 0.5))
    e = np.ones(left.shape, bool)
    raw = pair_counts(e, d_alone, ed, (W,), W)
    truth = pair_counts(e, d_scene, ed, (W,), W)
    flipped_raw = raw["beyond8"] - truth["beyond8"]
    flipped_guarded = raw["guard_beyond8"] - int((e & (ed > W) & (d_scene >= W)).sum())
    good = flipped_raw > 0 and flipped_guarded == 0
    ok &= good
    print(f"  guard: {flipped_raw} pixels wrongly 'beyond 8 m' without it, {flipped_guarded} with it "
          f"(must be >0 then 0)  [{'ok' if good else 'FAIL'}]")
    # and the guard must actually discard something, or it is vacuous
    good = raw["guard_n8"] < raw["n"]
    ok &= good
    print(f"  guard drops {raw['n'] - raw['guard_n8']:,} of {raw['n']:,} pixels "
          f"(must be non-zero)  [{'ok' if good else 'FAIL'}]")

    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--softmax-root")
    ap.add_argument("--split", default="test", choices=["test", "external_test"])
    ap.add_argument("--cell", default="stage1_baseline")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--pairs", nargs="+", default=["Seminatural:Grassland", "Grassland:Seminatural",
                                                   "Forest:Grassland", "Grassland:Forest",
                                                   "Cropland:Grassland"])
    ap.add_argument("--out-dir", default="analysis/pair_geometry")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    import sys
    sys.path.insert(0, str(root))
    if args.self_test:
        return self_test()
    if not args.softmax_root:
        raise SystemExit("--softmax-root is required")

    from scripts.analysis.seed_disagreement import boundary_distance, load_seed_stack
    from geoseg.geo import gsd_for

    inv = {v: k for k, v in FG.items()}
    pairs = [(inv[p.split(":")[0]], inv[p.split(":")[1]]) for p in args.pairs]

    sub = json.loads((root / f"artifacts/scoring_subset_{SPLIT_TAG}.json").read_text())
    keep = set(sub["splits"][args.split]["tiles"])
    tiles = sorted(p for p in (root / f"data/split_{SPLIT_TAG}/{args.split}/masks").glob("*.png")
                   if p.stem in keep)

    # Two guards, deliberately.
    #   per-band  -- guard each band at its own width. Uses the most pixels for that band, but the
    #                8 m and 32 m figures then come from DIFFERENT pixel populations, so the decay
    #                across widths is not comparable.
    #   common    -- guard once at the WIDEST band and report every band on that one subset. Fewer
    #                pixels, but the decay is comparable because the population is fixed.
    # Quote the common-guard column for any statement about how the rate CHANGES with width.
    WIDEST = max(BANDS)
    acc = {(a, b): {s: {"n": 0, **{f"beyond{w:g}": 0 for w in BANDS},
                        **{f"guard_n{w:g}": 0 for w in BANDS},
                        **{f"guard_beyond{w:g}": 0 for w in BANDS},
                        "common_n": 0, **{f"common_beyond{w:g}": 0 for w in BANDS},
                        "sizes": []} for s in args.seeds} for a, b in pairs}
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
        sampling = gsd_for(p.stem)
        ed = edge_distance(m.shape, sampling)
        px_area = float(sampling[0] * sampling[1])
        stack = load_seed_stack(args.softmax_root, args.seeds, args.cell, p.stem)
        for i, s in enumerate(args.seeds):
            pr = stack[i].argmax(axis=0)
            for a, b in pairs:
                e = (m == a) & (pr == b)
                if not e.any():
                    continue
                c = acc[(a, b)][s]
                pc = pair_counts(e, d, ed, BANDS, WIDEST)
                for k, v in pc.items():
                    c[k] += v
                c["sizes"].extend(component_sizes(e, px_area).tolist())

    out = {"split": args.split, "cell": args.cell, "seeds": args.seeds, "bands_m": list(BANDS),
           "n_tiles_scored": n_scored, "n_tiles_boundary_free_skipped": n_skipped,
           "distance_transform": "per tile; guarded columns exclude pixels within the band of a "
                                 "tile edge, where an out-of-tile boundary could change the answer",
           "pairs": {}}
    for (a, b), per in acc.items():
        key = f"{FG[a]}->{FG[b]}"
        rows = []
        for s in args.seeds:
            c = per[s]
            sizes = np.sort(np.array(c["sizes"]))[::-1]
            tot = sizes.sum()
            rows.append({
                "seed": s, "n": c["n"],
                **{f"beyond{w:g}_pct": 100 * c[f"beyond{w:g}"] / max(c["n"], 1) for w in BANDS},
                **{f"guarded_beyond{w:g}_pct":
                   100 * c[f"guard_beyond{w:g}"] / max(c[f"guard_n{w:g}"], 1) for w in BANDS},
                **{f"guarded_n{w:g}": c[f"guard_n{w:g}"] for w in BANDS},
                "common_guard_n": c["common_n"],
                **{f"common_guarded_beyond{w:g}_pct":
                   100 * c[f"common_beyond{w:g}"] / max(c["common_n"], 1) for w in BANDS},
                "n_components": int(sizes.size),
                "median_component_m2": float(np.median(sizes)) if sizes.size else 0.0,
                "largest_component_m2": float(sizes[0]) if sizes.size else 0.0,
                "share_mass_in_components_over_1000m2":
                    float(sizes[sizes > 1000].sum() / tot) if tot else 0.0,
                "share_mass_in_components_over_10000m2":
                    float(sizes[sizes > 10000].sum() / tot) if tot else 0.0,
            })
        out["pairs"][key] = rows

    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"pair_geometry_{args.split}_{args.cell}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"{args.split} {args.cell}: {n_scored} tiles, {n_skipped} boundary-free skipped")
    for key, rows in out["pairs"].items():
        n = np.mean([r["n"] for r in rows])
        b8 = np.mean([r["beyond8_pct"] for r in rows])
        g8 = np.mean([r["guarded_beyond8_pct"] for r in rows])
        g32 = np.mean([r["guarded_beyond32_pct"] for r in rows])
        c8 = np.mean([r["common_guarded_beyond8_pct"] for r in rows])
        c32 = np.mean([r["common_guarded_beyond32_pct"] for r in rows])
        s1k = np.mean([r["share_mass_in_components_over_1000m2"] for r in rows])
        med = np.mean([r["median_component_m2"] for r in rows])
        print(f"  {key:26s} n={n:>12,.0f}  >8m {b8:5.1f}% (guarded {g8:5.1f}%)  "
              f"guarded >32m {g32:5.1f}%  [common guard: >8m {c8:5.1f}% >32m {c32:5.1f}%]  "
              f"mass in >1000 m2 comps {100*s1k:5.1f}%  "
              f"median comp {med:8.1f} m2")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
