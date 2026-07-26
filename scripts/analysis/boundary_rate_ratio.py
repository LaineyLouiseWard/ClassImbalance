#!/usr/bin/env python3
"""
rho — the pre-registered primary statistic for the boundary-concentration claim.

    rho = (foreground error rate within 8 m of a GT boundary) / (error rate beyond 8 m)

Reported DESCRIPTIVELY. The pre-registration that carried a rho >= 4.0 threshold was withdrawn on
2026-07-26 (D18); the boundary claim rests on the trimap exclusion curve and the per-class boundary
and interior rates. rho is a one-line summary of those rates. There is no threshold and nothing fails.

WHY A RATE RATIO AND NOT A SHARE. Two earlier forms were registered and retracted the same day:

  v0  share of error within 8 m >= 65%.  The share has a mechanical floor equal to the share of
      foreground AREA within 8 m -- 37.8% inland, 26.5% upland -- so one absolute number asks a
      different question of each landscape.
  v1  that share divided by the area share (which is exactly `lift`).  Also landscape-dependent: its
      ceiling is 1/area_share, 2.65 inland against 3.78 upland, so a common threshold sits at
      different points in the two attainable ranges. It also RISES as a model improves, because a
      better model removes interior error first -- so it partly tests how well the model was trained,
      which is the alternative hypothesis the claim exists to exclude.

rho is a ratio of two rates over disjoint pixel sets. No area term enters, so its value does not
depend on how finely the landscape is parcelled.

TWO CONVENTIONS, fixed by the registration and not adjustable here:
  * strict band membership, `distance < 8.0 m`, per-site via `sampling=(gsd_y, gsd_x)`;
  * tiles with NO ground-truth boundary are EXCLUDED. 19 of 191 upland tiles are single-class, so the
    near-boundary set is empty by construction and any error they hold can only depress rho.

THE UNIT OF ANALYSIS IS A SPATIAL BLOCK, NOT A TILE. Tiles are chipped on a 50% stride, so neighbours
repeat ground: the 294-tile test split is 104 pixel-disjoint footprints and only 16 independent units
at the 950 m correlogram range. Resampling tile ids gives intervals ~1.6x too narrow at footprint
level and 10-26x too narrow at the correlation scale (round-2 audit, B6). The bootstrap resamples
grid blocks via utils.spatial_blocks, defaulting to 950 m.

Note single-linkage over overlapping footprints is the WRONG unit and was tried first: a contiguous
test strip is one connected component (A overlaps B, B overlaps C), so all 294 tiles collapse into one
group and the bootstrap becomes degenerate. A grid partitions space rather than chaining through it.

Run:
    PYTHONPATH=. python scripts/analysis/boundary_rate_ratio.py --self-test
    PYTHONPATH=. python scripts/analysis/boundary_rate_ratio.py \\
        --split-root data/split_f1 --split test \\
        --softmax-root sonic/results --cell stage3_clsbal --seeds 42 43 44 45 46 47 48 49 50 51
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

from scripts.analysis.seed_disagreement import boundary_distance
from scripts.analysis.utils import spatial_blocks, resample_blocks

BAND_M = 8.0
# NO THRESHOLD. The pre-registered rho >= 4.0, its 2.0 dead band and its weak band were retired on
# 2026-07-26 (D18): the bar was arbitrary, and a lower-bound rule at 12-16 blocks has an operating
# point near 5.2 and 6.3 rather than 4.0, so defending it needed two further pieces of apparatus.
# rho is reported as a descriptive summary of the boundary and interior rates. Nothing "fails".
FOREGROUND = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}
BACKGROUND = 0


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def site_of(tile_id: str) -> str:
    return tile_id.split("_")[0]



def per_tile_counts(mask_dir: Path, pred_for_tile) -> dict:
    """{tile_id: (err_near, n_near, err_far, n_far)}, foreground only, boundary-free tiles omitted."""
    out = {}
    for p in sorted(mask_dir.glob("*.png")):
        m = np.array(Image.open(p).convert("L"))
        fg = m != BACKGROUND
        if not fg.any():
            continue
        d = boundary_distance(m, p.stem)
        if not np.isfinite(d[fg]).any():
            continue                        # no GT boundary in this tile: rho undefined
        pred = pred_for_tile(p.stem)
        if pred is None:
            continue
        err = fg & (pred != m)
        near = fg & (d < BAND_M)
        far = fg & ~(d < BAND_M)
        out[p.stem] = (int((err & near).sum()), int(near.sum()),
                       int((err & far).sum()), int(far.sum()))
    return out


def rho_from(counts: dict, tiles) -> float:
    en = nn = ef = nf = 0
    for t in tiles:
        c = counts.get(t)
        if c is None:
            continue
        en += c[0]; nn += c[1]; ef += c[2]; nf += c[3]
    if nn == 0 or nf == 0:
        return float("nan")
    r_near, r_far = en / nn, ef / nf
    return float("inf") if r_far == 0 else r_near / r_far


def bootstrap(counts: dict, groups: dict, n_boot: int, rng) -> tuple:
    """Percentile CI, resampling SPATIAL BLOCKS. The tile is not an independent unit here."""
    n_units = len(set(groups.get(t, t) for t in counts))
    if n_units < 3:
        return (float("nan"), float("nan"), n_units)
    vals = []
    for _ in range(n_boot):
        tiles, _ = resample_blocks(list(counts), groups, rng)
        v = rho_from(counts, tiles)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"), n_units)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), n_units)


def report(name: str, counts: dict, groups: dict, n_boot: int, rng) -> dict:
    point = rho_from(counts, list(counts))
    lo, hi, n_groups = bootstrap(counts, groups, n_boot, rng)
    en = sum(c[0] for c in counts.values()); nn = sum(c[1] for c in counts.values())
    ef = sum(c[2] for c in counts.values()); nf = sum(c[3] for c in counts.values())
    print(f"\n{name}")
    print(f"  near-boundary error rate  {100*en/max(nn,1):7.4f}%   ({en:,} / {nn:,} px)")
    print(f"  beyond-8 m error rate     {100*ef/max(nf,1):7.4f}%   ({ef:,} / {nf:,} px)")
    print(f"  rho = {point:.3f}   95% CI [{lo:.3f}, {hi:.3f}]  over {n_groups} blocks "
          f"({len(counts)} tiles)")
    print(f"  descriptive: no threshold is applied to this number (D18). The interval is a "
          f"percentile block bootstrap and under-covers at this many blocks.")
    return {"rho": point, "ci95": [lo, hi], "n_groups": n_groups, "n_tiles": len(counts),
            "err_near": en, "n_near": nn, "err_far": ef, "n_far": nf,
            "rate_near": en / max(nn, 1), "rate_far": ef / max(nf, 1),
            "threshold": None}


def self_test() -> int:
    """Plant a known rho and confirm recovery, and confirm rho is blind to the band's area share.

    The second property is the whole point: it is what the two retracted versions lacked.
    """
    rng = np.random.default_rng(11)
    TRUE_RHO = 6.0
    R_FAR = 0.01
    ok = True

    print("planted rho = 6.0, interior error rate 1%. Two synthetic landscapes whose band area")
    print("shares differ by 2x -- rho must come out the same, a share or a lift would not.")
    for label, band_frac in (("coarse landscape (band = 20% of area)", 0.20),
                             ("fine landscape   (band = 40% of area)", 0.40)):
        counts, groups = {}, {}
        for i in range(120):
            n = 262144
            n_near = int(n * band_frac)
            n_far = n - n_near
            e_near = rng.binomial(n_near, min(TRUE_RHO * R_FAR, 1.0))
            e_far = rng.binomial(n_far, R_FAR)
            t = f"tile_{i:04d}"
            counts[t] = (e_near, n_near, e_far, n_far)
            groups[t] = ("synthetic", i // 3, 0)        # 3 tiles per footprint group
        r = rho_from(counts, list(counts))
        lo, hi, ng = bootstrap(counts, groups, 400, rng)
        good = abs(r - TRUE_RHO) < 0.15
        ok &= good
        print(f"  {label}: rho = {r:.3f} [{lo:.3f}, {hi:.3f}], {ng} groups  "
              f"[{'ok' if good else 'FAIL'}]")

    # The interval must widen when the unit of analysis is respected. Resampling 360 tiles as if
    # independent, versus 120 groups of 3, must not give the same width -- that was audit item B6.
    counts, groups, flat = {}, {}, {}
    for i in range(360):
        n_near, n_far = 52429, 209715
        # ground repeats within a group, so all three tiles of a group share one draw
        if i % 3 == 0:
            e_near = rng.binomial(n_near, TRUE_RHO * R_FAR); e_far = rng.binomial(n_far, R_FAR)
        t = f"t{i:04d}"
        counts[t] = (e_near, n_near, e_far, n_far)
        groups[t] = ("synthetic", i // 3, 0)
        flat[t] = ("synthetic", i, 0)
    _, _, ng = bootstrap(counts, groups, 400, rng)
    g_lo, g_hi, _ = bootstrap(counts, groups, 400, np.random.default_rng(3))
    t_lo, t_hi, nt = bootstrap(counts, flat, 400, np.random.default_rng(3))
    widened = (g_hi - g_lo) > 1.4 * (t_hi - t_lo)
    ok &= widened
    print(f"\n  by footprint group ({ng} units): CI width {g_hi-g_lo:.4f}")
    print(f"  by tile id        ({nt} units): CI width {t_hi-t_lo:.4f}")
    print(f"  group CI is {(g_hi-g_lo)/(t_hi-t_lo):.2f}x wider  "
          f"[{'ok, the unit of analysis matters' if widened else 'FAIL: no difference'}]")

    # The counts above are synthetic, so nothing so far exercises the three conventions the
    # registration actually fixes: the 8 m band, the STRICT `<` membership, the per-site anisotropic
    # pixel size, and the boundary-free exclusion. Those are the parts most likely to be wrong and
    # were untested by the test. Plant them in a real raster.
    print("\nconventions, on synthetic rasters rather than synthetic counts:")

    # (a) a two-class tile split down the middle: the boundary is one column, so at 0.5 m/px the
    # band of strict distance < 8.0 m is exactly 32 columns (16 either side).
    m = np.zeros((64, 64), np.uint8); m[:, :32] = 1; m[:, 32:] = 2
    d = boundary_distance(m, "biodiversity_0001")
    near = (d < BAND_M).sum(axis=0)[0], int((d < BAND_M).sum() / 64)
    good = near[1] == 32
    print(f"  inland 0.500 m/px, one vertical boundary: {near[1]} columns within 8 m "
          f"(expect 32)  [{'ok' if good else 'FAIL'}]")
    ok &= good

    # (b) strict membership: a pixel at exactly 8.0 m must be OUTSIDE the band.
    # Boundary pixels are at distance 0 and here they are columns 31 and 32, so exactly 8.0 m is
    # 16 px out from column 31, i.e. column 15. Getting this index wrong is what the check is for.
    at_edge = float(d[0, 15])
    good = abs(at_edge - 8.0) < 1e-9 and not (at_edge < BAND_M)
    print(f"  pixel at exactly {at_edge:.3f} m is excluded by strict `<`  "
          f"[{'ok' if good else 'FAIL'}]")
    ok &= good

    # (c) anisotropy: the uplands are ~0.641 m/px in x, so the SAME raster must give a NARROWER
    # band in columns. Omitting the tile id silently rescales every band -- that is the bug this
    # checks for.
    d_up = boundary_distance(m, "ireland1_0005")
    cols_up = int((d_up < BAND_M).sum() / 64)
    good = cols_up < near[1]
    print(f"  ireland1 0.641 m/px, same raster: {cols_up} columns within 8 m "
          f"(must be < {near[1]})  [{'ok' if good else 'FAIL'}]")
    ok &= good

    # (d) a single-class tile has no boundary anywhere, so it must be EXCLUDED, not scored as zero.
    import tempfile
    from PIL import Image as _I
    with tempfile.TemporaryDirectory() as td:
        _I.fromarray(np.full((64, 64), 2, np.uint8)).save(Path(td) / "biodiversity_9001.png")
        _I.fromarray(m).save(Path(td) / "biodiversity_9002.png")
        got = per_tile_counts(Path(td), lambda tid: np.full((64, 64), 2, np.uint8))
    good = "biodiversity_9001" not in got and "biodiversity_9002" in got
    print(f"  boundary-free tile excluded, two-class tile kept: {sorted(got)}  "
          f"[{'ok' if good else 'FAIL'}]")
    ok &= good

    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default="data/split_f1")
    ap.add_argument("--split", default="test", choices=["test", "external_test", "val"])
    ap.add_argument("--softmax-root", default=None,
                    help="per-seed softmax dump root; the ensemble argmax is the registered estimator")
    ap.add_argument("--cell", default="stage3_clsbal")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--per-site", action="store_true",
                    help="also report each site separately. Registered as binding for external_test, "
                         "which pools two sites whose band area shares differ by 2x.")
    ap.add_argument("--block-m", type=float, default=950.0,
                    help="bootstrap block size. 950 m = the correlogram range, the "
                         "conservative unit. 256 m gives the pixel-disjoint count.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    root = find_repo_root()
    sr = root / args.split_root
    if not args.softmax_root:
        raise SystemExit(
            "--softmax-root is required: rho is computed from the ten-seed ensemble argmax, which is "
            "the registered estimator. Run --self-test to check the arithmetic without predictions.")

    from scripts.analysis.seed_disagreement import load_seed_stack

    def pred_for_tile(tid: str):
        try:
            stack = load_seed_stack(str(root / args.softmax_root), args.seeds, args.cell, tid)
        except Exception:
            return None
        return stack.mean(axis=0).argmax(axis=0)

    groups = spatial_blocks(sr, args.split, args.block_m)
    fine = spatial_blocks(sr, args.split, 256.0)
    print(f"unit of analysis: {len(set(groups.values()))} blocks at {args.block_m:.0f} m "
          f"({len(set(fine.values()))} pixel-disjoint footprints at 256 m, "
          f"{len(groups)} tiles)")
    counts = per_tile_counts(sr / args.split / "masks", pred_for_tile)
    if not counts:
        raise SystemExit(f"no scorable tiles for {args.split}: no predictions found, or every tile "
                         f"lacks a GT boundary")
    rng = np.random.default_rng(args.seed)

    out = {"split_root": args.split_root, "split": args.split, "cell": args.cell,
           "seeds": args.seeds, "band_m": BAND_M,
           "unit_of_analysis": f"spatial block at {args.block_m:.0f} m"}
    out["pooled"] = report(f"{args.split} — pooled", counts, groups, args.n_boot, rng)

    if args.per_site:
        out["per_site"] = {}
        for site in sorted({site_of(t) for t in counts}):
            sub = {t: c for t, c in counts.items() if site_of(t) == site}
            out["per_site"][site] = report(f"{args.split} — {site}", sub, groups, args.n_boot, rng)

    p = Path(args.out) if args.out else root / f"artifacts/rho_{args.split}_{args.cell}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
