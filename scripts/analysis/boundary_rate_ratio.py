#!/usr/bin/env python3
"""
rho — the boundary/interior error rate ratio.

    rho = (foreground error rate within 8 m of a GT boundary) / (error rate beyond 8 m)

Reported descriptively. A pre-registered rho >= 4.0 threshold was withdrawn on 2026-07-26 (D18),
so rho is not tested against anything. The boundary claim rests on the trimap exclusion curve and
the per-class boundary and interior rates; rho summarises those rates in one number.

rho IS THE 8 m BAND. The contact-zone ratio in boundary_trimap_iou.py uses a 1.5 m near band
against the same 8 m interior floor, so the 1.5-8 m annulus falls in neither of its sets. It is a
different statistic on a different partition. Report the two separately and never under one name.

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

NO INTERVAL IS REPORTED ON RHO, and the block bootstrap that used to produce one was removed on
2026-07-26. Four reasons, in order of weight:

  1. It existed to supply the lower bound the pre-registered rho >= 4.0 threshold was to be judged
     on. D18 retired the threshold; the machinery outlived its only purpose.
  2. Both test sets are COMPLETE ENUMERATIONS of their ground -- every tile scored, every pixel
     counted. There is no sample, so there is no sampling error to interval.
  3. Every claim this study makes is either a cross-cell contrast on IDENTICAL pixels (n_near and
     n_far are bit-identical across the four cells, so the landscape is a constant in the difference
     and the variance that matters is the training run) or a census level. Neither needs it.
  4. Where spatial variance would genuinely be wanted -- does this hold on other landscapes -- a
     within-site bootstrap estimates the wrong component. Between-site n is 2.

Uncertainty in this study is PER-SEED AND PAIRED (scripts/analysis/aggregate_seeds.py, and the
per-seed curves in boundary_trimap_iou.py). One estimator, not two.

The 950 m grid is still reported, as DESCRIPTION: how many cells of ground the tiles touch. That is
a spread statistic, not a count of independent parcels -- Test A's 294 tiles span 16 cells over
6.767 km2, which is 7.50 cells' worth of ground, and Test B's 14 cells sit on 5.72 cells' worth.
Never call these independent.

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
from scripts.analysis.utils import spatial_blocks

BAND_M = 8.0
# Settable with --band-m so rho can be reported as a SWEEP over widths rather than at one arbitrary
# width (docs/CORRECTIONS_PAPER_PT2.md §8). 8 m stays the default because it is what was registered.
# Two widths matter beyond it: 1.5 m is 3 px at 0.5 m/px, the pixel-matched comparison to Volpi &
# Tuia's 3 px erosion, and the ladder in boundary_trimap_iou.RADII_PX shares this figure's x-axis.
# self_test() forces this back to 8.0 -- its expectations (32 columns either side of one boundary,
# a pixel at exactly 8.000 m excluded by strict `<`) are arithmetic on the 8 m band, so running the
# gate at another width would silently check nothing.
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


def report(name: str, counts: dict, groups: dict) -> dict:
    """rho and the two rates it is built from. NO INTERVAL -- see the module docstring."""
    point = rho_from(counts, list(counts))
    n_cells = len(set(groups.get(t, t) for t in counts))
    en = sum(c[0] for c in counts.values()); nn = sum(c[1] for c in counts.values())
    ef = sum(c[2] for c in counts.values()); nf = sum(c[3] for c in counts.values())
    print(f"\n{name}")
    # Both labels MUST name the band actually used. They read "beyond-8 m" unconditionally until
    # 2026-07-28, so a run swept to another width printed the right number under the wrong label.
    print(f"  within-{BAND_M:g} m error rate  {100*en/max(nn,1):7.4f}%   ({en:,} / {nn:,} px)")
    print(f"  beyond-{BAND_M:g} m error rate  {100*ef/max(nf,1):7.4f}%   ({ef:,} / {nf:,} px)")
    print(f"  rho = {point:.3f}   over {len(counts)} tiles spanning {n_cells} grid cells "
          f"of 950 m (cells touched, NOT independent parcels)")
    print(f"  descriptive: no threshold (D18) and no interval. This is a census of the ground "
          f"named above, not a sample of it; uncertainty in this study is per-seed and paired.")
    return {"rho": point, "n_cells_touched": n_cells, "n_tiles": len(counts),
            "err_near": en, "n_near": nn, "err_far": ef, "n_far": nf,
            "rate_near": en / max(nn, 1), "rate_far": ef / max(nf, 1),
            "threshold": None, "interval": None}


def self_test() -> int:
    global BAND_M
    BAND_M = 8.0   # see the note at BAND_M: this gate's arithmetic is specific to 8 m
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
        good = abs(r - TRUE_RHO) < 0.15
        ok &= good
        print(f"  {label}: rho = {r:.3f}  [{'ok' if good else 'FAIL'}]")

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

    # (e) THE NULL CONTROL, added 2026-07-27. Everything above plants a rho of 6.0 and checks it
    # comes back. Nothing had ever asked the opposite question: given ground with NO boundary
    # concentration, does the instrument correctly say so? An instrument that manufactures
    # concentration out of geometry -- the distance transform, the strict band edge, the anisotropic
    # sampling, the boundary-free exclusion -- would produce this paper's headline result from noise,
    # and every test above would still pass. This repository's own standard is that a gate which has
    # not been observed to fail does not exist; this is that observation for the primary statistic.
    #
    # Errors are placed at a UNIFORM rate over foreground, independent of distance to a boundary, on
    # a REAL raster geometry rather than synthetic counts, because the geometry is the thing at risk.
    import tempfile
    from PIL import Image as _I2
    rng2 = np.random.default_rng(7)
    P_ERR = 0.10
    with tempfile.TemporaryDirectory() as td:
        masks = {}
        for i in range(6):
            mm = np.ones((256, 256), np.uint8)
            for _ in range(9):                              # irregular multi-class patches
                y, x = rng2.integers(0, 200, 2); h, w = rng2.integers(25, 70, 2)
                mm[y:y+h, x:x+w] = rng2.integers(1, 6)
            tid = f"biodiversity_95{i:02d}"
            _I2.fromarray(mm).save(Path(td) / f"{tid}.png")
            masks[tid] = mm

        def uniform_err_pred(tid):
            mm = masks[tid]
            pred = mm.copy()
            flip = rng2.random(mm.shape) < P_ERR           # uniform in space, ignores distance
            wrong = (mm % 5) + 1
            pred[flip] = wrong[flip]
            return pred

        got = per_tile_counts(Path(td), uniform_err_pred)
    r_null = rho_from(got, list(got))
    good = abs(r_null - 1.0) < 0.05
    ok &= good
    print(f"  NULL CONTROL: error spread uniformly, no boundary concentration -> rho = {r_null:.3f} "
          f"(must be ~1.000)  [{'ok' if good else 'FAIL: the instrument invents concentration'}]")

    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default="data/split_f1")
    ap.add_argument("--split", default="test", choices=["test", "external_test", "val"])
    ap.add_argument("--softmax-root", default=None,
                    help="per-seed softmax dump root; rho is computed from the ten-seed ensemble argmax")
    ap.add_argument("--cell", default="stage3_clsbal")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--per-site", action="store_true",
                    help="also report each site separately. Registered as binding for external_test, "
                         "which pools two sites whose band area shares differ by 2x.")
    ap.add_argument("--block-m", type=float, default=950.0,
                    help="grid cell size for the DESCRIPTIVE cells-touched count. 950 m = the "
                         "conservative unit. 256 m gives the pixel-disjoint count.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--band-m", type=float, default=8.0,
                    help="near-band half-width in metres; 8.0 is the registered value, "
                         "1.5 is 3 px at 0.5 m/px (Volpi pixel-matched). Sweep it.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    global BAND_M
    BAND_M = float(args.band_m)

    if args.self_test:
        return self_test()

    root = find_repo_root()
    sr = root / args.split_root
    if not args.softmax_root:
        raise SystemExit(
            "--softmax-root is required: rho is computed from the ten-seed ensemble argmax. "
            "Run --self-test to check the arithmetic without predictions.")

    from scripts.analysis.seed_disagreement import load_seed_stack

    # A tile with no dump is a MISSING tile, not an absent one. The blanket `except Exception:
    # return None` this replaces swallowed a missing seed, a truncated .npy and a shape mismatch
    # alike, and per_tile_counts drops a None without a word -- so an incomplete ensemble came out
    # as a smaller tile count and a rho computed from whatever happened to be on disk. Total
    # failure was visible; PARTIAL failure was not, and partial is the likely one when nine of ten
    # seeds run in separate worktrees.
    missing: list[str] = []

    def pred_for_tile(tid: str):
        try:
            stack = load_seed_stack(str(root / args.softmax_root), args.seeds, args.cell, tid)
        except FileNotFoundError:
            missing.append(tid)
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
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(counts) + len(missing)} boundary-carrying {args.split} tiles "
            f"have no softmax dump for cell {args.cell} across seeds {args.seeds}.\n"
            f"  e.g. {missing[:5]}\n"
            f"rho would be computed from the {len(counts)} tiles that happen to be on disk, which "
            f"is a different estimand from the registered one. Complete the dumps "
            f"(`bash RUNBOOK.sh --from C5 --to C5` for each seed) or pass the seeds you actually "
            f"have via --seeds.")

    out = {"split_root": args.split_root, "split": args.split, "cell": args.cell,
           "seeds": args.seeds, "band_m": BAND_M,
           "grid_cells_touched_at_m": args.block_m, "interval": None}
    out["pooled"] = report(f"{args.split} — pooled", counts, groups)

    if args.per_site:
        out["per_site"] = {}
        for site in sorted({site_of(t) for t in counts}):
            sub = {t: c for t, c in counts.items() if site_of(t) == site}
            out["per_site"][site] = report(f"{args.split} — {site}", sub, groups)

    p = Path(args.out) if args.out else root / f"artifacts/rho_{args.split}_{args.cell}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    # relative_to raises for an --out outside the repo, which crashed the run AFTER the result had
    # been written and every number computed: exit 1 on a success, from a cosmetic print.
    try:
        shown = p.relative_to(root)
    except ValueError:
        shown = p
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
