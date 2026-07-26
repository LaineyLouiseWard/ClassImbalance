#!/usr/bin/env python3
"""
rho — the pre-registered primary statistic for the boundary-concentration claim.

    rho = (foreground error rate within 8 m of a GT boundary) / (error rate beyond 8 m)

Registered in docs/PREREGISTRATION_P1_AMENDMENT.md (version 2). Threshold: rho >= 4.0 on BOTH test
sets, judged on the LOWER bound of a bootstrap 95% CI; dead below 2.0; weak in between.

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

THE UNIT OF ANALYSIS IS THE FOOTPRINT GROUP, NOT THE TILE. Tiles are chipped on a 50% stride, so
neighbours repeat ground: 294 test tiles are roughly 105 pixel-disjoint footprints. Resampling tiles
would give intervals about 1.6x too narrow. The bootstrap here resamples non-overlapping footprint
groups, which is the correction the round-2 audit asked for (its B6).

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

BAND_M = 8.0
THRESHOLD = 4.0          # registered: claim holds at or above this, on the lower CI bound
DEAD_BELOW = 2.0         # registered: claim dead below this
FOREGROUND = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}
BACKGROUND = 0


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def site_of(tile_id: str) -> str:
    return tile_id.split("_")[0]


def footprint_groups(split_root: Path, split: str) -> dict:
    """{tile_id: group_id}, one group per set of mutually overlapping tiles.

    Single-linkage over footprint overlap, within a CRS. Two tiles sharing any ground land in the same
    group, so a group is a unit of independent ground and is the correct bootstrap unit.
    """
    rows = []
    for f in sorted((split_root / split / "images").glob("*.tif")):
        with rasterio.open(f) as d:
            b = d.bounds
            rows.append((f.stem, b.left, b.bottom, b.right, b.top, str(d.crs)))
    parent = {r[0]: r[0] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_crs = defaultdict(list)
    for r in rows:
        by_crs[r[5]].append(r)
    for _, rs in by_crs.items():
        arr = np.array([r[1:5] for r in rs], float)
        L, B, R, T = arr.T
        tol = 1e-9 * float(np.min(R - L))
        for i, ri in enumerate(rs):
            ox = np.minimum(R[i], R) - np.maximum(L[i], L)
            oy = np.minimum(T[i], T) - np.maximum(B[i], B)
            for j in np.where((ox > tol) & (oy > tol))[0]:
                if j != i:
                    union(ri[0], rs[j][0])
    return {t: find(t) for t in parent}


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
    """Percentile CI, resampling FOOTPRINT GROUPS. The tile is not an independent unit here."""
    by_group = defaultdict(list)
    for t in counts:
        by_group[groups.get(t, t)].append(t)
    keys = sorted(by_group)
    if len(keys) < 3:
        return (float("nan"), float("nan"), len(keys))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(keys, len(keys), replace=True)
        tiles = [t for k in pick for t in by_group[k]]
        v = rho_from(counts, tiles)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"), len(keys))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(keys))


def verdict(lo: float) -> str:
    if not np.isfinite(lo):
        return "UNDETERMINED"
    if lo >= THRESHOLD:
        return "HOLDS"
    if lo < DEAD_BELOW:
        return "DEAD"
    return "WEAK"


def report(name: str, counts: dict, groups: dict, n_boot: int, rng) -> dict:
    point = rho_from(counts, list(counts))
    lo, hi, n_groups = bootstrap(counts, groups, n_boot, rng)
    en = sum(c[0] for c in counts.values()); nn = sum(c[1] for c in counts.values())
    ef = sum(c[2] for c in counts.values()); nf = sum(c[3] for c in counts.values())
    print(f"\n{name}")
    print(f"  near-boundary error rate  {100*en/max(nn,1):7.4f}%   ({en:,} / {nn:,} px)")
    print(f"  beyond-8 m error rate     {100*ef/max(nf,1):7.4f}%   ({ef:,} / {nf:,} px)")
    print(f"  rho = {point:.3f}   95% CI [{lo:.3f}, {hi:.3f}]  over {n_groups} footprint groups "
          f"({len(counts)} tiles)")
    print(f"  registered verdict on the LOWER bound: {verdict(lo)}   "
          f"(holds >= {THRESHOLD}, dead < {DEAD_BELOW})")
    return {"rho": point, "ci95": [lo, hi], "n_groups": n_groups, "n_tiles": len(counts),
            "err_near": en, "n_near": nn, "err_far": ef, "n_far": nf,
            "rate_near": en / max(nn, 1), "rate_far": ef / max(nf, 1),
            "verdict_on_lower_bound": verdict(lo)}


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
            groups[t] = f"g{i // 3}"        # 3 tiles per footprint group
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
        groups[t] = f"g{i // 3}"
        flat[t] = t
    _, _, ng = bootstrap(counts, groups, 400, rng)
    g_lo, g_hi, _ = bootstrap(counts, groups, 400, np.random.default_rng(3))
    t_lo, t_hi, nt = bootstrap(counts, flat, 400, np.random.default_rng(3))
    widened = (g_hi - g_lo) > 1.4 * (t_hi - t_lo)
    ok &= widened
    print(f"\n  by footprint group ({ng} units): CI width {g_hi-g_lo:.4f}")
    print(f"  by tile id        ({nt} units): CI width {t_hi-t_lo:.4f}")
    print(f"  group CI is {(g_hi-g_lo)/(t_hi-t_lo):.2f}x wider  "
          f"[{'ok, the unit of analysis matters' if widened else 'FAIL: no difference'}]")

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

    groups = footprint_groups(sr, args.split)
    counts = per_tile_counts(sr / args.split / "masks", pred_for_tile)
    if not counts:
        raise SystemExit(f"no scorable tiles for {args.split}: no predictions found, or every tile "
                         f"lacks a GT boundary")
    rng = np.random.default_rng(args.seed)

    out = {"split_root": args.split_root, "split": args.split, "cell": args.cell,
           "seeds": args.seeds, "band_m": BAND_M, "threshold": THRESHOLD,
           "dead_below": DEAD_BELOW, "unit_of_analysis": "non-overlapping footprint group"}
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
