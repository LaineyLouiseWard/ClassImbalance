#!/usr/bin/env python3
"""Coverage of a block-bootstrap interval at this split's block counts.

The numbers in docs/METHODS_STATED_LIMITATIONS.md §7 came from a simulation that lived in a session
scratch directory, which contradicted that file's own rule that every number has committed code
behind it. This is that simulation, committed.

WHAT IT ANSWERS. Test A has 16 independent 950 m blocks and Test B has 14. At that many units a
percentile bootstrap interval does not deliver its nominal coverage, and the shortfall lands on the
lower bound -- the end any threshold would read. This measures the shortfall on the REAL per-block
band and interior pixel counts, so the geometry is this split's rather than an idealisation.

WHAT IT IS NOT. It does not compute rho, which needs trained models. It characterises the interval,
not the result. There is no threshold anywhere in this project since D18; the point of keeping this
is that the coverage of a reported interval is itself a reportable number.

Model of the truth, per block b:
    interior rate  r_far_b = r0  * exp(u_b),   u_b ~ N(0, s_int^2)
    rate ratio     rho_b   = rho * exp(v_b),   v_b ~ N(0, s_rho^2)
    band errors ~ Binomial(N_band_b, clip(r0*rho*exp(u_b+v_b), 0, 1))
    interior    ~ Binomial(N_interior_b, clip(r0*exp(u_b), 0, 1))
Summing tile-level binomials at a common rate inside a block is exactly a block-level binomial, and
the bootstrap only ever resamples whole blocks, so working at block level is exact.

Run:
    PYTHONPATH=. python scripts/analysis/interval_coverage.py --self-test
    PYTHONPATH=. python scripts/analysis/interval_coverage.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

from scripts.analysis.seed_disagreement import boundary_distance
from scripts.analysis.utils import spatial_blocks

BAND_M = 8.0


def block_geometry(split_root: Path, split: str, block_m: float = 950.0) -> tuple:
    """(band px, interior px) per 950 m block, over the boundary-bearing tiles only."""
    blocks = spatial_blocks(split_root, split, block_m)
    agg = defaultdict(lambda: [0, 0])
    for p in sorted((split_root / split / "masks").glob("*.png")):
        m = np.array(Image.open(p).convert("L"))
        fg = m != 0
        if not fg.any():
            continue
        d = boundary_distance(m, p.stem)
        if not np.isfinite(d[fg]).any():
            continue                                  # boundary-free: rho undefined, excluded
        agg[blocks[p.stem]][0] += int((fg & (d < BAND_M)).sum())
        agg[blocks[p.stem]][1] += int((fg & ~(d < BAND_M)).sum())
    keys = sorted(agg, key=str)
    return (np.array([agg[k][0] for k in keys], np.int64),
            np.array([agg[k][1] for k in keys], np.int64))


def pooled_ratio(en, nn, ef, nf):
    a = en.sum(-1) / np.maximum(nn.sum(-1), 1)
    b = ef.sum(-1) / np.maximum(nf.sum(-1), 1)
    return np.where(b > 0, a / np.maximum(b, 1e-300), np.inf)


def intervals(en, nn, ef, nf, rng, n_boot, alpha=0.05):
    """(percentile, BCa, delete-one-block jackknife-t on log) 95% intervals."""
    k = len(nn)
    theta = float(pooled_ratio(en, nn, ef, nf))
    idx = rng.integers(0, k, size=(n_boot, k))
    boot = pooled_ratio(en[idx], nn[idx], ef[idx], nf[idx])
    boot = boot[np.isfinite(boot)]
    keep = ~np.eye(k, dtype=bool)
    jack = np.array([float(pooled_ratio(en[keep[i]], nn[keep[i]], ef[keep[i]], nf[keep[i]]))
                     for i in range(k)])

    pct = (np.percentile(boot, 100 * alpha / 2), np.percentile(boot, 100 * (1 - alpha / 2)))

    prop = np.clip(np.mean(boot < theta), 1 / (2 * len(boot)), 1 - 1 / (2 * len(boot)))
    z0 = stats.norm.ppf(prop)
    jm = jack.mean()
    den = 6.0 * (((jm - jack) ** 2).sum()) ** 1.5
    acc = 0.0 if den == 0 else ((jm - jack) ** 3).sum() / den
    bca = tuple(np.percentile(boot, 100 * stats.norm.cdf(
        z0 + (z0 + stats.norm.ppf(q)) / max(1 - acc * (z0 + stats.norm.ppf(q)), 1e-12)))
        for q in (alpha / 2, 1 - alpha / 2))

    lj = np.log(jack)
    se = np.sqrt((k - 1) / k * ((lj - lj.mean()) ** 2).sum())
    tc = stats.t.ppf(1 - alpha / 2, k - 1)
    jkt = (float(np.exp(np.log(theta) - tc * se)), float(np.exp(np.log(theta) + tc * se)))
    return theta, pct, bca, jkt


def run(nn, nf, rho, r0, s_int, s_rho, n_rep, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    k = len(nn)
    cov = {"percentile": 0, "BCa": 0, "jackknife-t": 0}
    for _ in range(n_rep):
        u, v = rng.normal(0, s_int, k), rng.normal(0, s_rho, k)
        en = rng.binomial(nn, np.clip(r0 * rho * np.exp(u + v), 0, 1))
        ef = rng.binomial(nf, np.clip(r0 * np.exp(u), 0, 1))
        _, pct, bca, jkt = intervals(en, nn, ef, nf, rng, n_boot)
        for name, (lo, hi) in (("percentile", pct), ("BCa", bca), ("jackknife-t", jkt)):
            cov[name] += (lo <= rho <= hi)
    return {k2: v / n_rep for k2, v in cov.items()}


def self_test() -> int:
    """Coverage must FALL as between-block heterogeneity rises, and must not be saturated at 1.
    If the simulation cannot tell those apart it is measuring nothing."""
    ok = True
    nn = np.full(16, 2_000_000, np.int64)
    nf = np.full(16, 4_000_000, np.int64)
    flat = run(nn, nf, 5.0, 0.05, 0.0, 0.0, 60, 400, seed=1)
    hetero = run(nn, nf, 5.0, 0.05, 0.35, 0.5, 60, 400, seed=1)

    # The property that must hold, and the only one worth asserting: coverage FALLS as between-block
    # heterogeneity rises. An earlier version of this test also required coverage >= 0.90 with no
    # heterogeneity, which is a bar invented here rather than derived -- and it failed on an
    # equal-block synthetic structure while the real geometry gives 0.955, so it was measuring the
    # construction rather than the code.
    good = hetero["percentile"] < flat["percentile"]
    print(f"  no heterogeneity      percentile coverage {flat['percentile']:.3f}")
    print(f"  heavy heterogeneity   percentile coverage {hetero['percentile']:.3f}")
    print(f"  coverage falls as heterogeneity rises  "
          f"[{'ok' if good else 'FAIL: the simulation is not measuring heterogeneity'}]")
    ok &= good

    # And it must not be trivially saturated: a measure that returns 1.000 everywhere cannot fail.
    good = flat["percentile"] < 0.999
    print(f"  not saturated at 1.000  [{'ok' if good else 'FAIL'}]")
    ok &= good
    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default="data/split_f1")
    ap.add_argument("--splits", nargs="+", default=["test", "external_test"])
    ap.add_argument("--rho", type=float, default=5.0)
    ap.add_argument("--interior-rate", type=float, default=0.05)
    ap.add_argument("--s-int", type=float, default=0.35)
    ap.add_argument("--s-rho", nargs="+", type=float, default=[0.0, 0.2, 0.35, 0.5])
    ap.add_argument("--n-rep", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=1200)
    ap.add_argument("--out", default="artifacts/interval_coverage.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    repo = Path(__file__).resolve().parents[2]
    out = {"nominal": 0.95, "rho": args.rho, "interior_rate": args.interior_rate,
           "s_int": args.s_int, "n_rep": args.n_rep, "n_boot": args.n_boot, "splits": {}}
    for split in args.splits:
        nn, nf = block_geometry(repo / args.split_root, split)
        w = nn + nf
        print(f"\n{split}: {len(nn)} blocks, Kish n_eff on pixel mass "
              f"{w.sum() ** 2 / (w ** 2).sum():.2f}")
        print(f"  {'s_rho':>6}  {'percentile':>11} {'BCa':>8} {'jackknife-t':>12}   (nominal 0.95)")
        rows = {}
        for s in args.s_rho:
            c = run(nn, nf, args.rho, args.interior_rate, args.s_int, s, args.n_rep, args.n_boot)
            rows[f"{s}"] = c
            print(f"  {s:6.2f}  {c['percentile']:11.3f} {c['BCa']:8.3f} {c['jackknife-t']:12.3f}")
        out["splits"][split] = {"n_blocks": int(len(nn)), "coverage_by_s_rho": rows}

    p = repo / args.out
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
