#!/usr/bin/env python3
"""
Factorial contrasts computed WITHIN fold, then averaged across folds.

WHY. Under the spatially blocked design the same three folds are run for all four factorial cells,
so fold is a blocking factor. Between-fold variance is then common to both arms of every contrast and
cancels when the contrast is formed inside a fold. Pooling all runs per cell into one mean and SD
throws that cancellation away: the folds cut the inland site along different axes, and composition
shifts a lot with the cut (cropland's share of the test strip alone swings 1.1--7.1%), so the pooled
SD absorbs a between-region difference that the contrast never had to pay for.

That failure is symmetric and both directions are bad for the paper. Pooled SDs with a "the effects
are weak" conclusion is an artefact of the wrong error term. Pooled SDs with a "the effects are
strong" conclusion keeps the conclusion but quotes intervals that cannot be defended.

Two different questions get two different numbers here, and they are never added together:

  headline interval      how much does the effect move when the SEED changes, with the three held-out
                         regions fixed? Pooled within-fold seed variance, fold as a block. This is
                         the interval that belongs beside the effect in the results table.

  between-fold spread    how much does the effect move when the REGION changes? The spread of the
                         three per-fold effect estimates. Three regions cannot support a usable
                         t-interval (df=2), so this is reported as a spread in one robustness
                         sentence -- "consistent across all three held-out regions" -- and not folded
                         into the headline SD.

Reporting shape is fixed by notes/MANUSCRIPT_TODO.md §J3: the fold dimension never reaches a
main-text table. This script emits the headline row plus the per-fold values for the supplementary
table, and nothing else.

Run:
    PYTHONPATH=. python scripts/analysis/factorial_within_fold.py --self-test
    PYTHONPATH=. python scripts/analysis/factorial_within_fold.py \\
        --csv analysis/seed_aggregate/per_run_metrics.csv --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CELLS = ["baseline", "transfer-only", "sampler-only", "full"]
REQUIRED_COLUMNS = {"fold", "seed", "split", "cell_label", "metric", "value_pct"}


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "artifacts").is_dir() and (parent / "geoseg").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def contrasts(b, t, sa, f) -> dict:
    """The three factorial effects, on whatever shape the four cell arrays share.

    Identical definitions to factorial_normality_check.py, so the two scripts cannot drift: each main
    effect averages its two independent estimates (one at each level of the other factor).
    """
    return {
        "OEM transfer": ((t - b) + (f - sa)) / 2,
        "clsbal sampler": ((sa - b) + (f - t)) / 2,
        "transfer x sampler": ((f - sa) - (t - b)) / 2,
    }


def cell_cube(df: pd.DataFrame, metric: str, split: str, cls=None):
    """-> (folds, seeds, {cell: (n_fold, n_seed) array}). Every cell must cover every fold x seed."""
    sub = df[(df["split"] == split) & (df["metric"] == metric)]
    if cls is None:
        sub = sub[sub["class"].isna() | (sub["class"] == "")] if "class" in sub else sub
    else:
        sub = sub[sub["class"] == cls]
    if sub.empty:
        raise SystemExit(f"no rows for metric={metric!r} split={split!r} class={cls!r}")

    folds = sorted(sub["fold"].unique())
    seeds = sorted(sub["seed"].unique())
    absent = [c for c in CELLS if c not in set(sub["cell_label"])]
    if absent:
        raise SystemExit(f"cells absent from {metric}/{split}: {absent}")

    out = {}
    for c in CELLS:
        piv = (sub[sub["cell_label"] == c]
               .pivot_table(index="fold", columns="seed", values="value_pct")
               .reindex(index=folds, columns=seeds))
        if piv.isna().any().any():
            gaps = [(f, s) for f in folds for s in seeds if pd.isna(piv.loc[f, s])]
            raise SystemExit(
                f"cell {c!r} is missing {len(gaps)} fold x seed runs, e.g. {gaps[:4]}. The blocked "
                f"arithmetic requires a complete design: an absent run is not a zero, and dropping "
                f"it silently would unbalance the contrast.")
        out[c] = piv.to_numpy(float)
    return folds, seeds, out


def analyse(folds, seeds, cube) -> dict:
    """Blocked analysis. Returns one record per effect."""
    per_fold_seed = contrasts(*(cube[c] for c in CELLS))  # each (n_fold, n_seed)
    n_fold, n_seed = len(folds), len(seeds)
    records = []

    for name, c in per_fold_seed.items():
        per_fold = c.mean(axis=1)                     # effect within each fold, averaged over seeds
        effect = float(per_fold.mean())

        # Headline interval: seed variability with the regions held fixed. Fold is removed as a block,
        # so between-region differences never enter this error term.
        resid = c - per_fold[:, None]
        df_resid = n_fold * (n_seed - 1)
        if df_resid > 0:
            s_within = float(np.sqrt((resid ** 2).sum() / df_resid))
            se = s_within / np.sqrt(n_fold * n_seed)
            tcrit = float(stats.t.ppf(0.975, df_resid))
            ci = (effect - tcrit * se, effect + tcrit * se)
            pval = float(2 * stats.t.sf(abs(effect / se), df_resid)) if se > 0 else float("nan")
        else:
            s_within, se, ci, pval = float("nan"), float("nan"), (float("nan"),) * 2, float("nan")

        # Region generalisation, kept separate on purpose. Never added to the headline interval.
        between_sd = float(np.std(per_fold, ddof=1)) if n_fold > 1 else float("nan")
        same_sign = bool(np.all(per_fold > 0) or np.all(per_fold < 0))

        records.append({
            "effect": name,
            "estimate_pp": effect,
            "ci95_pp": list(ci),
            "p_value": pval,
            "within_fold_seed_sd_pp": s_within,
            "df": df_resid,
            "per_fold_pp": {str(f): float(v) for f, v in zip(folds, per_fold)},
            "between_fold_sd_pp": between_sd,
            "between_fold_range_pp": (float(per_fold.max() - per_fold.min())
                                      if n_fold > 1 else float("nan")),
            "consistent_sign_across_folds": same_sign,
        })
    return {"n_folds": n_fold, "n_seeds": n_seed, "folds": [str(f) for f in folds],
            "seeds": [int(s) for s in seeds], "effects": records}


def pooled_for_comparison(cube) -> dict:
    """Per-CELL mean and SD pooled over all runs — the naive results table, for diagnosis only.

    Note this is deliberately about cells, not contrasts. A fold offset shifts all four cells of that
    fold together, so it cancels inside any contrast and pooling contrast values would show nothing
    wrong. The mistake §J3 warns about happens one step earlier, when each cell is summarised as one
    mean +- SD over every run: there the between-region offset lands squarely in the SD, and the error
    bar can swallow an effect that the blocked contrast resolves cleanly.
    """
    out = {}
    n_fold, n_seed = next(iter(cube.values())).shape
    for c in CELLS:
        v = cube[c]
        naive = float(np.std(v.reshape(-1), ddof=1))
        resid = v - v.mean(axis=1)[:, None]
        blocked = float(np.sqrt((resid ** 2).sum() / (n_fold * (n_seed - 1))))
        out[c] = {"mean_pp": float(v.mean()), "naive_sd_pp": naive, "within_fold_sd_pp": blocked}
    return out


def report(res: dict, metric: str, split: str, pooled: dict | None) -> None:
    print(f"\n{metric} / {split} — {res['n_folds']} folds x {res['n_seeds']} seeds "
          f"(folds {', '.join(res['folds'])})")
    print(f"{'effect':<20} {'estimate':>10} {'95% CI':>18} {'p':>8} {'seed SD':>9}   per fold")
    for r in res["effects"]:
        lo, hi = r["ci95_pp"]
        pf = "  ".join(f"{k}={v:+.2f}" for k, v in r["per_fold_pp"].items())
        print(f"{r['effect']:<20} {r['estimate_pp']:+9.2f}  [{lo:+6.2f}, {hi:+6.2f}] "
              f"{r['p_value']:8.4f} {r['within_fold_seed_sd_pp']:9.2f}   {pf}")

    print("\nregion generalisation, reported separately (never added to the CI above):")
    for r in res["effects"]:
        flag = "same sign in every fold" if r["consistent_sign_across_folds"] else "SIGN FLIPS"
        print(f"  {r['effect']:<20} between-fold SD {r['between_fold_sd_pp']:.2f} pp, "
              f"range {r['between_fold_range_pp']:.2f} pp — {flag}")

    if pooled:
        print("\n  --- per-cell SD pooled over folds: WRONG ERROR TERM, diagnosis only ---")
        for c in CELLS:
            p = pooled[c]
            infl = (p["naive_sd_pp"] / p["within_fold_sd_pp"]
                    if p["within_fold_sd_pp"] > 0 else float("nan"))
            print(f"  {c:<16} {p['mean_pp']:6.2f}  SD pooled {p['naive_sd_pp']:5.2f} vs "
                  f"within-fold {p['within_fold_sd_pp']:5.2f}  ({infl:.1f}x inflated)")
        print("  Quoting the pooled column would report a between-region difference as run-to-run")
        print("  noise. The contrasts above are immune to it; a per-cell error bar is not.")


def self_test() -> int:
    """Plant known effects, add large between-fold offsets, confirm the blocked arithmetic recovers them.

    The point is the between-fold offsets. They are added identically to all four cells within a fold,
    exactly as a genuine composition difference between held-out regions would be, so a correct
    blocked contrast must be blind to them while a pooled SD must not be.
    """
    rng = np.random.default_rng(7)
    # Standard 2^2 parameterisation: cell means are mu, mu+A, mu+B, mu+A+B+AB.
    A, B, AB = 2.80, 0.40, -0.60
    FOLD_OFFSET = {"f1": 0.0, "f2": -6.0, "f3": +4.5}   # far larger than any effect
    RUN_SD = 0.30                                       # independent per run, as seeds actually are
    seeds = [42, 43, 44]

    rows = []
    for fold, off in FOLD_OFFSET.items():
        for s in seeds:
            mu = 87.0 + off
            cells = {"baseline": mu, "transfer-only": mu + A, "sampler-only": mu + B,
                     "full": mu + A + B + AB}
            for c, v in cells.items():
                rows.append({"fold": fold, "seed": s, "split": "test", "cell_label": c,
                             "metric": "mIoU", "class": "", "value_pct": v + rng.normal(0, RUN_SD)})
    df = pd.DataFrame(rows)

    folds, seed_list, cube = cell_cube(df, "mIoU", "test")
    res = analyse(folds, seed_list, cube)
    pooled = pooled_for_comparison(cube)
    report(res, "mIoU (synthetic)", "test", pooled)

    got = {r["effect"]: r for r in res["effects"]}
    # What the contrasts SHOULD equal, derived from the same parameterisation rather than assumed.
    # With an interaction present a main effect is not the simple effect: substituting the cell means
    # into ((t-b)+(f-sa))/2 gives A + AB/2, and the interaction contrast is AB/2, not AB.
    expect = {"OEM transfer": A + AB / 2, "clsbal sampler": B + AB / 2,
              "transfer x sampler": AB / 2}
    print()
    ok = True
    for name, want in expect.items():
        r = got[name]
        # Run noise enters each contrast, so allow a little slack, not exactness.
        near = abs(r["estimate_pp"] - want) < 0.25
        blind = r["between_fold_sd_pp"] < 0.5     # must NOT track the +-6 pp fold offsets
        covers = r["ci95_pp"][0] <= want <= r["ci95_pp"][1]
        ok &= near and blind and covers
        print(f"  {name:<20} planted {want:+.2f}  recovered {r['estimate_pp']:+.2f}  "
              f"[{'ok' if near else 'FAIL'}]  between-fold SD {r['between_fold_sd_pp']:.3f} "
              f"[{'blind to fold offsets' if blind else 'FAIL: absorbed them'}]  "
              f"CI covers truth [{'ok' if covers else 'FAIL'}]")

    # And the naive per-cell SD must visibly break, or blocking is solving nothing here. The planted
    # fold offsets span 10.5 pp, so a pooled per-cell SD has to be several times the run noise.
    infl = min(pooled[c]["naive_sd_pp"] / pooled[c]["within_fold_sd_pp"] for c in CELLS)
    caught = infl > 3.0
    ok &= caught
    print(f"\n  naive per-cell SD inflation (worst cell): {infl:.1f}x "
          f"[{'ok, blocking is doing real work' if caught else 'FAIL: no difference to detect'}]")
    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="analysis/seed_aggregate/per_run_metrics.csv",
                    help="per-run metrics with a 'fold' column (written by the campaign aggregation)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--metric", default="mIoU")
    ap.add_argument("--class-name", default=None, help="per-class IoU instead of the macro metric")
    ap.add_argument("--show-pooled", action="store_true",
                    help="also print the pooled-over-folds SD, labelled as the wrong error term")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="verify the arithmetic on synthetic data with planted effects")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    root = find_repo_root()
    csv = root / args.csv
    if not csv.exists():
        raise SystemExit(
            f"{args.csv} not found. It is written by the campaign aggregation once the folds have "
            f"been trained; until then run --self-test to check the arithmetic.")
    df = pd.read_csv(csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(
            f"{args.csv} lacks {sorted(missing)}. A per-seed CSV without a 'fold' column cannot be "
            f"blocked by fold — pooling it would be the exact mistake this script exists to avoid.")

    folds, seeds, cube = cell_cube(df, args.metric, args.split, args.class_name)
    res = analyse(folds, seeds, cube)
    pooled = pooled_for_comparison(cube) if args.show_pooled else None
    report(res, args.metric if args.class_name is None else f"{args.metric}[{args.class_name}]",
           args.split, pooled)

    res["metric"] = args.metric
    res["split"] = args.split
    res["class"] = args.class_name
    out = Path(args.out) if args.out else (root / "analysis" / "seed_aggregate" /
                                           f"factorial_within_fold_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
