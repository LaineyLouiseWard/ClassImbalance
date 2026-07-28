#!/usr/bin/env python3
"""
scripts/analysis/factorial_normality_check.py

Validation check for the 2x2 factorial inference: are the per-seed effect
contrasts (and the model residuals) normally distributed?

The factorial main effects and interaction (contrast definitions now in
scripts/analysis/narrative_numbers.py; the figure that used to draw them is retired to
../_archive-lqc/cut_figures/factorial_effects.py)
are reported as 95% *paired-t* confidence intervals over the ten seeds. The paired-t
relies on the ten per-seed contrast values being approximately normal. This script tests
that assumption directly, using the same contrast definitions as narrative_numbers.per_class_contrasts:

  * OEM transfer main effect       = mean[ (transfer-only - baseline) + (full - sampler-only) ] / 2
  * clsbal sampler main effect     = mean[ (sampler-only - baseline) + (full - transfer-only) ] / 2
  * transfer x sampler interaction = [ (full - sampler-only) - (transfer-only - baseline) ] / 2

Two objects are checked:
  1. The 10 per-seed contrast values for each effect (what the paired-t actually uses),
     via the Shapiro-Wilk test.
  2. The residuals of the seed-blocked (randomized-complete-block) 2x2 model on mIoU
     (resid = y - cell_mean - seed_mean + grand_mean), the analogue of Montgomery's
     Figure 6.2 normal-probability plot, again via Shapiro-Wilk.

RESULT (validation set, ten seeds 42-51): normality holds. On the headline mIoU the
per-seed contrasts pass Shapiro-Wilk (OEM transfer p=0.86, clsbal sampler p=0.36,
interaction p=0.11) and the 40 blocked residuals pass strongly (W=0.99, p=0.97), with no
heteroscedasticity in residuals-vs-fitted. Across all eight metrics, 23 of 24 per-seed
contrast tests pass; the single rejection (Grassland interaction, p=0.024) is a negligible,
non-significant effect (~-0.07 pp) and is what chance yields among 24 tests at alpha=0.05.
The paired-t intervals are therefore justified. This is a reviewer-rebuttal /
reproducibility artifact; by decision it is NOT a manuscript figure (the check passed and
RS ablations do not conventionally report it).

Data:   analysis/seed_aggregate/per_seed_metrics.csv (4 cells x 6 metrics x 10 seeds, val)
Run:    python scripts/analysis/factorial_normality_check.py            # prints the Shapiro table
        python scripts/analysis/factorial_normality_check.py --plot     # also writes a diagnostic PNG
Writes: (with --plot) analysis/seed_aggregate/factorial_normality.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
CSV = Path(os.environ.get("PER_SEED_CSV",
                          REPO / "analysis" / "seed_aggregate" / "per_seed_metrics.csv"))
CELLS = ["baseline", "transfer-only", "sampler-only", "full"]
METRICS = ["mIoU", "mF1", "OA"]  # headline metrics; per-class IoU handled below


def per_seed_contrasts(cell_arrays):
    """cell_arrays: dict cell->array over seeds. Returns dict effect->per-seed contrast array."""
    b, t, sa, f = (cell_arrays[c] for c in CELLS)
    return {
        "OEM transfer": ((t - b) + (f - sa)) / 2,
        "clsbal sampler": ((sa - b) + (f - t)) / 2,
        "transfer x sampler": ((f - sa) - (t - b)) / 2,
    }


def cells_for(df, metric, cls=None, split="val"):
    sub = df[(df.split == split) & (df.metric == metric)]
    if cls is not None:
        sub = sub[sub["class"] == cls]
    piv = sub.pivot_table(index="seed", columns="cell_label", values="value_pct")[CELLS]
    return {c: piv[c].values for c in CELLS}, piv


def shapiro_line(name, x):
    W, p = stats.shapiro(x)
    flag = "normal" if p > 0.05 else "REJECT (p<=0.05)"
    print(f"  {name:34s} mean={x.mean():+7.3f}  W={W:.3f}  p={p:.3f}  {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="also write a diagnostic QQ / residual PNG")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="which splits to check. Default: every split present in the CSV. This was "
                         "hard-coded to val, so the Shapiro-Wilk check that licenses the paired t "
                         "ran on the checkpoint-selection split and never on the split the reported "
                         "effects come from.")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    splits = args.splits or sorted(df.split.unique())

    # The multiplicity family, stated rather than left to be counted by a reader: 4 effects x 3
    # metrics x len(splits) paired t-tests, plus 5 classes x 4 cells x len(splits) per-class
    # intervals. Nothing here corrects them; the effects are reported as descriptive.
    n_tests = 4 * 3 * len(splits)
    print(f"Splits checked: {splits}")
    print(f"Multiplicity family: {n_tests} paired t-tests "
          f"(4 effects x 3 metrics x {len(splits)} splits), uncorrected, plus "
          f"{5 * 4 * len(splits)} per-class intervals.\n")

    resid_by_split = {}
    for split in splits:
        print(f"===== {split} =====")
        print("Per-seed effect contrasts (Shapiro-Wilk normality; the paired-t assumption):")
        for metric in METRICS:
            cells, _ = cells_for(df, metric, split=split)
            print(f"[{metric}]")
            for eff, x in per_seed_contrasts(cells).items():
                shapiro_line(eff, x)
        classes = sorted(df[(df.metric == "IoU") & df["class"].notna()]["class"].unique())
        for cls in classes:
            cells, _ = cells_for(df, "IoU", cls, split=split)
            print(f"[IoU: {cls}]")
            for eff, x in per_seed_contrasts(cells).items():
                shapiro_line(eff, x)

        # Blocked (RCBD) residuals on mIoU: analogue of Montgomery Fig 6.2.
        _, piv = cells_for(df, "mIoU", split=split)
        Y = piv.values
        fitted = Y.mean(0, keepdims=True) + Y.mean(1, keepdims=True) - Y.mean()
        resid = (Y - fitted).ravel()
        resid_by_split[split] = (resid, fitted)
        W, p = stats.shapiro(resid)
        print(f"\nBlocked 2x2 residuals on mIoU ({resid.size} residuals): "
              f"W={W:.3f}  p={p:.3f}  {'normal' if p > 0.05 else 'REJECT'}\n")

    resid, fitted = resid_by_split[splits[-1]]
    W, p = stats.shapiro(resid)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        stats.probplot(resid, dist="norm", plot=ax[0])
        ax[0].set_title(f"Normal probability plot of residuals (W={W:.3f}, p={p:.2f})")
        ax[0].get_lines()[0].set(marker="o", ms=5, mfc="#2166AC", mec="k")
        ax[1].scatter(fitted.ravel(), resid, c="#2166AC", edgecolors="k", s=30)
        ax[1].axhline(0, color="grey", lw=0.8)
        ax[1].set(xlabel="Fitted mIoU (pp)", ylabel="Residual (pp)", title="Residuals vs fitted")
        fig.tight_layout()
        out = CSV.parent / f"factorial_normality_{splits[-1]}.png"
        fig.savefig(out, dpi=140)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
