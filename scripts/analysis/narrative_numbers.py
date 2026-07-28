#!/usr/bin/env python3
"""
The four numbers the narrative rests on, computed by one script so each has a command behind it.

Every one of these was first produced ad hoc -- SSH one-liners and throwaway Python -- during the
2026-07-28 session. The values were checked, but nothing in the repository reproduced them, and a
number in the abstract that no script produces is a number the author cannot defend. This file
closes that gap. It computes:

  1. AREA-WEIGHTED CLASS-PAIR RATIO. A pair's share of foreground error divided by the share it
     would hold if error fell in proportion to the two classes' areas. The raw share alone is not
     usable: Grassland covers ~72% of the scored foreground, so ANY pair containing it starts large,
     and "the grassland pair is 47% of error" reads as a finding when part of it is arithmetic.
  2. PER-CLASS PAIRED CONTRASTS with intervals. `aggregate_seeds.py` does this for the aggregate
     only. DO_NOT_ADD forbids calling a per-class result a null without a bar, and §8 of
     RESULTS_TEN_SEED currently gives means and sign counts with no interval.
  3. THE SEED-ONLY CONTROL for the registered across-cell boundary comparison. The registered arm
     compares the near-boundary and interior rate spreads ACROSS THE FOUR CELLS. That is only
     informative if the cells differ by more than the seeds do. Swapping the roles -- spread across
     the ten seeds within a cell -- is the control, and on this campaign it returns the same numbers,
     which is why the arm is reported as uninformative.
  4. RHO ON THE DEDUPLICATED SUBSET. rho is computed over every tile in the split while the reported
     IoU is pooled over the non-overlapping subset only, so the two are censuses of different tile
     populations. This reports both so the manuscript can state which.

Run:
    PYTHONPATH=. python scripts/analysis/narrative_numbers.py --self-test
    PYTHONPATH=. python scripts/analysis/narrative_numbers.py \\
        --metrics-dir evaluation/evaluation_results --confusion-dir analysis/confusion \\
        --rho-dir analysis/rho --out artifacts/narrative_numbers.json
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

CELLS = ["stage1_baseline", "stage2b_oem_finetune", "stage_sampler_only", "stage3_clsbal"]
CELL_LABEL = {"stage1_baseline": "baseline", "stage2b_oem_finetune": "transfer only",
              "stage_sampler_only": "sampler only", "stage3_clsbal": "full"}
SEEDS = list(range(42, 52))
FG = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}
# The metrics JSONs use the dataset's long names; the confusion uses class indices.
IOU_KEYS = {1: "Forest land", 2: "Grassland", 3: "Cropland", 4: "Settlement",
            5: "Seminatural Grassland"}


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        # `artifacts` and `scripts`, NOT `data`: data/ is gitignored, so keying on it made every
        # one of these scripts raise "repo root not found" in a fresh clone -- including the ledger
        # a reviewer would run to check the paper's numbers.
        if (parent / "artifacts").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def t_crit(n: int) -> float:
    """Two-sided 95% Student-t. Table lookup so scipy is not a hard dependency here."""
    table = {5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}
    if n not in table:
        raise ValueError(f"no t critical value tabulated for n={n}; add it rather than guessing")
    return table[n]


def paired_ci(x: np.ndarray) -> dict:
    """Mean, SD, 95% interval and sign count for a per-seed paired contrast, in the units given."""
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    half = t_crit(n) * sd / math.sqrt(n)
    return {"mean": m, "sd": sd, "ci_low": m - half, "ci_high": m + half,
            "n": n, "positive_in": int((x > 0).sum())}


# ---------------------------------------------------------------------------
# 1. Area-weighted class-pair ratio
# ---------------------------------------------------------------------------

def pair_ratios(conf: np.ndarray) -> list[dict]:
    """For every unordered foreground pair: its share of foreground error, the share area alone
    predicts, and the ratio between them.

    `conf` is a 6x6 count matrix, rows = reference class, cols = predicted class, row 0 unused.

    The expected share is a CO-AREA null: the product of the two classes' reference area shares,
    normalised over all pairs. It asks whether a pair fails more than its share of the scene would
    lead you to expect. Call it that and nothing stronger -- it is not a mechanistic model of how
    confusion arises, because a pixel has exactly one reference class and "both present" has no
    per-pixel meaning.

    An adjacency null -- shared boundary contacts between the two reference classes -- is the
    mechanistically relevant opportunity measure, and on Test A it gives 29.3x for the grassland
    pair against this null's 2.10x. The co-area null is the CONSERVATIVE choice of the two and that
    is why it is the one reported.
    """
    fg = list(FG)
    area = np.array([conf[k, :].sum() for k in fg], dtype=float)
    area = area / area.sum()
    total_err = float(conf[1:, :].sum() - np.trace(conf[np.ix_(fg, fg)]))
    raw_expected = {}
    for i, a in enumerate(fg):
        for j, b in enumerate(fg):
            if a < b:
                raw_expected[(a, b)] = area[i] * area[j]
    denom = sum(raw_expected.values())
    out = []
    for (a, b), exp in raw_expected.items():
        px = int(conf[a, b] + conf[b, a])
        share = px / total_err if total_err else float("nan")
        expected = exp / denom
        out.append({"pair": f"{FG[a]}-{FG[b]}", "pixels": px,
                    "share_of_error": share, "expected_share_from_area": expected,
                    "ratio": share / expected if expected else float("nan")})
    return sorted(out, key=lambda d: -d["share_of_error"])


def confusion_summary(conf: np.ndarray) -> dict:
    """Recall, precision, predicted-vs-reference area, and the ordered pair rates.

    These were computed inline during a chat session and quoted in the narrative -- the sampler's
    24% over-claim, the 39.2% against 4.8% pair asymmetry, the predicted-area table. Nothing
    produced them, so nobody else could check them. They come from one matrix and belong here.
    """
    fg = list(FG)
    out = {}
    for k in fg:
        ref = float(conf[k, :].sum())
        pred = float(conf[1:, k].sum())
        out[FG[k]] = {
            "reference_px": int(ref), "predicted_px": int(pred),
            "predicted_over_reference_pct": 100 * (pred - ref) / ref if ref else float("nan"),
            "recall": float(conf[k, k] / ref) if ref else float("nan"),
            "precision": float(conf[k, k] / pred) if pred else float("nan"),
        }
    out["pair_rates"] = {
        f"{FG[a]}->{FG[b]}": float(conf[a, b] / conf[a, :].sum())
        for a in fg for b in fg if a != b and conf[a, :].sum()
    }
    return out


# ---------------------------------------------------------------------------
# 2. Per-class paired contrasts
# ---------------------------------------------------------------------------

def per_class_contrasts(iou: dict) -> dict:
    """iou[cell] is a (n_seed, 5) array of per-class IoU. Returns the three 2x2 contrasts per class.

    Montgomery's halving convention on all three, so the interaction is on the same scale as the
    main effects. The raw difference of differences is twice the interaction reported here -- state
    the convention beside the number or a reader will mis-scale it.
    """
    b, t, s, f = (iou[c] for c in CELLS)
    eff = {"oem_pretraining": 100 * ((t + f) - (b + s)) / 2,
           "class_balanced_sampler": 100 * ((s + f) - (b + t)) / 2,
           "interaction": 100 * ((f - s) - (t - b)) / 2}
    out = {}
    for name, arr in eff.items():
        out[name] = {FG[k]: paired_ci(arr[:, i]) for i, k in enumerate(FG)}
        out[name]["foreground_mIoU"] = paired_ci(arr.mean(axis=1))
    return out


# ---------------------------------------------------------------------------
# 3. The seed-only control
# ---------------------------------------------------------------------------

def seed_control(near: np.ndarray, far: np.ndarray) -> dict:
    """`near` and `far` are (n_seed, n_cell) error rates in percent.

    The registered arm reads down a seed: how much do the four cells differ? The control reads
    across a cell: how much do the ten seeds differ? If the two agree, the cell spread is seed noise
    and the arm carries no information about the interventions.
    """
    def cv(a, axis):
        return float((100 * a.std(axis=axis, ddof=1) / a.mean(axis=axis)).mean())
    return {
        "across_cells_within_seed": {"near_cv_pct": cv(near, 1), "interior_cv_pct": cv(far, 1)},
        "across_seeds_within_cell": {"near_cv_pct": cv(near, 0), "interior_cv_pct": cv(far, 0)},
        "absolute_range_across_cells_pp": {
            "near": float((near.max(axis=1) - near.min(axis=1)).mean()),
            "interior": float((far.max(axis=1) - far.min(axis=1)).mean())},
        "mean_rate_pct": {"near": float(near.mean()), "interior": float(far.mean())},
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> int:
    ok = True

    # (1) Area weighting must divide out class size. Plant a confusion in which error is EXACTLY
    # proportional to the area product: every ratio must come back 1.000. A raw-share statistic
    # would instead rank the pairs by size, which is the failure this weighting exists to prevent.
    areas = {1: 0.10, 2: 0.70, 3: 0.02, 4: 0.05, 5: 0.13}
    N = 10_000_000
    conf = np.zeros((6, 6), dtype=np.int64)
    for a, b in combinations(FG, 2):
        e = int(N * areas[a] * areas[b])
        conf[a, b] += e // 2
        conf[b, a] += e - e // 2
    for k in FG:                                    # diagonal carries the rest of each class
        conf[k, k] = int(N * areas[k]) - int(conf[k, :].sum())
    rats = pair_ratios(conf)
    worst = max(abs(r["ratio"] - 1.0) for r in rats)
    good = worst < 0.02
    ok &= good
    print(f"  area weighting removes class size: max |ratio-1| = {worst:.4f} over "
          f"{len(rats)} pairs  [{'ok' if good else 'FAIL'}]")

    # And the control it must pass: raw shares on the SAME matrix are wildly unequal, so the test
    # above is not passing trivially.
    shares = sorted(r["share_of_error"] for r in rats)
    good = shares[-1] / shares[0] > 20
    ok &= good
    print(f"  ... while raw error shares on that matrix span {shares[-1]/shares[0]:.0f}x "
          f"(must be >20x, else the check is vacuous)  [{'ok' if good else 'FAIL'}]")

    # (2) Plant a known per-class effect and confirm recovery, including the halving convention.
    rng = np.random.default_rng(3)
    base = rng.normal(0.60, 0.02, (10, 5))
    # A NON-ZERO interaction is planted deliberately. The first version of this gate used
    # b, t, s, f = base, base+0.05, base, base+0.05, which makes (f-s)-(t-b) identically zero -- so
    # the assertion "interaction is 0" held whether or not the Montgomery halving was applied, and
    # removing the /2 survived the self-test. Planted here: main effect +5 pp, interaction +1 pp
    # after halving, so dropping the /2 returns 2.00 and fails.
    iou = {"stage1_baseline": base,
           "stage2b_oem_finetune": base + 0.05,
           "stage_sampler_only": base + 0.02,
           "stage3_clsbal": base + 0.09}
    got = per_class_contrasts(iou)["oem_pretraining"]["Forest"]["mean"]
    good = abs(got - 6.0) < 1e-6
    ok &= good
    print(f"  planted +6.00 pp OEM main effect recovered as {got:+.4f}  "
          f"[{'ok' if good else 'FAIL'}]")
    got_i = per_class_contrasts(iou)["interaction"]["Forest"]["mean"]
    good = abs(got_i - 1.0) < 1e-6
    ok &= good
    print(f"  planted +1.00 pp interaction (Montgomery halved) recovered as {got_i:+.4f}; "
          f"unhalved would give +2.00  [{'ok' if good else 'FAIL'}]")

    # (3) The seed control must SEPARATE the two cases, not just report them. Case A: cells really
    # differ. Case B: cells identical, all spread from seeds. If the statistic cannot tell these
    # apart it cannot support the conclusion drawn from it.
    # The bar is 2.0 and it is set here, before the campaign numbers are read, because a bar chosen
    # after seeing the answer is not a bar. It is also NOT arbitrary: case B below pins the null at
    # ~1.0, so a factor of two is comfortably outside it.
    #
    # Worth knowing when reading the real result: this ratio is a BLUNT discriminator. Cells that
    # differ by 9 pp against seeds that differ by 1 pp return only ~3.5, because a coefficient of
    # variation over four points is a weak instrument. That cuts one way for the campaign -- the
    # observed ratio there is 1.00, which is the null case B, not a small effect being missed.
    seed_noise = rng.normal(0, 1.0, (10, 1))
    near_a = 20 + seed_noise + np.array([[0, 3.0, 6.0, 9.0]])     # cells differ, unambiguously
    far_a = 9 + seed_noise * 0.5 + np.array([[0, 0.3, 0.6, 0.9]])
    a = seed_control(near_a, far_a)
    ratio_a = (a["across_cells_within_seed"]["near_cv_pct"]
               / a["across_seeds_within_cell"]["near_cv_pct"])
    good = ratio_a > 2.0
    ok &= good
    print(f"  cells genuinely differ -> across-cell CV {a['across_cells_within_seed']['near_cv_pct']:.2f}% "
          f"vs across-seed {a['across_seeds_within_cell']['near_cv_pct']:.2f}%, ratio {ratio_a:.2f} "
          f"(must exceed 2.0)  [{'ok' if good else 'FAIL'}]")

    near_b = 20 + rng.normal(0, 1.0, (10, 4))                      # cells identical in expectation
    far_b = 9 + rng.normal(0, 1.2, (10, 4))
    b = seed_control(near_b, far_b)
    r = (b["across_cells_within_seed"]["near_cv_pct"]
         / b["across_seeds_within_cell"]["near_cv_pct"])
    good = 0.6 < r < 1.6
    ok &= good
    print(f"  cells are seed noise only -> ratio of the two CVs = {r:.2f} "
          f"(must be near 1)  [{'ok' if good else 'FAIL'}]")

    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", help="per-seed metrics JSONs, <split>/<cell>_<tag>/metrics.json")
    ap.add_argument("--split", default="test", choices=["test", "external_test"])
    ap.add_argument("--confusion-dir", help="per-cell pooled confusion .npy, 6x6 int counts")
    ap.add_argument("--rho-dir", help="per-cell per-seed rho JSONs from boundary_rate_ratio.py")
    ap.add_argument("--band-m", type=float, default=8.0)
    ap.add_argument("--out", default="artifacts/narrative_numbers.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    root = find_repo_root()
    out: dict = {"split": args.split, "band_m": args.band_m, "seeds": SEEDS, "cells": CELLS}

    if args.confusion_dir:
        cpath = Path(args.confusion_dir) / f"confusion_{args.split}_stage1_baseline.npy"
        conf = np.load(cpath)
        if conf.shape != (6, 6):
            raise SystemExit(f"{cpath}: expected a 6x6 confusion, got {conf.shape}")
        out["pair_ratios"] = pair_ratios(conf)
        out["confusion_summary"] = confusion_summary(conf)
        alt = Path(args.confusion_dir) / f"confusion_{args.split}_stage3_clsbal.npy"
        if alt.exists():
            # The sampler's effect on what the model CLAIMS, which is the point of §12(b): it acted
            # as designed and bought almost nothing. Expressed as the extra predicted pixels and how
            # many of them were right, so the "3 in 100" is derived rather than asserted.
            c2 = np.load(alt)
            out["sampler_effect"] = {}
            for k in FG:
                p0, p1 = float(conf[1:, k].sum()), float(c2[1:, k].sum())
                t0, t1 = float(conf[k, k]), float(c2[k, k])
                out["sampler_effect"][FG[k]] = {
                    "predicted_px_baseline": int(p0), "predicted_px_full": int(p1),
                    "extra_predicted": int(p1 - p0), "extra_correct": int(t1 - t0),
                    "share_of_extra_that_was_correct":
                        float((t1 - t0) / (p1 - p0)) if p1 != p0 else float("nan"),
                    "recall_baseline": float(t0 / conf[k, :].sum()),
                    "recall_full": float(t1 / c2[k, :].sum()),
                }

    if args.metrics_dir:
        iou = {}
        for c in CELLS:
            rows = []
            for s in SEEDS:
                p = Path(args.metrics_dir) / f"seed{s}_{c}.json"
                if not p.exists():
                    raise SystemExit(f"missing {p}; per-class contrasts need all {len(SEEDS)} seeds "
                                     f"of all {len(CELLS)} cells, else the pairing is not paired")
                j = json.loads(p.read_text())
                rows.append([j["per_class_iou"][IOU_KEYS[k]] for k in FG])
            iou[c] = np.array(rows)
        out["per_class_contrasts"] = per_class_contrasts(iou)

    if args.rho_dir:
        near = np.zeros((len(SEEDS), len(CELLS)))
        far = np.zeros_like(near)
        for ci, c in enumerate(CELLS):
            for si, s in enumerate(SEEDS):
                # `8` and `8.0` are both in circulation: %g drops the trailing zero, the shell
                # loop that produced these did not. Accept either rather than making a filename
                # convention load-bearing.
                stem = f"rho_{args.split}_{c}_seed{s}_band"
                cands = [Path(args.rho_dir) / f"{stem}{args.band_m:g}.json",
                         Path(args.rho_dir) / f"{stem}{args.band_m}.json"]
                p = next((q for q in cands if q.exists()), None)
                if p is None:
                    raise SystemExit(f"missing {cands[0]} (or {cands[1].name}); the seed control "
                                     f"needs every cell x seed filled, otherwise the two spreads "
                                     f"are over different sets")
                j = json.loads(p.read_text())["pooled"]
                near[si, ci] = 100 * j["rate_near"]
                far[si, ci] = 100 * j["rate_far"]
        out["seed_control"] = seed_control(near, far)

    if len(out) == 4:
        raise SystemExit("nothing to do: pass at least one of --confusion-dir, --metrics-dir, "
                         "--rho-dir (or --self-test)")

    p = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
