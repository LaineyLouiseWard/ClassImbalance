"""Foreground-only confident-learning cross-check (background excluded).

Correctness fix over confident_learning_check.py: the semantic-segmentation
cleanlab API has no ignore-class option, so it scored the Background class
(the "outside-farm" catch-all the manuscript excludes from every metric).
Background is only 4.8% of pixels but the model predicts real land cover there,
so confident learning flags ~17% of it and that noise dominates the global
joint-distribution estimate, suppressing foreground flags.

Here we instead treat each FOREGROUND pixel as a classification sample, drop
background entirely, and renormalise the predicted probabilities over the five
foreground classes. This is the canonical confident-learning setting (Northcutt
et al. 2021) run through cleanlab's classification API, consistent with the
paper's foreground-only evaluation. Still a localisation cross-check, not a
retrain/relabel experiment.

Run (label-quality-ceiling env):
    python scripts/analysis/confident_learning_foreground.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.confident_learning_check import build_arrays  # noqa: E402
from geoseg.taxonomy import STUDENT_CLASSES  # noqa: E402

import os  # noqa: E402
# The untagged split directories belong to the WITHDRAWN random split (219 val / 218 test tiles).
SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")


FG_NAMES = STUDENT_CLASSES[1:]          # Forest, Grassland, Cropland, Settlement, Seminatural
BND_PX, INT_PX = 3.0, 16.0              # <=1.5 m boundary band; >8 m deep interior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--softmax-root", default="sonic/results")
    ap.add_argument("--mask-dir", default=f"data/split_{SPLIT_TAG}/val/masks")
    ap.add_argument("--cell", default="stage3_clsbal")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--out-dir", default="analysis/label_ceiling")
    args = ap.parse_args()

    ids, labels, pred, ent, dist = build_arrays(
        args.softmax_root, args.mask_dir, args.cell, args.seeds)

    fg = np.isin(labels, np.arange(1, 6))
    # gather foreground pixels as classification samples
    probs = np.transpose(pred, (0, 2, 3, 1))[fg]        # (M, 6)
    probs = probs[:, 1:6]                                # drop background channel
    probs = probs / probs.sum(axis=1, keepdims=True)    # renormalise over 5 fg classes
    y = (labels[fg] - 1).astype(np.int64)               # 0..4
    d = dist[fg]
    e = ent[fg]
    argmax = probs.argmax(axis=1)
    M = y.size
    print(f"[fg-only] {M:,} foreground pixels, {len(FG_NAMES)} classes")

    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    print("[cleanlab] classification confident learning on foreground pixels...")
    probs32 = probs.astype(np.float32)
    issues = find_label_issues(y, probs32, filter_by="both", n_jobs=1)   # boolean (M,)
    qual = get_label_quality_scores(y, probs32)                          # per-pixel, low = likely issue
    rng = np.random.default_rng(0)
    sub = rng.choice(M, size=min(2_000_000, M), replace=False)
    pearson_qual_entropy = float(np.corrcoef(1.0 - qual[sub], e[sub])[0, 1])

    nflag = int(issues.sum())
    # boundary concentration
    bnd = d <= BND_PX
    inter = d > INT_PX
    p_bnd = issues[bnd].mean()
    p_int = issues[inter].mean()
    # class pairs among flagged
    sel = issues & (argmax != y)
    pairs, cnt = np.unique(np.stack([y[sel], argmax[sel]], 1), axis=0, return_counts=True)
    order = np.argsort(cnt)[::-1][:10]
    top_pairs = [{"gt": FG_NAMES[int(pairs[j, 0])], "pred": FG_NAMES[int(pairs[j, 1])],
                  "n": int(cnt[j]), "share": float(cnt[j] / cnt.sum())} for j in order]
    # per-class flag rate (does CL rank the same classes hardest?)
    per_class = {}
    for c in range(len(FG_NAMES)):
        m = y == c
        per_class[FG_NAMES[c]] = {
            "flag_rate": float(issues[m].mean()) if m.any() else float("nan"),
            "n_px": int(m.sum()), "n_flagged": int((issues & m).sum()),
        }

    # entropy agreement
    ent_flag = float(e[issues].mean()) if nflag else float("nan")
    ent_unflag = float(e[~issues].mean())

    result = {
        "mode": "foreground_only_classification_confident_learning",
        "cell": args.cell, "seeds": args.seeds,
        "n_foreground_px": M,
        "flagged_px": nflag,
        "flagged_frac_of_foreground": float(nflag / M),
        "boundary_concentration": {
            "flag_rate_boundary_<=1.5m": float(p_bnd),
            "flag_rate_interior_>8m": float(p_int),
            "enrichment": float(p_bnd / p_int) if p_int else float("inf"),
            "share_of_flags_within_8m": float((d[issues] <= INT_PX).mean()) if nflag else float("nan"),
        },
        "entropy_agreement": {
            "mean_entropy_flagged": ent_flag,
            "mean_entropy_unflagged": ent_unflag,
            "ratio": ent_flag / ent_unflag if ent_unflag else float("nan"),
            "pearson_(1-quality)_vs_entropy": pearson_qual_entropy,
        },
        "per_class_flag_rate": per_class,
        "top_flagged_class_pairs": top_pairs,
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = out / "confident_learning_foreground.json"
    fp.write_text(json.dumps(result, indent=2))
    print("\n============ FOREGROUND-ONLY CONFIDENT LEARNING ============")
    print(json.dumps(result, indent=2))
    print(f"\n[written] {fp}")


if __name__ == "__main__":
    main()
