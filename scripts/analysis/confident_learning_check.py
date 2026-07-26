"""Independent label-error cross-check with confident learning (cleanlab).

Purpose: our label-quality-ceiling diagnosis locates the residual error at class
boundaries and on spectrally overlapping class pairs using a ten-seed ensemble
(entropy / aleatoric-epistemic decomposition). This script asks whether an
orthogonal, community-standard label-error detector, confident learning
(Northcutt et al., implemented in cleanlab), independently flags the SAME pixels.

It reuses the exact loaders, masks and boundary logic of seed_disagreement.py so
the comparison is on identical data. It does NOT retrain or relabel anything: it
is a localisation cross-check, not an accuracy horse-race (measuring "improvement"
here would require a clean reference we do not have, and would be circular).

Run (label-quality-ceiling env):
    python scripts/analysis/confident_learning_check.py \
        --softmax-root sonic/results \
        --mask-dir data/biodiversity_split/val/masks \
        --cell stage3_clsbal --seeds 42 43 44 45 46 47 48 49 50 51
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.seed_disagreement import (  # noqa: E402
    load_seed_stack, load_mask, boundary_distance, tile_uncertainty,
    list_val_tiles, GSD_M,
)
from geoseg.taxonomy import STUDENT_CLASSES  # noqa: E402

import os  # noqa: E402
# The untagged split directories belong to the WITHDRAWN random split (219 val / 218 test tiles).
SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")


C = len(STUDENT_CLASSES)                 # 6 (0 = background, 1..5 = foreground)
FOREGROUND = np.arange(1, C)
# Metres. These were BND_M/INT_M = 3.0/16.0 applied to distances that boundary_distance
# already returns in metres, so the comments said 1.5 m and 8 m while the code did 3 m and
# 16 m, and the output keys named _within_8m held within-16 m shares (fixed 2026-07-26).
BND_M = 1.5                              # boundary band
INT_M = 8.0                              # deep interior


def build_arrays(softmax_root, mask_dir, cell, seeds):
    ids, dropped = list_val_tiles(softmax_root, seeds, cell, mask_dir)
    print(f"[data] {len(ids)} val tiles ({len(dropped)} dumped tiles without an Irish mask dropped)")
    n = len(ids)
    H = W = 512
    labels = np.empty((n, H, W), dtype=np.int16)
    pred = np.empty((n, C, H, W), dtype=np.float32)
    ent = np.empty((n, H, W), dtype=np.float32)
    dist = np.empty((n, H, W), dtype=np.float32)
    for i, iid in enumerate(ids):
        stack = load_seed_stack(softmax_root, seeds, cell, iid)     # (N,C,H,W)
        mean_p = stack.mean(axis=0)                                 # (C,H,W)
        total, _expected, _mi = tile_uncertainty(stack)            # total entropy in [0,1]
        m = load_mask(mask_dir, iid)
        if m.shape != (H, W):
            raise ValueError(f"{iid}: mask {m.shape} != (512,512)")
        labels[i] = m
        pred[i] = mean_p
        ent[i] = total
        dist[i] = boundary_distance(m, iid)
        if (i + 1) % 50 == 0:
            print(f"[data] loaded {i + 1}/{n}")
    return ids, labels, pred, ent, dist


def enrichment(flag, dist, fg):
    """P(flag | boundary band) vs P(flag | deep interior), foreground pixels only."""
    bnd = fg & (dist <= BND_M)
    inter = fg & (dist > INT_M)
    p_bnd = flag[bnd].mean() if bnd.any() else np.nan
    p_int = flag[inter].mean() if inter.any() else np.nan
    return {
        "flag_rate_boundary_band_<=1.5m": float(p_bnd),
        "flag_rate_deep_interior_>8m": float(p_int),
        "enrichment_boundary_over_interior": float(p_bnd / p_int) if p_int else float("inf"),
        "n_boundary_px": int(bnd.sum()), "n_interior_px": int(inter.sum()),
    }


def within_band_share(flag, dist, fg, edge_px):
    """Of all flagged foreground pixels, what share fall within edge_px of a boundary."""
    ff = flag & fg
    if not ff.any():
        return float("nan")
    return float((dist[ff] <= edge_px).mean())


def confusion_pairs(flag, labels, argmax, fg, top=12):
    """Top GT-class -> predicted-class pairs among flagged foreground pixels."""
    sel = flag & fg & (labels != argmax)
    g = labels[sel]; p = argmax[sel]
    pairs, counts = np.unique(np.stack([g, p], 1), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:top]
    tot = counts.sum()
    out = []
    for j in order:
        gi, pi = int(pairs[j, 0]), int(pairs[j, 1])
        out.append({
            "gt": STUDENT_CLASSES[gi], "pred": STUDENT_CLASSES[pi],
            "n": int(counts[j]), "share_of_flagged": float(counts[j] / tot),
        })
    return out


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
    argmax = pred.argmax(axis=1).astype(np.int16)
    fg = np.isin(labels, FOREGROUND)
    err = fg & (argmax != labels)

    from cleanlab.segmentation.filter import find_label_issues
    from cleanlab.segmentation.rank import get_label_quality_scores
    print("[cleanlab] running confident learning (find_label_issues, n_jobs=1)...")
    issues = find_label_issues(labels.astype(np.int64), pred, n_jobs=1, verbose=True)
    _img_scores, pixel_scores = get_label_quality_scores(labels.astype(np.int64), pred, n_jobs=1)

    n_px = labels.size
    n_fg = int(fg.sum())
    flagged_fg = int((issues & fg).sum())

    # 1) how much is flagged
    overview = {
        "n_tiles": len(ids), "n_pixels": int(n_px), "n_foreground_px": n_fg,
        "flagged_all_px": int(issues.sum()),
        "flagged_all_frac": float(issues.mean()),
        "flagged_foreground_px": flagged_fg,
        "flagged_foreground_frac": float(flagged_fg / n_fg),
    }

    # 2) do cleanlab flags concentrate at GT boundaries the way our error does?
    boundary = {
        "cleanlab_flags": enrichment(issues, dist, fg),
        "model_error (reference)": enrichment(err, dist, fg),
        "share_of_cleanlab_flags_within_8m": within_band_share(issues, dist, fg, INT_M),
        "share_of_model_error_within_8m": within_band_share(err, dist, fg, INT_M),
        "share_of_cleanlab_flags_within_2m": within_band_share(issues, dist, fg, 4.0),
    }

    # 3) do cleanlab flags agree with our ensemble entropy (aleatoric map)?
    ent_flag = float(ent[issues & fg].mean()) if flagged_fg else float("nan")
    ent_unflag = float(ent[(~issues) & fg].mean())
    # continuous agreement: (1 - cleanlab pixel quality) vs our entropy, on a subsample
    rng = np.random.default_rng(0)
    idx = rng.choice(n_fg, size=min(1_000_000, n_fg), replace=False)
    q = (1.0 - pixel_scores[fg])[idx]          # high = cleanlab thinks likely error
    e = ent[fg][idx]                           # high = ensemble uncertain
    pear = float(np.corrcoef(q, e)[0, 1])
    agreement = {
        "mean_entropy_on_cleanlab_flagged_fg": ent_flag,
        "mean_entropy_on_unflagged_fg": ent_unflag,
        "entropy_ratio_flagged_over_unflagged": ent_flag / ent_unflag if ent_unflag else float("nan"),
        "pearson_(1-cleanlab_quality)_vs_ensemble_entropy": pear,
    }

    # 4) which class pairs do the flags fall on?
    pairs = confusion_pairs(issues, labels, argmax, fg)

    result = {
        "cell": args.cell, "seeds": args.seeds,
        "note": "Localisation cross-check only; no retraining/relabelling. "
                "pred_probs = ten-seed ensemble mean (out-of-sample on val).",
        "overview": overview,
        "boundary_concentration": boundary,
        "agreement_with_ensemble_entropy": agreement,
        "top_flagged_class_pairs": pairs,
    }

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = out / "confident_learning_check.json"
    fp.write_text(json.dumps(result, indent=2))

    print("\n================ CONFIDENT LEARNING CROSS-CHECK ================")
    print(json.dumps(result, indent=2))
    print(f"\n[written] {fp}")


if __name__ == "__main__":
    main()
