#!/usr/bin/env python
"""
Boundary-tolerance (trimap) IoU recovery + error-vs-distance-to-boundary.

The label-ceiling KEYSTONE on single-label data. Two non-circular analyses, both using the
seed-ensemble argmax prediction and GROUND-TRUTH boundaries only (no repeated labels needed):

  A1  Trimap IoU recovery.  Re-score per-class IoU while EXCLUDING a band of width N px around
      GT class boundaries (keep = dist_to_boundary > N). If IoU jumps as the boundary shell is
      removed, the residual error is a boundary/label-ambiguity phenomenon, not an interior
      (capacity) failure. Standard trimap / boundary-band diagnostic (Kohli et al. 2009, origin;
      Csurka et al. 2013, the IoU-in-band variant), read in the complementary exclude-the-band direction.

  A2  Error rate vs distance-to-boundary.  The prediction-error mirror of the existing
      uncertainty-vs-distance curve (figure_label_ceiling N4). A spike at distance 0 decaying to a
      near-zero interior FLOOR is direct evidence the error is a boundary phenomenon; a non-zero
      far-from-boundary floor (esp. for Settlement) would instead flag an interior/capacity problem
      and FALSIFY the label-ambiguity reading for that class.

Reuses the validated primitives in seed_disagreement.py (boundary_distance, load_seed_stack,
load_mask, list_tiles, taxonomy). Predictions = argmax of the per-seed dumped softmax, which
reproduces the compute_metrics non-TTA per-class IoU exactly.

Usage:
    PYTHONPATH=. python scripts/analysis/boundary_trimap_iou.py \
        --softmax-root sonic/results --mask-dir data/biodiversity_split/val/masks \
        --cell stage1_baseline --cell stage3_clsbal --seeds 42 43 44 45 46 47 48 49 50 51 \
        --out-dir analysis/label_ceiling
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.seed_disagreement import (  # noqa: E402
    STUDENT_CLASSES, C, DIST_BIN_EDGES_PX, DIST_BIN_EDGES_M,
    boundary_distance, load_mask, load_seed_stack, list_val_tiles, seed_dir,
    tile_uncertainty,
)

import os  # noqa: E402
# The untagged paths belong to the WITHDRAWN random split.
SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")

FOREGROUND = list(range(1, C))                 # 1..5
HARD = {1: "Forest", 4: "Settlement", 5: "Seminatural"}  # narrative-focus classes
# Boundary-exclusion radii in px. -1 = no exclusion (the true whole-image baseline == compute_metrics).
RADII_PX = [-1, 0, 1, 2, 3, 4, 6, 8, 12, 16]
# boundary_distance() returns METRES (per-site anisotropic sampling), so exclusion radii and bin
# edges must be in metres as well. Digitising a metre distance against pixel edges would silently
# halve every band. -1 stays a sentinel for "no exclusion".
RADII_M = [(-1.0 if N < 0 else N * 0.5) for N in RADII_PX]
# Boundary-IoU band widths (Cheng et al. 2021), in px. Headline = 3 px = 1.5 m at 0.5 m GSD;
# swept 0.5-4 m so no single d is load-bearing.
BIOU_D_PX = [1, 2, 3, 4, 6, 8]


def boundary_band(m_bin: np.ndarray, d: int) -> np.ndarray:
    """Cheng et al. (2021) boundary region: pixels INSIDE m_bin within Euclidean distance d
    of the mask's contour. EDT of the binary mask gives, per True pixel, distance to the
    nearest False pixel; the band is those within d. Degrades gracefully for thin objects
    (whole object falls in the band -> Boundary IoU -> Mask IoU)."""
    if not m_bin.any():
        return m_bin
    edt = ndimage.distance_transform_edt(m_bin)
    return m_bin & (edt <= d)


def boundary_bands_multi(m_bin: np.ndarray, ds) -> dict:
    """One EDT, thresholded at every d (avoids recomputing the transform per band width)."""
    if not m_bin.any():
        return {d: m_bin for d in ds}
    edt = ndimage.distance_transform_edt(m_bin)
    return {d: m_bin & (edt <= d) for d in ds}


def conf_over(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """6x6 confusion (rows=gt, cols=pred) over the given 1-D pixel selections."""
    return np.bincount(C * gt.astype(np.int64) + pred.astype(np.int64),
                       minlength=C * C).reshape(C, C)


def iou_from_conf(conf: np.ndarray) -> dict:
    """Per-foreground-class IoU + macro from a (gt,pred) confusion. Excludes Background (0)."""
    out = {}
    ious = []
    for k in FOREGROUND:
        tp = float(conf[k, k])
        fp = float(conf[:, k].sum() - conf[k, k])
        fn = float(conf[k, :].sum() - conf[k, k])
        denom = tp + fp + fn
        v = tp / denom if denom > 0 else float("nan")
        out[STUDENT_CLASSES[k]] = v
        ious.append(v)
    out["macro_fg"] = float(np.nanmean(ious))
    return out


def run_cell(softmax_root, mask_dir, cell, seeds, out_dir):
    img_ids, dropped = list_val_tiles(softmax_root, seeds, cell, mask_dir)
    if dropped:
        print(f"[{cell}] scoring {len(img_ids)} Irish val tiles; dropped {len(dropped)} "
              f"non-Irish dump tiles ({', '.join(dropped[:6])}{'...' if len(dropped) > 6 else ''})")
    nR = len(RADII_M)
    # Global confusions: ensemble + one per seed, per radius.
    conf_ens = np.zeros((nR, C, C), dtype=np.int64)
    conf_seed = np.zeros((len(seeds), nR, C, C), dtype=np.int64)
    support = np.zeros((nR, C), dtype=np.int64)     # GT foreground support per radius
    n_kept = np.zeros(nR, dtype=np.int64)

    # A2: error-vs-distance accumulators (ensemble), overall + per HARD class.
    nb = len(DIST_BIN_EDGES_M) - 1
    err_n = np.zeros(nb, dtype=np.int64)            # foreground pixels per bin
    err_e = np.zeros(nb, dtype=np.int64)            # foreground errors per bin
    # per-class error vs distance for ALL foreground classes (panel c: boundary vs interior floor)
    err_n_k = {k: np.zeros(nb, dtype=np.int64) for k in FOREGROUND}
    err_e_k = {k: np.zeros(nb, dtype=np.int64) for k in FOREGROUND}
    # PER-SEED as well as ensemble, added 2026-07-27. A1 and A3 already accumulate per seed via
    # `seed_preds`; A2 was the only block left ensemble-only, which meant the boundary and interior
    # error rates -- and therefore rho -- carried no seed-level variation and so no uncertainty at
    # all. Since this study's uncertainty is per-seed and paired (METHODS §7), that made the headline
    # boundary numbers the one place with no interval available. It also biased them: an ensemble
    # removes more error in easy interior regions than at hard boundaries, so rho computed on the
    # ensemble argmax is higher than the mean of the per-seed rhos, in the direction of the claim.
    # `seed_preds` is already computed for A1/A3, so this costs one bincount per seed per tile.
    err_n_seed = np.zeros((len(seeds), nb), dtype=np.int64)
    err_e_seed = np.zeros((len(seeds), nb), dtype=np.int64)
    ent_sum = np.zeros(nb)                           # sum of ensemble total entropy H[mean_p], fg

    # A3 (panel c): Boundary-IoU per class per seed, micro-pooled over tiles, at each band d.
    nD = len(BIOU_D_PX)
    biou_inter = np.zeros((nD, len(seeds), C), dtype=np.int64)
    biou_union = np.zeros((nD, len(seeds), C), dtype=np.int64)

    n_scored = 0
    for iid in img_ids:
        stack = load_seed_stack(softmax_root, seeds, cell, iid)   # (N,C,H,W) float32
        mask = load_mask(mask_dir, iid)
        if mask.shape != stack.shape[-2:]:
            raise ValueError(f"shape mismatch {iid}: mask {mask.shape} vs {stack.shape}")
        dist = boundary_distance(mask, iid)
        if not np.isfinite(dist[mask != 0]).any():
            continue    # no GT boundary: EXCLUDED, as in boundary_rate_ratio.per_tile_counts
        n_scored += 1
        ens_pred = stack.mean(axis=0).argmax(axis=0).astype(np.int64)
        seed_preds = stack.argmax(axis=1).astype(np.int64)        # (N,H,W)
        fg = mask != 0

        # --- A1: trimap recovery per radius ---
        for ri, N in enumerate(RADII_M):
            keep = fg & (dist > N)
            if not keep.any():
                continue
            gt_k = mask[keep]
            n_kept[ri] += int(keep.sum())
            for k in FOREGROUND:
                support[ri, k] += int((gt_k == k).sum())
            conf_ens[ri] += conf_over(gt_k, ens_pred[keep])
            for si in range(len(seeds)):
                conf_seed[si, ri] += conf_over(gt_k, seed_preds[si][keep])

        # --- A3: Boundary-IoU per class per seed (one EDT per mask, reused over d; GT reused over seeds) ---
        gt_bands = {k: boundary_bands_multi(mask == k, BIOU_D_PX) for k in FOREGROUND}
        for si in range(len(seeds)):
            for k in FOREGROUND:
                pb = boundary_bands_multi(seed_preds[si] == k, BIOU_D_PX)
                for di, d in enumerate(BIOU_D_PX):
                    gb = gt_bands[k][d]
                    biou_inter[di, si, k] += int((gb & pb[d]).sum())
                    biou_union[di, si, k] += int((gb | pb[d]).sum())

        # --- A2: error + ensemble entropy vs distance ---
        total_H, _, _ = tile_uncertainty(stack)                  # H[mean_p], (H,W) in [0,1]
        err = (ens_pred != mask) & fg
        bidx = np.digitize(dist, DIST_BIN_EDGES_M[1:-1])          # 0..nb-1 (metres)
        fg_b = bidx[fg]
        err_b = bidx[err]
        err_n += np.bincount(fg_b, minlength=nb)
        err_e += np.bincount(err_b, minlength=nb)
        ent_sum += np.bincount(fg_b, weights=total_H[fg].astype(float), minlength=nb)
        for k in FOREGROUND:
            sel_k = (mask == k)
            err_n_k[k] += np.bincount(bidx[sel_k], minlength=nb)
            err_e_k[k] += np.bincount(bidx[sel_k & (ens_pred != mask)], minlength=nb)
        # the same stratification, per seed, so the rates carry a spread over training runs
        for si in range(len(seeds)):
            err_n_seed[si] += np.bincount(fg_b, minlength=nb)
            err_e_seed[si] += np.bincount(bidx[fg & (seed_preds[si] != mask)], minlength=nb)

    # --- reduce A1 ---
    ens_curve = [iou_from_conf(conf_ens[ri]) for ri in range(nR)]
    seed_curve = np.array([[ [iou_from_conf(conf_seed[si, ri])[STUDENT_CLASSES[k]]
                              for k in FOREGROUND] for ri in range(nR)]
                            for si in range(len(seeds))])          # (nSeed, nR, 5)
    # Every tile can be boundary-free -- on a `--mask-dir` subset of ireland2, all of them are.
    # `list_val_tiles` only guards the dump/mask intersection, so without this the run returns
    # NaN at every radius and exits 0. `boundary_rate_ratio.py` has the matching guard; this
    # path did not, which is the shape of defect C7 in the pre-submission ledger.
    if n_scored == 0:
        raise SystemExit(
            f"no scorable tiles for cell {cell}: all {len(img_ids)} tiles lack a GT boundary, "
            f"so every distance stratum is empty. Check --mask-dir.")

    seed_macro = np.nanmean(seed_curve, axis=2)                    # (nSeed, nR)

    radii_m = [(-0.5 if N < 0 else N) for N in RADII_M]            # -0.5 marks the 'none' baseline slot
    # per-class per-seed recovery (FLAG-1: headline panel a uses per-seed mean, NOT ensemble argmax)
    cls_mean = np.nanmean(seed_curve, axis=0)                      # (nR, 5)
    cls_std = np.nanstd(seed_curve, axis=0)                        # (nR, 5)
    recovery = {
        "cell": cell, "n_seeds": len(seeds), "n_tiles": n_scored,
        "n_tiles_boundary_free_excluded": len(img_ids) - n_scored,
        "radii_px": RADII_PX, "radii_m": radii_m,
        "ensemble_iou_per_radius": ens_curve,
        "per_seed_class_iou_mean": {STUDENT_CLASSES[FOREGROUND[j]]: cls_mean[:, j].tolist()
                                    for j in range(len(FOREGROUND))},
        "per_seed_class_iou_std": {STUDENT_CLASSES[FOREGROUND[j]]: cls_std[:, j].tolist()
                                   for j in range(len(FOREGROUND))},
        "support_per_radius": {STUDENT_CLASSES[k]: support[:, k].tolist() for k in FOREGROUND},
        "n_kept_per_radius": n_kept.tolist(),
        "per_seed_macro_iou": seed_macro.tolist(),
        "per_seed_macro_mean": np.nanmean(seed_macro, axis=0).tolist(),
        "per_seed_macro_std": np.nanstd(seed_macro, axis=0).tolist(),
    }

    # --- reduce A3: Boundary-IoU vs standard IoU per class (per seed -> mean over seeds) ---
    std_iou_seed = np.array([[iou_from_conf(conf_seed[si, 0])[STUDENT_CLASSES[k]]
                              for k in FOREGROUND] for si in range(len(seeds))])   # (nSeed,5)
    with np.errstate(divide="ignore", invalid="ignore"):
        biou_seed = np.where(biou_union > 0, biou_inter / np.maximum(biou_union, 1), np.nan)  # (nD,nSeed,C)
    biou_fg = biou_seed[:, :, FOREGROUND]                          # (nD, nSeed, 5)
    boundary_iou = {
        # PIXELS, not metres. boundary_bands_multi runs an UNSAMPLED EDT, so d is a pixel
        # radius. A `d_m = d * GSD_M` field was emitted here until 2026-07-28; it is right on
        # the inland site (0.500 m/px) and wrong on the uplands (0.515 x 0.641 m), where the
        # band is also anisotropic. Cheng et al. define Boundary IoU in pixels, so pixels are
        # reported and no ground-distance equivalent is offered.
        "d_px": BIOU_D_PX, "headline_d_px": 3,
        "standard_iou_mean": {STUDENT_CLASSES[FOREGROUND[j]]: float(np.nanmean(std_iou_seed[:, j]))
                              for j in range(len(FOREGROUND))},
        "standard_iou_std": {STUDENT_CLASSES[FOREGROUND[j]]: float(np.nanstd(std_iou_seed[:, j]))
                             for j in range(len(FOREGROUND))},
        "boundary_iou_mean": {STUDENT_CLASSES[FOREGROUND[j]]: np.nanmean(biou_fg[:, :, j], axis=1).tolist()
                              for j in range(len(FOREGROUND))},
        "boundary_iou_std": {STUDENT_CLASSES[FOREGROUND[j]]: np.nanstd(biou_fg[:, :, j], axis=1).tolist()
                             for j in range(len(FOREGROUND))},
    }

    # --- reduce A2 ---
    def rate(e, n):
        n = np.where(n > 0, n, 1)
        return (e / n).tolist()
    ent_n = np.where(err_n > 0, err_n, 1)
    edges_m = DIST_BIN_EDGES_M
    # boundary band <= 1.5 m (matches headline d=3px); interior > 8 m (the floor).
    # TWO DIFFERENT STATISTICS, TWO DIFFERENT NAMES (2026-07-27).
    # This block's near band is 1.5 m, matching Cheng's Boundary-IoU distance; boundary_rate_ratio.py
    # uses the registered 8 m band. They are NOT the same partition -- 71.75% of the 8 m band lies
    # beyond 1.5 m, so the 1.5-8 m annulus is in neither set here -- and until now both ratios were
    # called "rho". The 8 m ratio is rho. This one is the CONTACT-ZONE ratio and must never be
    # reported under the same name or in the same column.
    BND_MAX_M, INT_MIN_M = 1.5, 8.0        # contact-zone band / deep-interior floor
    bnd_bins = np.where(edges_m[1:] <= BND_MAX_M + 1e-9)[0]      # bins fully within 1.5 m
    int_bins = np.where(edges_m[:-1] >= INT_MIN_M - 1e-9)[0]     # bins starting beyond 8 m

    def agg_rate(e, n, bins):
        N = int(n[bins].sum())
        return (float(e[bins].sum()) / N if N > 0 else float("nan")), N

    bvi = {}
    for k in FOREGROUND:
        be, bn = agg_rate(err_e_k[k], err_n_k[k], bnd_bins)
        ie, in_ = agg_rate(err_e_k[k], err_n_k[k], int_bins)
        bvi[STUDENT_CLASSES[k]] = {"boundary_error": be, "boundary_n": bn,
                                   "interior_error": ie, "interior_n": in_}

    err_curve = {
        "edges_px": DIST_BIN_EDGES_PX.tolist(),
        "edges_m": edges_m.tolist(),
        "n_foreground": err_n.tolist(),
        "error_rate_foreground": rate(err_e, err_n),
        "entropy_total_foreground": (ent_sum / ent_n).tolist(),
        "per_class": {STUDENT_CLASSES[k]: {"n": err_n_k[k].tolist(),
                                           "error_rate": rate(err_e_k[k], err_n_k[k])}
                      for k in FOREGROUND},
        "contact_zone_vs_interior": {"statistic": "contact-zone ratio, NOT rho (rho is the 8 m band in boundary_rate_ratio.py)",
                                 "contact_max_m": BND_MAX_M, "interior_min_m": INT_MIN_M,
                                 "per_class": bvi},
        # PER-SEED boundary and interior rates, and the CONTACT-ZONE RATIO each seed implies. It is
        # NOT rho: the near band here is 1.5 m, not the registered 8 m, so this key carried the
        # wrong statistic's name until 2026-07-28 — exactly what the note above forbids. This is
        # what carries the uncertainty: a spread over training runs, which is the only interval
        # this study reports (METHODS §7). The ensemble figures above stay as the registered point
        # estimate; they are NOT the centre of this spread, because the ratio is a ratio and the
        # ensemble removes more interior error than boundary error. Report the gap, do not hide it.
        "per_seed": {
            "seeds": list(seeds),
            "contact_max_m": BND_MAX_M,
            "interior_min_m": INT_MIN_M,
            "boundary_error_rate": [agg_rate(err_e_seed[si], err_n_seed[si], bnd_bins)[0]
                                    for si in range(len(seeds))],
            "interior_error_rate": [agg_rate(err_e_seed[si], err_n_seed[si], int_bins)[0]
                                    for si in range(len(seeds))],
            "contact_zone_ratio": [
                (agg_rate(err_e_seed[si], err_n_seed[si], bnd_bins)[0]
                 / agg_rate(err_e_seed[si], err_n_seed[si], int_bins)[0])
                if agg_rate(err_e_seed[si], err_n_seed[si], int_bins)[0] > 0 else float("nan")
                for si in range(len(seeds))],
        },
    }

    out = {"recovery_trimap": recovery, "error_vs_distance": err_curve, "boundary_iou": boundary_iou}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"boundary_trimap_{cell}.json").write_text(json.dumps(out, indent=2))

    # --- console keystone summary ---
    base = ens_curve[0]            # N=-1, no exclusion
    print(f"\n[{cell}]  trimap IoU recovery (ensemble argmax, {len(seeds)} seeds, {n_scored} tiles; "
          f"{len(img_ids) - n_scored} boundary-free tiles excluded)")
    print(f"  {'class':12s} {'baseline':>9s} {'-1px':>7s} {'-2px':>7s} {'-4px':>7s} {'-8px':>7s}  recovery(0->8px)")
    cols = {-1: 0, 1: RADII_M.index(0.5), 2: RADII_M.index(1.0), 4: RADII_M.index(2.0), 8: RADII_M.index(4.0)}
    for name in [STUDENT_CLASSES[k] for k in HARD] + ["macro_fg"]:
        b = base[name]
        vals = [ens_curve[cols[r]][name] for r in (1, 2, 4, 8)]
        print(f"  {name:12s} {b:9.3f} " + " ".join(f"{v:7.3f}" for v in vals) +
              f"   +{(vals[-1]-b)*100:5.1f}pp")
    di = BIOU_D_PX.index(3)
    print(f"  Boundary-IoU vs standard IoU (per-seed mean, d=3px=1.5m):")
    for k in FOREGROUND:
        name = STUDENT_CLASSES[k]
        s = boundary_iou["standard_iou_mean"][name]
        bdry = boundary_iou["boundary_iou_mean"][name][di]
        print(f"    {name:12s} std={s:.3f}  bdry={bdry:.3f}  gap={(s-bdry)*100:5.1f}pp")
    return out


def make_figure(results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3})
    cells = list(results)
    fig, axes = plt.subplots(2, len(cells), figsize=(5.2 * len(cells), 7.2), squeeze=False)
    fg_names = [STUDENT_CLASSES[1], STUDENT_CLASSES[4], STUDENT_CLASSES[5]]
    colours = {STUDENT_CLASSES[1]: "#2c7", STUDENT_CLASSES[4]: "#c33",
               STUDENT_CLASSES[5]: "#36c", "macro_fg": "#444"}
    for ci, cell in enumerate(cells):
        rec = results[cell]["recovery_trimap"]
        xm = [0.0 if m < 0 else m for m in rec["radii_m"]]   # baseline plotted at x=0
        axA = axes[0][ci]
        for name in fg_names + ["macro_fg"]:
            y = [rec["ensemble_iou_per_radius"][ri][name] for ri in range(len(xm))]
            axA.plot(xm, y, "-o", ms=3, color=colours[name], label=name)
        axA.set_title(f"{cell}: IoU vs boundary exclusion"); axA.set_xlabel("excluded boundary band (m)")
        axA.set_ylabel("IoU"); axA.legend(fontsize=8)
        # A2
        ev = results[cell]["error_vs_distance"]
        cen = [(a + b) / 2 for a, b in zip(ev["edges_m"][:-1], ev["edges_m"][1:-1] + [ev["edges_m"][-2] * 1.5])]
        axB = axes[1][ci]
        axB.plot(cen, ev["error_rate_foreground"], "-o", ms=3, color="#444", label="all foreground")
        for k, name in [("Forest", "Forest"), ("Settlement", "Settlement"), ("Seminatural", "Seminatural")]:
            cmap = {"Forest": "#2c7", "Settlement": "#c33", "Seminatural": "#36c"}
            axB.plot(cen, ev["per_class"][name]["error_rate"], "-o", ms=3, color=cmap[name], label=name)
        axB.set_xscale("log"); axB.set_title(f"{cell}: error rate vs distance-to-boundary")
        axB.set_xlabel("distance to GT boundary (m, log)"); axB.set_ylabel("misclass. rate"); axB.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[fig] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--softmax-root", default="sonic/results",
                    help="root holding seed<N>/analysis/seed_softmax/<cell>/seed<N>/")
    ap.add_argument("--mask-dir", default=f"data/split_{SPLIT_TAG}/test/masks",
                    help="masks to score against. Defaults to Test A. The withdrawn split's "
                         "val/ directory was the old default; validation is the split every "
                         "checkpoint is selected on, so the boundary evidence must not come "
                         "from it.")
    ap.add_argument("--cell", action="append", dest="cells", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--out-dir", default="analysis/label_ceiling")
    args = ap.parse_args()
    cells = args.cells or ["stage1_baseline", "stage3_clsbal"]
    results = {}
    for cell in cells:
        results[cell] = run_cell(args.softmax_root, args.mask_dir, cell, args.seeds, args.out_dir)
    make_figure(results, Path(args.out_dir) / "boundary_trimap_preview.png")


if __name__ == "__main__":
    main()
