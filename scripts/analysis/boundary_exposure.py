#!/usr/bin/env python3
"""
scripts/analysis/boundary_exposure.py

Boundary-exposure vs baseline difficulty (§4.4). For each foreground class, the fraction of
its ground-truth pixels lying within a fixed band of a class boundary ("boundary exposure"),
regressed against that class's baseline per-class IoU. Tests whether class difficulty is a
geometric property (boundary exposure) rather than a function of class frequency.

Ireland-only: uses the current split's validation masks (data/split_$SPLIT_TAG/val/masks), the
same set the per-class IoU is scored on. Boundary distance = Euclidean distance (scipy EDT) from each
class pixel to the nearest edge of its own class, in metres (GSD 0.5 m/px).

Baseline per-class IoU is supplied by --baseline-iou. It is NOT defaulted: this read
analysis/eval_219/per_class_iou.json unconditionally until 2026-07-26, which is the withdrawn
leaking campaign's output.

Output: analysis/label_ceiling/boundary_exposure.json (+ printed summary).
Run: PYTHONPATH=. python scripts/analysis/boundary_exposure.py --baseline-iou <current.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import os  # noqa: E402
# The untagged split directories belong to the WITHDRAWN random split (219 val / 218 test tiles).
SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")


GSD_M = 0.5  # NOTE: inland only; see GSD_BY_SITE
# Per-site ground sample distance: ONE definition, in geoseg/geo.py. This exact block was
# duplicated verbatim in four files, and gsd_for() silently returned the inland (0.5, 0.5) for any
# site it did not recognise -- rescaling every boundary distance by up to 28% in x without a word.
# It now raises. geoseg/geo.py also records why the registered values are the spherical
# approximation rather than the 0.2%-larger geodesic one.
from geoseg.geo import GSD_BY_SITE, gsd_for  # noqa: E402,F401

BAND_M = 2.0
NAMES = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}


def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "artifacts").is_dir() and (parent / "scripts").is_dir():  # not data/: gitignored
            return parent
    raise RuntimeError("repo root not found")


def load_mask(path: str) -> np.ndarray:
    from PIL import Image
    m = np.array(Image.open(path))
    return m[..., 0] if m.ndim == 3 else m


def main() -> None:
    ap = argparse.ArgumentParser()
    # No default, and the withdrawn path is refused outright. This read
    #     analysis/eval_219/per_class_iou.json
    # unconditionally and untagged until 2026-07-26. That file is the WITHDRAWN campaign's output:
    # 219 val tiles of the random-by-tile split, whose held-out tiles share ~93% of their ground
    # with training, giving leakage-inflated values (Cropland IoU 0.962, foreground mIoU 0.877).
    # Nothing in RUNBOOK.sh or build_all_figures.py calls this script, so it never fired -- it was
    # a landmine rather than a fire, and the fix is to disarm it rather than to note it.
    ap.add_argument("--baseline-iou", required=True,
                    help="JSON of per-class IoU for the baseline cell, from the CURRENT campaign "
                         "(analysis/seed_aggregate/, or a per-cell metrics.json).")
    ap.add_argument("--cell", default="stage1_baseline")
    args = ap.parse_args()

    root = find_repo_root()
    iou_path = Path(args.baseline_iou).resolve()
    if "eval_219" in iou_path.parts or "_archive" in iou_path.parts:
        raise SystemExit(
            f"{iou_path}\n  belongs to the withdrawn pre-2026-07-26 campaign (the leaking "
            f"219-tile random split). Point --baseline-iou at the current campaign's output.")
    masks = sorted(glob.glob(str(root / f"data/split_{SPLIT_TAG}/val/masks/*")))
    iou = json.load(open(iou_path))[args.cell]["per_class_iou_mean"]

    within = {c: [] for c in NAMES}   # per-pixel indicator lists, per class
    for f in masks:
        m = load_mask(f)
        for c in NAMES:
            b = m == c
            if not b.any():
                continue
            # sampling=(gsd_y, gsd_x) per site: upland pixels are ~0.64 x 0.51 m, so a single
            # scalar 0.5 m overstates the band by up to 1.28x there. Returns metres directly.
            gsd_y, gsd_x = gsd_for(pathlib.Path(f).stem)
            dist_m = distance_transform_edt(b, sampling=(gsd_y, gsd_x))
            within[c].append((dist_m[b] <= BAND_M).astype(np.float32))

    exposure = {NAMES[c]: float(100.0 * np.concatenate(within[c]).mean()) for c in NAMES}
    x = np.array([exposure[NAMES[c]] for c in NAMES])
    y = np.array([iou[NAMES[c]] * 100.0 for c in NAMES])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r2 = float(1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    out = {
        "n_tiles": len(masks),
        "band_m": BAND_M,
        "gsd_by_site": {k: list(v) for k, v in GSD_BY_SITE.items()},
        "distance": "euclidean EDT to own-class boundary",
        "exposure_pct_within_band": {k: round(v, 1) for k, v in exposure.items()},
        "baseline_iou_pct": {NAMES[c]: round(iou[NAMES[c]] * 100, 1) for c in NAMES},
        "R2_iou_vs_exposure": round(r2, 3),
        "ols_slope": round(float(slope), 3),
    }
    (root / "analysis/label_ceiling/boundary_exposure.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
