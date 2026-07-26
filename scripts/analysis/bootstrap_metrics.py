#!/usr/bin/env python3
"""
Spatial-block bootstrap confidence intervals for the four factorial cells.

Runs inference on val, Test A and Test B, collects per-tile confusion matrices, then bootstraps
by resampling 950 m SPATIAL BLOCKS to produce 95% CIs for mIoU, mF1, OA. Reports Kish n_eff
beside the nominal block count, because the blocks are badly unequal in size.

Usage:
    python scripts/analysis/bootstrap_metrics.py [--device cuda|cpu] [--n-boot 2000]

Outputs:
    analysis/bootstrap_results.md   — markdown summary
    analysis/per_tile_cms/          — cached per-tile confusion matrices (.npz)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── repo root ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.utils import SPLIT_TAG, cell_dir, find_repo_root, spatial_blocks

REPO = find_repo_root()

from evaluation.compute_metrics import (
    build_model,
    load_checkpoint_into_model,
    _apply_ignore_mask,
)
from geoseg.datasets.biodiversity_dataset import (
    BiodiversityValDataset,
    BiodiversityTestWithMasksDataset,
)

# ── constants ───────────────────────────────────────────────────────────────
NUM_CLASSES = 6
IGNORE_INDEX = 0
CLASS_NAMES_5 = ["Forest", "Grassland", "Cropland", "Settlement", "Seminatural"]

# All four factorial cells, tagged. The untagged spellings this used to hard-code point at the
# WITHDRAWN campaign's checkpoints, which still exist under _archive/stale_checkpoints_pre_rebuild/;
# restoring one made this script print bootstrap intervals on the leaking split and exit 0.
CELLS = ["stage1_baseline", "stage2b_oem_finetune", "stage_sampler_only", "stage3_clsbal"]


def stage_ckpt(cell: str) -> Path:
    d = cell_dir(cell)
    return REPO / "model_weights" / "biodiversity" / d / f"{d}.ckpt"


# ── per-tile inference ──────────────────────────────────────────────────────

@torch.no_grad()
def collect_per_tile_cms(
    ckpt_path: Path,
    split: str,
    device: torch.device,
    data_root: Path,
) -> list[np.ndarray]:
    """Return a list of (6,6) int64 confusion matrices, one per tile, in sorted tile-id order."""
    model = build_model(num_classes=NUM_CLASSES).to(device)
    model = load_checkpoint_into_model(model, ckpt_path, device)
    model.eval()

    if split == "val":
        ds = BiodiversityValDataset(data_root=str(data_root))
    else:
        ds = BiodiversityTestWithMasksDataset(data_root=str(data_root))

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    softmax = nn.Softmax(dim=1)
    tile_cms = []

    for batch in tqdm(loader, desc=f"{ckpt_path.stem}/{split}", leave=False):
        img = batch["img"].to(device)
        mask = batch["gt_semantic_seg"].cpu().numpy()[0]  # (H,W)
        pred = softmax(model(img)).argmax(dim=1).cpu().numpy()[0]  # (H,W)

        t, p = _apply_ignore_mask(mask, pred, IGNORE_INDEX)
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for gt_val, pr_val in zip(t, p):
            cm[gt_val, pr_val] += 1
        tile_cms.append(cm)

    return tile_cms


# ── metrics from confusion matrix ──────────────────────────────────────────

def metrics_from_cm(cm: np.ndarray) -> dict:
    """Compute mIoU, mF1, OA from a (6,6) confusion matrix (bg-aware)."""
    eps = 1e-8
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    # A class absent from BOTH ground truth and prediction is NaN, not 0 -- otherwise nanmean
    # averages in a hard zero at full weight. This bites here more than anywhere: a block resample
    # that happens to draw no Cropland is common (Test A cropland sits in 8 blocks of 16, Test B in
    # 4 of 12), and scoring it 0 drags the resampled distribution down and widens it.
    denom = tp + fp + fn
    present = denom > 0
    iou = np.where(present, tp / (denom + eps), np.nan)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = np.where(present, 2 * precision * recall / (precision + recall + eps), np.nan)

    oa = float(tp.sum() / (cm.sum() + eps))
    miou = float(np.nanmean(iou[1:]))  # foreground only
    mf1 = float(np.nanmean(f1[1:]))
    per_class_iou = {CLASS_NAMES_5[i]: float(iou[i + 1]) for i in range(5)}

    return {"mIoU": miou, "mF1": mf1, "OA": oa, "per_class_iou": per_class_iou}


# ── bootstrap ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    tile_cms: list[np.ndarray],
    block_of: list,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap over SPATIAL BLOCKS, aggregate CMs, compute metrics.

    `block_of[i]` is the block id of `tile_cms[i]`, from utils.spatial_blocks, and it is REQUIRED.
    The tile is not an independent unit: on a 50% stride the 294-tile test split is 104
    pixel-disjoint footprints and 16 independent units at the 950 m correlation scale, so
    resampling tile indices returns intervals 1.6-26x too narrow (round-2 audit, B6). This used to
    default to None and merely log a warning, and main() never passed it -- so the guarantee sat on
    a code path the shipped command never took, and every interval it printed was a tile-id
    interval. Making it a required argument is what removes that possibility rather than
    documenting it.
    """
    if len(block_of) != len(tile_cms):
        raise ValueError(f"block_of has {len(block_of)} entries for {len(tile_cms)} tiles")
    rng = np.random.default_rng(seed)
    n = len(tile_cms)
    cms = np.stack(tile_cms)  # (N, 6, 6)
    from collections import defaultdict
    by_block = defaultdict(list)
    for i, b in enumerate(block_of):
        by_block[b].append(i)
    groups = [np.array(by_block[k]) for k in sorted(by_block, key=str)]

    boot_miou, boot_mf1, boot_oa = [], [], []
    boot_per_class = {c: [] for c in CLASS_NAMES_5}

    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        agg_cm = cms[idx].sum(axis=0)
        m = metrics_from_cm(agg_cm)
        boot_miou.append(m["mIoU"])
        boot_mf1.append(m["mF1"])
        boot_oa.append(m["OA"])
        for c in CLASS_NAMES_5:
            boot_per_class[c].append(m["per_class_iou"][c])

    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

    # Point estimate from full aggregate
    full_cm = cms.sum(axis=0)
    point = metrics_from_cm(full_cm)

    # Kish effective n over the realised tiles per block. 16 blocks holding 43..2 tiles are not 16
    # exchangeable draws, and the nominal count overstates the support if reported alone.
    w = np.array([len(g) for g in groups], dtype=float)
    result = {
        "n_tiles": n,
        "n_blocks": len(groups),
        "n_eff_kish": float(w.sum() ** 2 / (w ** 2).sum()),
        "tiles_per_block": sorted((int(x) for x in w), reverse=True),
        "n_boot": n_boot,
        "point": point,
        "ci_95": {
            "mIoU": [float(np.percentile(boot_miou, lo)), float(np.percentile(boot_miou, hi))],
            "mF1": [float(np.percentile(boot_mf1, lo)), float(np.percentile(boot_mf1, hi))],
            "OA": [float(np.percentile(boot_oa, lo)), float(np.percentile(boot_oa, hi))],
            "per_class_iou": {
                c: [float(np.percentile(boot_per_class[c], lo)),
                    float(np.percentile(boot_per_class[c], hi))]
                for c in CLASS_NAMES_5
            },
        },
    }
    return result


# ── caching ────────────────────────────────────────────────────────────────

def cache_path(stage: str, split: str) -> Path:
    return REPO / "analysis" / "per_tile_cms" / f"{stage}_{split}.npz"


def save_cms(tile_cms: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, *tile_cms)


def load_cms(path: Path) -> list[np.ndarray]:
    data = np.load(path)
    return [data[k] for k in sorted(data.files, key=lambda x: int(x.split("_")[1]))]


# ── main ───────────────────────────────────────────────────────────────────

def self_test() -> int:
    """Three properties, each checked against an input that would break it.

    Runs without a trained model: the block geometry comes from the real split, the confusion
    matrices are synthetic.
    """
    ok = True

    print("absent class must be NaN, so nanmean drops it rather than averaging in a zero")
    cm = np.zeros((6, 6), np.int64)
    cm[1, 1] = cm[2, 2] = 1000
    cm[4, 4] = cm[5, 5] = 500                       # Cropland absent from GT and prediction
    m = metrics_from_cm(cm)
    good = np.isnan(m["per_class_iou"]["Cropland"]) and abs(m["mIoU"] - 1.0) < 1e-6
    print(f"  perfect prediction, Cropland absent: IoU={m['per_class_iou']['Cropland']}, "
          f"mIoU={m['mIoU']:.6f}  [{'ok' if good else 'FAIL: 0.8 means the zero was averaged in'}]")
    ok &= good

    print("\nblock_of must be impossible to omit or mis-length")
    cms = [np.eye(6, dtype=np.int64) * 100 for _ in range(10)]
    try:
        bootstrap_ci(cms, n_boot=5)                 # type: ignore[call-arg]
        print("  omitted block_of was accepted  [FAIL]")
        ok = False
    except TypeError:
        print("  omitted block_of raises TypeError  [ok]")
    try:
        bootstrap_ci(cms, ["a"] * 9, n_boot=5)
        print("  mis-length block_of was accepted  [FAIL]")
        ok = False
    except ValueError:
        print("  mis-length block_of raises ValueError  [ok]")

    print("\nthe block unit must widen the interval against tile ids, on the real test geometry")
    sr = REPO / "data" / f"split_{SPLIT_TAG}"
    if not (sr / "test" / "images").is_dir():
        print(f"  SKIP: {sr}/test not materialised")
    else:
        blocks = spatial_blocks(sr, "test", 950.0)
        tids = sorted(q.stem for q in (sr / "test" / "images").glob("*.tif"))
        rng = np.random.default_rng(0)
        # One correlated error rate per block, shared by every tile in it -- the dependence the
        # block unit exists to respect. Resampling tile ids treats these as independent draws.
        per_block = {b: rng.normal(0, 0.25) for b in set(blocks.values())}
        cms = []
        for t in tids:
            cm = np.zeros((6, 6), np.int64)
            err = float(np.clip(0.10 * np.exp(per_block[blocks[t]]), 0, 0.9))
            n_px, n_err = 200_000, int(200_000 * err)
            for c in range(1, 6):
                cm[c, c] += (n_px - n_err) // 5
                cm[c, (c % 5) + 1] += n_err // 5
            cms.append(cm)
        rb = bootstrap_ci(cms, [blocks[t] for t in tids], n_boot=800)
        rt = bootstrap_ci(cms, [(t,) for t in tids], n_boot=800)
        wb = rb["ci_95"]["mIoU"][1] - rb["ci_95"]["mIoU"][0]
        wt = rt["ci_95"]["mIoU"][1] - rt["ci_95"]["mIoU"][0]
        good = wb > 1.4 * wt
        print(f"  by 950 m block ({rb['n_blocks']} units, Kish n_eff {rb['n_eff_kish']:.2f}): "
              f"CI width {wb:.5f}")
        print(f"  by tile id     ({rt['n_blocks']} units, Kish n_eff {rt['n_eff_kish']:.2f}): "
              f"CI width {wt:.5f}")
        print(f"  block CI is {wb / wt:.2f}x wider  "
              f"[{'ok, the unit of analysis matters' if good else 'FAIL: no difference'}]")
        ok &= good

    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="Block-bootstrap CIs for the four factorial cells")
    parser.add_argument("--self-test", action="store_true",
                        help="check the three properties above without a trained model")
    parser.add_argument("--split-root", default=f"data/split_{SPLIT_TAG}",
                        help="spatially blocked split root. The old hard-coded "
                             "data/biodiversity_split is the WITHDRAWN random split and is still "
                             "on disk (219 val / 218 test tiles), so it cannot be a default.")
    parser.add_argument("--splits", nargs="+", default=["val", "test", "external_test"])
    parser.add_argument("--block-m", type=float, default=950.0,
                        help="bootstrap block size in metres; see utils.spatial_blocks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="Re-run inference even if cache exists")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    split_root = REPO / args.split_root
    if not split_root.is_dir():
        raise SystemExit(f"split root not found: {split_root}")
    device = torch.device(args.device)
    all_results = {}

    for cell in CELLS:
        ckpt = stage_ckpt(cell)
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint not found: {ckpt.relative_to(REPO)} "
                             f"(SPLIT_TAG={SPLIT_TAG}); run the campaign first")
        for split in args.splits:
            key = f"{cell}/{split}"
            cp = cache_path(f"{cell_dir(cell)}_{args.split_root.replace('/', '_')}", split)
            data_root = split_root / split

            if cp.exists() and not args.force:
                print(f"[cache] Loading {cp}")
                tile_cms = load_cms(cp)
            else:
                print(f"[infer] {key} on {device}")
                tile_cms = collect_per_tile_cms(ckpt, split, device, data_root)
                save_cms(tile_cms, cp)
                print(f"[cache] Saved {len(tile_cms)} tile CMs -> {cp}")

            # Tiles come back in sorted img_id order (both datasets sort their listing), so the
            # block ids line up positionally. The length check inside bootstrap_ci is what catches
            # it if that ever stops being true.
            blocks = spatial_blocks(split_root, split, args.block_m)
            tile_ids = sorted(q.stem for q in (data_root / "images").glob("*.tif"))
            block_of = [blocks[tid] for tid in tile_ids]

            result = bootstrap_ci(tile_cms, block_of, n_boot=args.n_boot)
            all_results[key] = result
            p = result["point"]
            ci = result["ci_95"]
            print(f"  {key}: mIoU={p['mIoU']:.1%} [{ci['mIoU'][0]:.1%}, {ci['mIoU'][1]:.1%}]  "
                  f"({result['n_blocks']} blocks at {args.block_m:.0f} m, "
                  f"Kish n_eff {result['n_eff_kish']:.2f}, {result['n_tiles']} tiles)")

    # ── write markdown report ──────────────────────────────────────────────
    out_dir = REPO / "analysis"
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / "bootstrap_results.md"

    lines = [
        "# Bootstrap Confidence Intervals (950 m Spatial-Block Resampling)",
        "",
        f"Split root: {args.split_root} | Block: {args.block_m:.0f} m | "
        f"Resamples: {args.n_boot} | Seed: 42 | Level: 95%",
        "",
    ]

    for key, r in all_results.items():
        p = r["point"]
        ci = r["ci_95"]
        lines.append(f"## {key}")
        lines.append(f"- Tiles: {r['n_tiles']} in {r['n_blocks']} blocks "
                     f"(Kish n_eff {r['n_eff_kish']:.2f}); tiles per block {r['tiles_per_block']}")
        lines.append(f"- **mIoU**: {p['mIoU']:.1%}  [{ci['mIoU'][0]:.1%}, {ci['mIoU'][1]:.1%}]")
        lines.append(f"- **mF1**:  {p['mF1']:.1%}  [{ci['mF1'][0]:.1%}, {ci['mF1'][1]:.1%}]")
        lines.append(f"- **OA**:   {p['OA']:.1%}  [{ci['OA'][0]:.1%}, {ci['OA'][1]:.1%}]")
        lines.append("")
        lines.append("| Class | IoU | 95% CI |")
        lines.append("|-------|-----|--------|")
        for c in CLASS_NAMES_5:
            iou_val = p["per_class_iou"][c]
            lo, hi = ci["per_class_iou"][c]
            lines.append(f"| {c} | {iou_val:.1%} | [{lo:.1%}, {hi:.1%}] |")
        lines.append("")

    # ── Stage 1→3 delta CI ──────────────────────────────────────────────────
    for split in args.splits:
        s1 = all_results.get(f"stage1_baseline/{split}")
        s3 = all_results.get(f"stage3_clsbal/{split}")
        if s1 and s3:
            lines.append(f"## Stage 1→3 improvement ({split})")
            for metric in ("mIoU", "mF1", "OA"):
                delta = s3["point"][metric] - s1["point"][metric]
                # Delta of values rounded to 1 dp -- the convention used in the manuscript
                # tables, where per-class/mean deltas are differences of the displayed rounded
                # cells. This can differ from the raw delta by up to ~0.1 pp due to endpoint
                # rounding (e.g. test mIoU: raw +10.5 vs rounded-endpoint +10.6).
                delta_rounded = round(s3["point"][metric] * 100, 1) - round(s1["point"][metric] * 100, 1)
                # Width of individual CIs as proxy for uncertainty
                w1 = s1["ci_95"][metric][1] - s1["ci_95"][metric][0]
                w3 = s3["ci_95"][metric][1] - s3["ci_95"][metric][0]
                lines.append(f"- **Δ{metric}**: +{delta:.1%} raw (+{delta_rounded:.1f} pp from rounded endpoints, as reported in the manuscript)  (individual CI widths: ±{w1/2:.1%}, ±{w3/2:.1%})")
            lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"\nResults written to {md_path}")

    # Also save raw JSON
    json_path = out_dir / "bootstrap_results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"Raw JSON written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
