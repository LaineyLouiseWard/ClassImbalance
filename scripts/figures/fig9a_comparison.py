#!/usr/bin/env python3
"""
scripts/figures/fig9a_comparison.py

Fig 9a: Qualitative ablation comparison on FOUR tiles, shown as 4 columns.

Layout (8 rows x 4 columns):
  Rows:    Image, GT, Stage 1, Stage 2, Stage 3, Stage 4, Stage 5, Stage 6
  Columns: (a)–(d) = four chosen tile IDs

Key visual rule (copied from fig6a_comparison.py):
- For visualisation only, force GT + predictions to background wherever the RGB image is near-black (outside AOI).

Writes:
  figures/fig9a_comparison.pdf

Run:
    python -m scripts.figures.fig9a_comparison   --img-ids biodiversity_1382,biodiversity_0259,ireland2_0090,den1_0020   --data-root data/biodiversity_split   --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

from geoseg.datasets.biodiversity_dataset import (
    BiodiversityValDataset,
    BiodiversityTestWithMasksDataset,
    CLASSES,
    PALETTE as DATASET_PALETTE,
)
from geoseg.models.ftunetformer import ft_unetformer


# -----------------------------------------------------------------------------
# Plot style (match Fig 8 vibe)
# -----------------------------------------------------------------------------

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })


# -----------------------------------------------------------------------------
# Classes & palette (canonical from dataset)
# -----------------------------------------------------------------------------

CLASS_NAMES: Dict[int, str] = {i: n for i, n in enumerate(CLASSES)}
PALETTE: Dict[int, Tuple[int, int, int]] = {i: tuple(rgb) for i, rgb in enumerate(DATASET_PALETTE)}


# Albumentations Normalize() defaults
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(12):
        if (p / "scripts").is_dir() and (p / "geoseg").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root. Run from inside the repo.")


def resolve_ckpt(path_like: str) -> Path:
    """
    Accept either:
      - a direct .ckpt path
      - a directory -> pick most recent .ckpt in it (recursive)
    """
    p = Path(path_like).expanduser().resolve()
    if p.is_file():
        if p.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got: {p}")
        return p

    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")

    ckpts = list(p.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under: {p}")

    ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return ckpts[0]


def load_net_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    """
    Load Lightning .ckpt into raw nn.Module.
    Handles 'net.' or 'model.' prefixes.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in sd.items():
        if k.startswith("net."):
            cleaned[k.replace("net.", "", 1)] = v
        elif k.startswith("model."):
            cleaned[k.replace("model.", "", 1)] = v
        else:
            cleaned[k] = v

    net.load_state_dict(cleaned, strict=False)
    return net


@torch.no_grad()
def predict_mask(net: torch.nn.Module, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    net.eval()
    logits = net(img_t.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)


def denormalize_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: (C,H,W) float tensor AFTER albu.Normalize(mean/std).
    returns: (H,W,3) uint8 RGB for plotting.
    """
    img = img_t.detach().cpu().numpy().astype(np.float32)
    img = img[:3] if img.shape[0] >= 3 else np.repeat(img, 3, axis=0)
    img = (img.transpose(1, 2, 0) * IMAGENET_STD) + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def make_valid_mask_from_rgb(img_rgb: np.ndarray, thresh: int = 5) -> np.ndarray:
    """
    Treat near-black as nodata/outside AOI.
    Returns (H,W) bool where True means valid imagery.
    """
    return (img_rgb[..., 0] > thresh) | (img_rgb[..., 1] > thresh) | (img_rgb[..., 2] > thresh)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, rgb in PALETTE.items():
        out[mask == k] = rgb
    return out


def make_legend_handles(include_background: bool = True) -> List[Patch]:
    keys = list(range(len(CLASS_NAMES))) if include_background else list(range(1, len(CLASS_NAMES)))
    return [
        Patch(
            facecolor=np.array(PALETTE[k]) / 255.0,
            edgecolor="none",
            label=CLASS_NAMES[k],
        )
        for k in keys
    ]


# -----------------------------------------------------------------------------
# Data loading (auto-find in val or test)
# -----------------------------------------------------------------------------

def _try_load_from_split(split_root: Path, img_id: str):
    """
    Returns (img_t, mask_t) or None if not found.
    """
    if not split_root.exists():
        return None

    # val: standard dataset
    if split_root.name == "val":
        ds = BiodiversityValDataset(data_root=str(split_root))
        if img_id not in ds.img_ids:
            return None
        idx = ds.img_ids.index(img_id)
        item = ds[idx]
        return item["img"], item["gt_semantic_seg"]

    # test: use the WITH masks dataset for figures
    if split_root.name == "test":
        ds = BiodiversityTestWithMasksDataset(data_root=str(split_root))
        if img_id not in ds.img_ids:
            return None
        idx = ds.img_ids.index(img_id)
        item = ds[idx]
        return item["img"], item["gt_semantic_seg"]

    return None


def load_tile_any_split(data_root: Path, img_id: str, split_order: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    data_root is the biodiversity_split root, containing val/ and test/.
    """
    for sp in split_order:
        got = _try_load_from_split(data_root / sp, img_id)
        if got is not None:
            return got
    raise FileNotFoundError(f"{img_id} not found under {data_root} in splits {split_order}")


def build_ftunetformer() -> torch.nn.Module:
    return ft_unetformer(num_classes=6, pretrained=False)


# -----------------------------------------------------------------------------
# Figure builder
# -----------------------------------------------------------------------------

def make_grid_figure(
    columns: List[Dict[str, np.ndarray]],
    col_titles: List[str],
    row_labels: List[str],
    legend_handles: List[Patch],
) -> plt.Figure:
    """
    columns: list of dicts with keys:
      "img", "gt", "s1", "s2", "s3", "s4", "s5", "s6"
      each value is an RGB uint8 image (H,W,3)
    """
    n_cols = len(columns)
    n_rows = len(row_labels)

    # Add a skinny label column + N image columns
    fig = plt.figure(figsize=(3.1 * n_cols + 1.6, 2.55 * n_rows))
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_cols + 1,
        figure=fig,
        width_ratios=[0.55] + [1.0] * n_cols,  # label col, then images
        wspace=0.02,
        hspace=0.03,  # tighten vertical spacing
    )

    # Left-side row labels (INLINE + vertically centered per row)
    for r, label in enumerate(row_labels):
        ax_lab = fig.add_subplot(gs[r, 0])
        ax_lab.axis("off")
        ax_lab.text(
            0.98, 0.5, label,
            ha="right", va="center",
            fontsize=20, fontweight="bold",  # bigger
        )

    # Image panels
    for c in range(n_cols):
        col = columns[c]
        keys = ["img", "gt", "s1", "s2", "s3", "s4", "s5", "s6"]
        for r, k in enumerate(keys):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(col[k])
            ax.set_aspect("auto")
            ax.axis("off")

            # Column header only on the top row
            if r == 0:
                ax.set_title(col_titles[c], fontsize=18, pad=10, fontweight="bold")

    # Legend (bigger boxes/text, INCLUDE background)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.6, -0.06),
        handlelength=2.2,
        fontsize=20,
        handleheight=2.0,
        columnspacing=2.0,
        labelspacing=1.1,
        borderaxespad=0.0,
    )

    # Tighten around content but leave room for legend
    fig.subplots_adjust(left=0.03, right=0.995, top=0.985, bottom=0.02)

    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--img-ids",
        default="biodiversity_1382,biodiversity_0259,ireland2_0090,den1_0020",
        help="Comma-separated tile stems (no extension). Exactly 4 recommended.",
    )
    ap.add_argument(
        "--data-root",
        default="data/biodiversity_split",
        help="Root containing val/ and test/ folders.",
    )
    ap.add_argument(
        "--split-order",
        default="val,test",
        help="Where to search for the IDs, in order. e.g. 'val,test' or 'test,val'",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-path", default="figures/fig9a_comparison.pdf")

    ap.add_argument("--stage1-ckpt", default="model_weights/biodiversity/stage1_baseline_ftunetformer")
    ap.add_argument("--stage2-ckpt", default="model_weights/biodiversity/stage2_replication_ftunetformer")
    ap.add_argument("--stage3-ckpt", default="model_weights/biodiversity/stage3_replication_difficulty_ftunetformer")
    ap.add_argument("--stage4-ckpt", default="model_weights/biodiversity/stage4_replication_difficulty_minoritycrop_ftunetformer")
    ap.add_argument("--stage5-ckpt", default="model_weights/biodiversity/stage5_finetune_after_oem_ftunetformer")
    ap.add_argument("--stage6-ckpt", default="model_weights/biodiversity/stage6_final_kd_ftunetformer")

    # Visual nodata masking threshold (same idea as fig6a)
    ap.add_argument("--nodata-thresh", type=int, default=5, help="RGB threshold for nodata masking.")

    args = ap.parse_args()

    set_plot_style()
    repo_root = find_repo_root()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    img_ids = [s.strip() for s in args.img_ids.split(",") if s.strip()]
    split_order = [s.strip() for s in args.split_order.split(",") if s.strip()]

    data_root = (repo_root / args.data_root).resolve()

    # Resolve checkpoints (pick most recent if directory)
    ckpt_paths = {
        "s1": resolve_ckpt(str((repo_root / args.stage1_ckpt).resolve())),
        "s2": resolve_ckpt(str((repo_root / args.stage2_ckpt).resolve())),
        "s3": resolve_ckpt(str((repo_root / args.stage3_ckpt).resolve())),
        "s4": resolve_ckpt(str((repo_root / args.stage4_ckpt).resolve())),
        "s5": resolve_ckpt(str((repo_root / args.stage5_ckpt).resolve())),
        "s6": resolve_ckpt(str((repo_root / args.stage6_ckpt).resolve())),
    }

    # Load models ONCE (reuse for all columns)
    nets: Dict[str, torch.nn.Module] = {}
    for k, ck in ckpt_paths.items():
        net = build_ftunetformer()
        net = load_net_from_lightning_ckpt(net, ck).to(device)
        nets[k] = net

    # Build each column (one per image id)
    columns: List[Dict[str, np.ndarray]] = []
    for img_id in img_ids:
        img_t, mask_t = load_tile_any_split(data_root, img_id, split_order)

        img_rgb = denormalize_to_uint8(img_t)
        valid = make_valid_mask_from_rgb(img_rgb, thresh=args.nodata_thresh)

        gt = mask_t.detach().cpu().numpy().astype(np.uint8)
        gt_vis = gt.copy()
        gt_vis[~valid] = 0

        pred_vis: Dict[str, np.ndarray] = {}
        for stage_key in ["s1", "s2", "s3", "s4", "s5", "s6"]:
            p = predict_mask(nets[stage_key], img_t, device)
            p[~valid] = 0
            pred_vis[stage_key] = p

        columns.append({
            "img": img_rgb,
            "gt": colorize_mask(gt_vis),
            "s1": colorize_mask(pred_vis["s1"]),
            "s2": colorize_mask(pred_vis["s2"]),
            "s3": colorize_mask(pred_vis["s3"]),
            "s4": colorize_mask(pred_vis["s4"]),
            "s5": colorize_mask(pred_vis["s5"]),
            "s6": colorize_mask(pred_vis["s6"]),
        })

    # Column titles (a)-(d) only
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    col_titles = [letters[i] if i < len(letters) else f"({i})" for i in range(len(columns))]

    # Left-side row labels
    row_labels = [
        "Satellite\nImage",
        "Ground\nTruth",
        "Stage 1:\nBaseline",
        "Stage 2:\n+Replication",
        "Stage 3:\n+Difficulty-Weighted\nSampling",
        "Stage 4:\n+Minority-Aware\nCropping",
        "Stage 5:\n+OEM Integration",
        "Stage 6:\n+Knowledge\nDistillation",
    ]


    # Legend (INCLUDE background)
    handles = make_legend_handles(include_background=True)

    fig = make_grid_figure(
        columns=columns,
        col_titles=col_titles,
        row_labels=row_labels,
        legend_handles=handles,
    )

    out = (repo_root / args.out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out)
    print("Data root:", data_root)
    print("IDs:", img_ids)
    print("Splits searched:", split_order)
    print("Device:", device)
    for k, p in ckpt_paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
