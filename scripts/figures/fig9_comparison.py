#!/usr/bin/env python3
"""
scripts/figures/fig9_comparison.py

Fig 9: Qualitative ablation comparison on a single Biodiversity *validation* tile.

Layout (2 rows x 4 columns):
  Row 1: (a) RGB image, (b) GT mask, (c) Stage 1, (d) Stage 2
  Row 2: (e) Stage 3, (f) Stage 4, (g) Stage 5, (h) Stage 6

Writes:
  figures/fig9_comparison.pdf

Run:
  python -m scripts.figures.fig9_comparison \
    --img-id biodiversity_1382 \
    --data-root data/biodiversity_split/val \
    --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from geoseg.datasets.biodiversity_dataset import (
    BiodiversityValDataset,
    CLASSES,
    PALETTE as DATASET_PALETTE,
)
from geoseg.models.ftunetformer import ft_unetformer


# -----------------------------------------------------------------------------
# Plot style (match Fig 8)
# -----------------------------------------------------------------------------

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
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
PALETTE: Dict[int, Tuple[int, int, int]] = {
    i: tuple(rgb) for i, rgb in enumerate(DATASET_PALETTE)
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "geoseg").is_dir() and (cand / "scripts").is_dir():
            return cand
    return p


def resolve_ckpt(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        ckpts = sorted(p.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt found in {p}")
        non_last = [c for c in ckpts if c.name != "last.ckpt"]
        return str((non_last[0] if non_last else ckpts[0]).resolve())
    if p.is_file():
        return str(p.resolve())
    raise FileNotFoundError(path)


def load_net_from_lightning_ckpt(
    net: torch.nn.Module, ckpt_path: str
) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            cleaned[k[len("net."):]] = v

    net.load_state_dict(cleaned, strict=False)
    return net


@torch.no_grad()
def predict_mask(
    net: torch.nn.Module, img_t: torch.Tensor, device: torch.device
) -> np.ndarray:
    net.eval()
    logits = net(img_t.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)


def denormalize_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = img_t[:3].cpu().numpy().transpose(1, 2, 0)
    arr = (arr * std) + mean
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, rgb in PALETTE.items():
        out[mask == k] = rgb
    return out


def make_legend_handles() -> List[Patch]:
    return [
        Patch(
            facecolor=np.array(PALETTE[k]) / 255.0,
            edgecolor="none",
            label=CLASS_NAMES[k],
        )
        for k in range(1, len(CLASS_NAMES))  # exclude background
    ]


def make_grid_figure(
    panels: List[np.ndarray],
    titles: List[str],
    suptitle: str,
):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()

    for ax, im, ttl in zip(axes, panels, titles):
      ax.imshow(im)
      ax.set_aspect("auto")   # <<< THIS LINE
      ax.set_title(ttl, fontsize=10)
      ax.axis("off")

    fig.suptitle(suptitle, y=0.96)
    fig.legend(
        handles=make_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.90, bottom=0.12,
        wspace=0.05, hspace=0.2
    )
    return fig


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_single_tile(
    data_root: Path, img_id: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = BiodiversityValDataset(data_root=str(data_root))

    if img_id not in ds.img_ids:
        raise FileNotFoundError(f"{img_id} not found in {data_root}")

    idx = ds.img_ids.index(img_id)
    item = ds[idx]

    return item["img"], item["gt_semantic_seg"]


def build_ftunetformer() -> torch.nn.Module:
    return ft_unetformer(num_classes=6, pretrained=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-id", required=True)
    ap.add_argument("--data-root", default="data/biodiversity_split/val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-path", default="figures/fig9_comparison.pdf")

    ap.add_argument(
    "--stage1-ckpt",
    default="model_weights/biodiversity/stage1_baseline_ftunetformer",
    )
    ap.add_argument(
        "--stage2-ckpt",
        default="model_weights/biodiversity/stage2_replication_ftunetformer",
    )
    ap.add_argument(
        "--stage3-ckpt",
        default="model_weights/biodiversity/stage3_replication_difficulty_ftunetformer",
    )
    ap.add_argument(
        "--stage4-ckpt",
        default="model_weights/biodiversity/stage4_replication_difficulty_minoritycrop_ftunetformer",
    )
    ap.add_argument(
        "--stage5-ckpt",
        default="model_weights/biodiversity/stage5_finetune_after_oem_ftunetformer",
    )
    ap.add_argument(
        "--stage6-ckpt",
        default="model_weights/biodiversity/stage6_final_kd_ftunetformer",
    )


    args = ap.parse_args()

    set_plot_style()
    repo_root = find_repo_root()
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    img_t, mask_t = load_single_tile(
        repo_root / args.data_root, args.img_id
    )

    ckpts = [
        args.stage1_ckpt,
        args.stage2_ckpt,
        args.stage3_ckpt,
        args.stage4_ckpt,
        args.stage5_ckpt,
        args.stage6_ckpt,
    ]

    preds = []
    for ck in ckpts:
        net = build_ftunetformer().to(device)
        net = load_net_from_lightning_ckpt(net, resolve_ckpt(ck))
        preds.append(colorize_mask(predict_mask(net, img_t, device)))

    panels = [
        denormalize_to_uint8(img_t),
        colorize_mask(mask_t.numpy()),
        *preds,
    ]

    titles = [
        "(a) Satellite image",
        "(b) Ground truth",
        "(c) Stage 1: Baseline",
        "(d) Stage 2: +Replication",
        "(e) Stage 3: +Difficulty",
        "(f) Stage 4: +Minority crop",
        "(g) Stage 5: +OEM",
        "(h) Stage 6: +KD",
    ]

    fig = make_grid_figure(panels, titles, suptitle=args.img_id)

    out = repo_root / args.out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out}")


if __name__ == "__main__":
    main()
