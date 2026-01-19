#!/usr/bin/env python3
"""
scripts/figures/graphical_abstract.py

Save GA-ready prediction masks (as separate PNGs) for one chosen tile:
- Stage 1 (baseline)
- Stage 4
- Stage 6

Outputs:
  figures/graphical_abstract/<img_id>_stage1_baseline.png
  figures/graphical_abstract/<img_id>_stage4.png
  figures/graphical_abstract/<img_id>_stage6.png

Optional (helpful for GA assembly):
  figures/graphical_abstract/<img_id>_rgb.png
  figures/graphical_abstract/<img_id>_gt.png

Run:
  python -m scripts.figures.graphical_abstract --img-id biodiversity_1310 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from geoseg.datasets.biodiversity_dataset import (
    BiodiversityValDataset,
    CLASSES,
    PALETTE as DATASET_PALETTE,
)
from geoseg.models.ftunetformer import ft_unetformer


# -----------------------------------------------------------------------------
# Canonical classes & palette (from dataset)
# -----------------------------------------------------------------------------

PALETTE: Dict[int, Tuple[int, int, int]] = {i: tuple(rgb) for i, rgb in enumerate(DATASET_PALETTE)}

# Albumentations Normalize() defaults (used by your dataset)
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


def build_ftunetformer() -> torch.nn.Module:
    return ft_unetformer(num_classes=len(CLASSES), pretrained=False)


@torch.no_grad()
def predict_mask(net: torch.nn.Module, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    net.eval()
    logits = net(img_t.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)


def denormalize_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: (C,H,W) float tensor AFTER albu.Normalize(mean/std).
    returns: (H,W,3) uint8 RGB for plotting/saving.
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


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-id", default="biodiversity_1310", help="Tile stem (no extension).")
    ap.add_argument("--data-root", default="data/biodiversity_split", help="Root containing val/ etc.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--nodata-thresh", type=int, default=5)

    # checkpoints (dirs or .ckpt)
    ap.add_argument("--stage1-ckpt", default="model_weights/biodiversity/stage1_baseline_ftunetformer")
    ap.add_argument("--stage4-ckpt", default="model_weights/biodiversity/stage4_replication_difficulty_minoritycrop_ftunetformer")
    ap.add_argument("--stage6-ckpt", default="model_weights/biodiversity/stage6_final_kd_ftunetformer")

    ap.add_argument("--out-dir", default="figures/graphical_abstract", help="Output directory.")
    ap.add_argument("--also-save-rgb-gt", action="store_true", help="Also save RGB + GT PNGs for GA assembly.")
    args = ap.parse_args()

    repo_root = find_repo_root()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # load val tile via dataset (keeps identical preprocessing)
    val_root = (repo_root / args.data_root / "val").resolve()
    ds = BiodiversityValDataset(data_root=str(val_root))
    if args.img_id not in ds.img_ids:
        raise FileNotFoundError(f"{args.img_id} not found in val split under {val_root}")

    idx = ds.img_ids.index(args.img_id)
    item = ds[idx]
    img_t: torch.Tensor = item["img"]
    gt_t: torch.Tensor = item["gt_semantic_seg"]

    img_rgb = denormalize_to_uint8(img_t)
    valid = make_valid_mask_from_rgb(img_rgb, thresh=args.nodata_thresh)

    gt = gt_t.detach().cpu().numpy().astype(np.uint8)
    #gt[~valid] = 0

    # resolve ckpts
    ckpt_s1 = resolve_ckpt(str((repo_root / args.stage1_ckpt).resolve()))
    ckpt_s4 = resolve_ckpt(str((repo_root / args.stage4_ckpt).resolve()))
    ckpt_s6 = resolve_ckpt(str((repo_root / args.stage6_ckpt).resolve()))

    # load models
    net_s1 = load_net_from_lightning_ckpt(build_ftunetformer(), ckpt_s1).to(device)
    net_s4 = load_net_from_lightning_ckpt(build_ftunetformer(), ckpt_s4).to(device)
    net_s6 = load_net_from_lightning_ckpt(build_ftunetformer(), ckpt_s6).to(device)

    # predict
    p1 = predict_mask(net_s1, img_t, device)#; p1[~valid] = 0
    p4 = predict_mask(net_s4, img_t, device)#; p4[~valid] = 0
    p6 = predict_mask(net_s6, img_t, device)#; p6[~valid] = 0

    # colorize
    p1_rgb = colorize_mask(p1)
    p4_rgb = colorize_mask(p4)
    p6_rgb = colorize_mask(p6)

    out_dir = (repo_root / args.out_dir).resolve()
    save_png(out_dir / f"{args.img_id}_stage1_baseline.png", p1_rgb)
    save_png(out_dir / f"{args.img_id}_stage4.png", p4_rgb)
    save_png(out_dir / f"{args.img_id}_stage6.png", p6_rgb)

    if args.also_save_rgb_gt:
        save_png(out_dir / f"{args.img_id}_rgb.png", img_rgb)
        save_png(out_dir / f"{args.img_id}_gt.png", colorize_mask(gt))

    print("Saved to:", out_dir)
    print("Stage1 ckpt:", ckpt_s1)
    print("Stage4 ckpt:", ckpt_s4)
    print("Stage6 ckpt:", ckpt_s6)
    print("Device:", device)


if __name__ == "__main__":
    main()
