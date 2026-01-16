#!/usr/bin/env python3
"""scripts/figures/fig9_comparison.py

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

from geoseg.datasets.biodiversity_dataset import BiodiversityValDataset
from geoseg.models.ftunetformer import ft_unetformer


# --- class names + palette (keep consistent with other figure scripts) ---
CLASS_NAMES: Dict[int, str] = {
    0: "Background",
    1: "Forest",
    2: "Grassland",
    3: "Cropland",
    4: "Settlement",
    5: "Semi-natural grassland",
}

# RGB colors in 0..255
PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),          # background - black
    1: (0, 100, 0),        # forest - dark green
    2: (124, 252, 0),      # grassland - bright green
    3: (255, 215, 0),      # cropland - gold
    4: (255, 0, 0),        # settlement - red
    5: (148, 0, 211),      # semi-natural - purple
}


def find_repo_root() -> Path:
    """Walk parents until we find geoseg/ and scripts/."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "geoseg").is_dir() and (cand / "scripts").is_dir():
            return cand
    return p


def resolve_ckpt(path: str) -> str:
    """Allow passing either a .ckpt file or a directory containing .ckpt files."""
    p = Path(path)
    if p.is_dir():
        ckpts = sorted(p.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in: {p}")
        # Prefer a non-last.ckpt if present
        non_last = [c for c in ckpts if c.name != "last.ckpt"]
        return str((non_last[0] if non_last else ckpts[0]).resolve())
    if p.is_file() and p.suffix == ".ckpt":
        return str(p.resolve())
    raise FileNotFoundError(f"Checkpoint not found: {p}")


def load_net_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    """Load PyTorch Lightning checkpoint weights into a plain torch model.

    In this repo, state_dict keys commonly look like 'model.xxx' (LightningModule wraps the net).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading {ckpt_path} (showing first 10): {unexpected[:10]}"
        )
    # missing keys can be okay (e.g., if checkpoint includes extra heads not present here)
    return net


@torch.no_grad()
def predict_mask(net: torch.nn.Module, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    net.eval()
    x = img_t.unsqueeze(0).to(device, non_blocking=True)
    logits = net(x)
    pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def denormalize_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """Undo albumentations Normalize() for display (ImageNet mean/std)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = img_t.detach().cpu().numpy()  # C,H,W
    rgb = arr[:3].transpose(1, 2, 0).astype(np.float32)  # H,W,3
    rgb = (rgb * std) + mean
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert (H,W) uint8 labels -> (H,W,3) RGB."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, rgb in PALETTE.items():
        out[mask == k] = rgb
    return out


def make_legend_handles(include_background: bool = True) -> List[Patch]:
    keys = sorted(PALETTE.keys())
    if not include_background:
        keys = [k for k in keys if k != 0]
    return [
        Patch(facecolor=np.array(PALETTE[k]) / 255.0, edgecolor="none", label=CLASS_NAMES[k])
        for k in keys
    ]


def make_grid_figure(panels: List[np.ndarray], titles: List[str], *, suptitle: str = ""):
    assert len(panels) == 8 and len(titles) == 8

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()

    for ax, im, ttl in zip(axes, panels, titles):
        ax.imshow(im)
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.98)

    handles = make_legend_handles(include_background=True)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.subplots_adjust(bottom=0.14, top=0.92, wspace=0.02, hspace=0.10)
    return fig


def build_ftunetformer() -> torch.nn.Module:
    """Create the student model architecture (checkpoint will supply weights)."""
    return ft_unetformer(pretrained=False, num_classes=6)


def load_single_tile(data_root: Path, img_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (image_tensor, mask_tensor) for a given tile id.

    NOTE: BiodiversityValDataset returns a dict: {"image":..., "mask":..., "img_id":...}
    and expects data_root that contains images/ and masks/.
    """
    ds = BiodiversityValDataset(data_root=str(data_root))
    if img_id not in ds.img_ids:
        raise FileNotFoundError(f"Tile id not found in {data_root}: {img_id}")
    idx = ds.img_ids.index(img_id)
    item = ds[idx]
    return item["image"], item["mask"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-id", default="biodiversity_1382", help="Tile stem without extension.")
    ap.add_argument(
        "--data-root",
        default="data/biodiversity_split/val",
        help="Directory containing images/ and masks/ for the chosen split.",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-path", default="figures/fig9_comparison.pdf")

    # stage checkpoints: can be directories OR specific .ckpt files
    ap.add_argument("--stage1-ckpt", default="model_weights/biodiversity/stage1_baseline_ftunetformer")
    ap.add_argument("--stage2-ckpt", default="model_weights/biodiversity/stage2_replication_ftunetformer")
    ap.add_argument("--stage3-ckpt", default="model_weights/biodiversity/stage3_replication_difficulty_ftunetformer")
    ap.add_argument("--stage4-ckpt", default="model_weights/biodiversity/stage4_replication_difficulty_minoritycrop_ftunetformer")
    ap.add_argument("--stage5-ckpt", default="model_weights/biodiversity/stage5_finetune_after_oem_ftunetformer")
    ap.add_argument("--stage6-ckpt", default="model_weights/biodiversity/stage6_final_kd_ftunetformer")

    args = ap.parse_args()

    repo_root = find_repo_root()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    data_root = (repo_root / args.data_root).resolve()
    img_t, mask_t = load_single_tile(data_root, args.img_id)

    ckpt_paths = {
        "Stage 1: Baseline": resolve_ckpt(str((repo_root / args.stage1_ckpt).resolve())),
        "Stage 2: +Replication": resolve_ckpt(str((repo_root / args.stage2_ckpt).resolve())),
        "Stage 3: +Difficulty": resolve_ckpt(str((repo_root / args.stage3_ckpt).resolve())),
        "Stage 4: +Minority crop": resolve_ckpt(str((repo_root / args.stage4_ckpt).resolve())),
        "Stage 5: +OEM": resolve_ckpt(str((repo_root / args.stage5_ckpt).resolve())),
        "Stage 6: +KD": resolve_ckpt(str((repo_root / args.stage6_ckpt).resolve())),
    }

    nets: Dict[str, torch.nn.Module] = {}
    for name, ckpt in ckpt_paths.items():
        net = build_ftunetformer()
        net = load_net_from_lightning_ckpt(net, ckpt).to(device)
        nets[name] = net

    # panels
    rgb_img = denormalize_to_uint8(img_t)
    gt_mask = colorize_mask(mask_t.detach().cpu().numpy().astype(np.uint8))

    preds_rgb: List[np.ndarray] = []
    for name in ckpt_paths.keys():
        pred = predict_mask(nets[name], img_t, device)
        preds_rgb.append(colorize_mask(pred))

    panels = [
        rgb_img,
        gt_mask,
        preds_rgb[0],
        preds_rgb[1],
        preds_rgb[2],
        preds_rgb[3],
        preds_rgb[4],
        preds_rgb[5],
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

    out_path = (repo_root / args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print("Repo root:", repo_root)
    print("Data root:", data_root)
    print("Tile:", args.img_id)
    print("Device:", device)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
