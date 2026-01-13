#!/usr/bin/env python3
"""
Generate per-sample difficulty weights for Stage 3 difficulty-weighted sampling.

Uses:
- A trained Stage 2 checkpoint (Lightning .ckpt)
- The Train_rep split (images/*.tif, masks/*.png)

Writes:
- artifacts/sample_weights.txt (tab-separated: img_id <TAB> weight)

Notes:
- Dataset order exactly matches BiodiversityTrainDataset iteration order
- Label 0 is treated as ignore_index for loss (matches training)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from geoseg.models.ftunetformer import ft_unetformer
from geoseg.datasets.biodiversity_dataset import (
    BiodiversityTrainDataset,
    val_aug,
)


def load_net_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError(f"Invalid Lightning checkpoint: {ckpt_path}")

    sd = ckpt["state_dict"]

    # Extract underlying segmentation network weights
    net_sd = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    if not net_sd:
        net_sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}

    if not net_sd:
        raise ValueError("Could not locate model weights in checkpoint")

    net.load_state_dict(net_sd, strict=False)
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        default="model_weights/biodiversity/stage2_replication_ftunetformer/stage2_replication_ftunetformer.ckpt",
        help="Stage 2 Lightning checkpoint",
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default="data/biodiversity_split/train_rep",
        help="Train_rep split root",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="artifacts/sample_weights.txt",
        help="Output weights file",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--ignore-index", type=int, default=0)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    data_root = Path(args.data_root)
    out_path = Path(args.out)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not (data_root / "images").exists():
        raise FileNotFoundError(f"Invalid data root: {data_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # IMPORTANT:
    # This must exactly match the dataset used in Stage 3
    ds = BiodiversityTrainDataset(
        data_root=str(data_root),
        transform=val_aug,  # deterministic, no cropping
    )

    net = ft_unetformer(num_classes=6, decoder_channels=256)
    net = load_net_from_lightning_ckpt(net, ckpt_path)
    net.to(device).eval()

    losses = np.zeros(len(ds), dtype=np.float32)
    img_ids: list[str] = []

    print("[analyze_hard_samples]")
    print(f"  ckpt:    {ckpt_path.resolve()}")
    print(f"  data:    {data_root.resolve()}")
    print(f"  samples: {len(ds)}")
    print(f"  device:  {device}")
    print(f"  out:     {out_path.resolve()}")

    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="Scoring difficulty"):
            sample = ds[i]
            img = sample["img"].unsqueeze(0).to(device)
            mask = sample["gt_semantic_seg"].unsqueeze(0).to(device)

            logits = net(img)
            loss = F.cross_entropy(
                logits,
                mask.long(),
                ignore_index=args.ignore_index,
                reduction="mean",
            )

            losses[i] = loss.item()
            img_ids.append(sample["img_id"])

    lo, hi = losses.min(), losses.max()
    norm = (losses - lo) / (hi - lo) if hi > lo else np.zeros_like(losses)
    weights = 1.0 + args.alpha * norm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for img_id, w in zip(img_ids, weights):
            f.write(f"{img_id}\t{w:.6f}\n")

    print("✓ Wrote sample weights")
    print(f"  loss min/med/max: {lo:.6f} / {np.median(losses):.6f} / {hi:.6f}")
    print(f"  weight min/med/max: {weights.min():.6f} / {np.median(weights):.6f} / {weights.max():.6f}")


if __name__ == "__main__":
    main()
