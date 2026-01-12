#!/usr/bin/env python3
"""
Generate per-sample difficulty weights for Stage 3 difficulty-weighted sampling.

Uses:
- A trained Stage 2 checkpoint (Lightning .ckpt)
- The Train_rep split (images/*.tif, masks/*.png)

Writes:
- artifacts/sample_weights.txt  (tab-separated: img_id <TAB> weight)
  IMPORTANT: the line order matches the dataset ordering, so the weights list
  aligns with train_dataset indexing.

No KD / teacher models.
"""

from __future__ import annotations

# --- make sure repo root is importable so `import geoseg` works ---
import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parents[1]  # .../ClassImbalance
sys.path.insert(0, str(REPO_ROOT))
# ---------------------------------------------------------------

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from geoseg.models.FTUNetFormer import ft_unetformer
from geoseg.datasets.biodiversity_dataset import BiodiversityTrainDataset, val_aug


def load_net_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint does not look like a Lightning .ckpt (missing state_dict): {ckpt_path}")

    sd = ckpt["state_dict"]

    # train_supervision.py saves the LightningModule where the actual network is under "net."
    # So keys look like "net.backbone...."
    net_sd = {}
    for k, v in sd.items():
        if k.startswith("net."):
            net_sd[k.replace("net.", "", 1)] = v

    if not net_sd:
        # fallback: sometimes people save directly, or with "model."
        for k, v in sd.items():
            if k.startswith("model."):
                net_sd[k.replace("model.", "", 1)] = v

    if not net_sd:
        raise ValueError(
            "Could not find model weights under 'net.' or 'model.' in state_dict. "
            "Open the ckpt and inspect keys."
        )

    missing, unexpected = net.load_state_dict(net_sd, strict=False)
    if missing:
        print(f"[warn] Missing keys (non-fatal): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys (non-fatal): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        default="model_weights/stage2_replication_ftunetformer/stage2_replication_ftunetformer.ckpt",
        help="Stage 2 Lightning checkpoint (.ckpt) used to define difficulty",
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default="data/biodiversity_split/train_rep",
        help="Train_rep split root (expects images/ and masks/ inside)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="artifacts/sample_weights.txt",
        help="Output weights file (tab-separated: img_id\\tweight)",
    )
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument(
        "--alpha",
        type=float,
        default=4.0,
        help="Weight scaling. Final weights are in [1, 1+alpha] after normalization.",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    data_root = Path(args.data_root)
    out_path = Path(args.out)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")
    if not (data_root / "images").exists() or not (data_root / "masks").exists():
        raise FileNotFoundError(f"Expected {data_root}/images and {data_root}/masks to exist.")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Build dataset in a deterministic order, no mosaic, no random crop
    ds = BiodiversityTrainDataset(
        data_root=str(data_root),
        img_dir="images",
        mask_dir="masks",
        img_suffix=".tif",
        mask_suffix=".png",
        mosaic_ratio=0.0,
        transform=val_aug,  # just normalize; no random cropping
    )

    # Build Stage 2 architecture
    net = ft_unetformer(num_classes=6, decoder_channels=256)
    net = load_net_from_lightning_ckpt(net, ckpt_path)
    net.to(device).eval()

    losses = np.zeros(len(ds), dtype=np.float32)
    img_ids: list[str] = []

    print("[analyze_hard_samples]")
    print(f"  ckpt:     {ckpt_path.resolve()}")
    print(f"  data:     {data_root.resolve()}")
    print(f"  samples:  {len(ds)}")
    print(f"  device:   {device}")
    print(f"  out:      {out_path.resolve()}")

    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="Scoring difficulty"):
            sample = ds[i]
            img = sample["img"].unsqueeze(0).to(device)              # (1,C,H,W)
            mask = sample["gt_semantic_seg"].unsqueeze(0).to(device) # (1,H,W)
            img_id = sample.get("img_id", str(i))
            img_ids.append(img_id)

            logits = net(img)  # (1,num_classes,H,W)

            # Cross-entropy difficulty (stable + monotonic)
            # ignore_index=255 is safe even if masks are 0..5 (it will just never apply)
            loss = F.cross_entropy(logits, mask.long(), ignore_index=255, reduction="mean")
            losses[i] = float(loss.item())

    # Normalize losses -> weights in [1, 1+alpha]
    lo = float(losses.min())
    hi = float(losses.max())
    if hi > lo:
        norm = (losses - lo) / (hi - lo)
    else:
        norm = np.zeros_like(losses)

    weights = 1.0 + float(args.alpha) * norm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for img_id, w in zip(img_ids, weights.tolist()):
            f.write(f"{img_id}\t{w:.6f}\n")

    print("\n✓ Wrote sample weights")
    print(f"  loss min/median/max: {lo:.6f} / {np.median(losses):.6f} / {hi:.6f}")
    print(f"  weight min/median/max: {weights.min():.6f} / {np.median(weights):.6f} / {weights.max():.6f}")
    print(f"  file: {out_path.resolve()}")


if __name__ == "__main__":
    main()
