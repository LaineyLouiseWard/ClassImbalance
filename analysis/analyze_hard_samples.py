"""Identify hard samples and write sample_weights.txt for WeightedRandomSampler."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from geoseg.utils.cfg import py2cfg


def compute_sample_iou(pred, target, num_classes=6):
    """Compute per-sample IoU for each class and mean IoU over present classes."""
    ious = {}
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        ious[cls] = np.nan if union == 0 else (intersection / union)

    present = [v for v in ious.values() if not np.isnan(v)]
    mean_iou = float(np.mean(present)) if present else 0.0
    return ious, mean_iou


def load_student_from_lightning_ckpt(net, ckpt_path: str):
    """Loads only student weights from a Lightning checkpoint (keys prefixed with 'net.')."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no 'state_dict'.")
    model_state = {k.replace("net.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("net.")}
    net.load_state_dict(model_state, strict=False)
    return net


def analyze_dataset(config_path: str, checkpoint_path: str, out_dir: str = "."):
    print("Loading config...")
    config = py2cfg(Path(config_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Student model from config
    net = config.net
    net = load_student_from_lightning_ckpt(net, checkpoint_path)
    net.to(device).eval()

    # Pull dataset (prefer dataset object, else loader.dataset)
    if hasattr(config, "train_dataset") and config.train_dataset is not None:
        dataset = config.train_dataset
    else:
        dataset = config.train_loader.dataset

    print(f"Analyzing {len(dataset)} training samples on {device}...")

    class_names = ["Background", "Forest", "Grassland", "Cropland", "Settlement", "SemiNatural"]
    results = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing samples"):
            sample = dataset[idx]

            img = sample["img"]
            if not torch.is_tensor(img):
                img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()
            img = img.unsqueeze(0).to(device)

            mask = sample["gt_semantic_seg"]
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            else:
                mask = np.asarray(mask)

            img_id = str(sample.get("img_id", idx))

            logits = net(img)
            pred = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy()

            class_ious, mean_iou = compute_sample_iou(pred, mask, num_classes=6)
            class_counts = {cls: int((mask == cls).sum()) for cls in range(6)}

            results.append(
                {
                    "idx": idx,
                    "img_id": img_id,
                    "mean_iou": float(mean_iou),
                    "class_ious": {
                        class_names[k]: (float(v) if not np.isnan(v) else None) for k, v in class_ious.items()
                    },
                    "class_pixel_counts": {class_names[k]: class_counts[k] for k in range(6)},
                }
            )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = out_dir / "hard_samples_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis to: {analysis_path}")

    # Generate oversampling weights
    weights = []
    for r in results:
        weight = 1.0
        if r["mean_iou"] < 0.6:
            weight += 2.0
        elif r["mean_iou"] < 0.7:
            weight += 1.0

        forest_iou = r["class_ious"]["Forest"]
        settlement_iou = r["class_ious"]["Settlement"]

        if forest_iou is not None and forest_iou < 0.6 and r["class_pixel_counts"]["Forest"] > 100:
            weight += 1.5
        if settlement_iou is not None and settlement_iou < 0.6 and r["class_pixel_counts"]["Settlement"] > 100:
            weight += 1.5

        weights.append(float(weight))

    weights_path = out_dir / "sample_weights.txt"
    with open(weights_path, "w") as f:
        for idx, w in enumerate(weights):
            f.write(f"{idx}\t{w}\n")

    print(f"Saved {len(weights)} sample weights to: {weights_path}")
    print(f"Weight stats: mean={np.mean(weights):.2f}, max={np.max(weights):.2f}, >1={sum(w>1 for w in weights)}")

    return results, weights


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Config used to build the dataset (stage2 recommended).")
    p.add_argument("--ckpt", required=True, help="Baseline student checkpoint (.ckpt) used to score difficulty.")
    p.add_argument("--out_dir", default=".", help="Where to write sample_weights.txt")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    analyze_dataset(args.config, args.ckpt, out_dir=args.out_dir)
