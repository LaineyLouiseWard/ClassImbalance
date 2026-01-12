#!/usr/bin/env python3
"""
Analyze EFFECTIVE class distribution (what the model actually sees) for a given stage config.

This estimates the distribution under stochastic sampling/augmentation by drawing N samples
from config.train_dataset and config.val_dataset (default N=200).

Usage:
  python analyze_class_distribution.py --config config/biodiversity/stage1_baseline.py
  python analyze_class_distribution.py --config config/biodiversity/stage4_minority.py --n-samples 200 --seed 0
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from geoseg.utils.cfg import py2cfg

# Class names (fixed taxonomy)
CLASSES = ["Background", "Forest", "Grassland", "Cropland", "Settlement", "SemiNatural"]


def _seed_everything(seed: int) -> None:
    """Best-effort reproducibility for stochastic dataset transforms."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mask_to_numpy(sample) -> np.ndarray:
    """
    Extract mask array from a dataset sample.
    Expected key (per geoseg): 'gt_semantic_seg'
    """
    mask = sample["gt_semantic_seg"]
    if hasattr(mask, "detach"):
        mask = mask.detach()
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    return mask


def analyze_dataset_effective(dataset, name: str, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate class distribution by sampling N items from the dataset."""
    print(f"\n{'='*70}")
    print(f"Analyzing {name} (effective distribution)")
    print(f"{'='*70}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Sampling N={n_samples} samples\n")

    class_counts = np.zeros(len(CLASSES), dtype=np.int64)
    total_pixels = 0

    # Sample indices with replacement if dataset is smaller than n_samples
    if len(dataset) == 0:
        raise ValueError(f"{name}: dataset length is 0")

    indices = np.random.randint(0, len(dataset), size=n_samples)

    for idx in tqdm(indices, desc=f"Sampling {name}"):
        sample = dataset[int(idx)]
        mask = _mask_to_numpy(sample)

        # Count pixels per class
        for class_id in range(len(CLASSES)):
            class_counts[class_id] += np.sum(mask == class_id)

        total_pixels += mask.size

    class_percentages = (class_counts / max(total_pixels, 1)) * 100.0

    # Inverse-frequency weights (illustrative; not automatically used in training)
    class_weights = np.zeros_like(class_percentages, dtype=np.float64)
    for i, pct in enumerate(class_percentages):
        class_weights[i] = (100.0 / pct) if pct > 0 else 0.0

    # Normalize so min non-zero weight is 1.0
    non_zero = class_weights[class_weights > 0]
    if non_zero.size > 0:
        class_weights = class_weights / non_zero.min()

    # Print table
    print(f"\nTotal sampled pixels: {total_pixels:,}")
    print(f"\nClass Distribution (estimated):")
    print(f"{'Class':<20} {'Pixels':>15} {'Percentage':>12}")
    print("-" * 50)
    for class_name, count, pct in zip(CLASSES, class_counts, class_percentages):
        print(f"{class_name:<20} {count:>15,} {pct:>11.2f}%")

    print(f"\nRecommended Class Weights (inverse frequency; estimated):")
    print(f"{'Class':<20} {'Weight':>10}")
    print("-" * 32)
    for class_name, w in zip(CLASSES, class_weights):
        print(f"{class_name:<20} {w:>10.2f}")

    return class_counts, class_percentages, class_weights


def plot_distribution(train_pct: np.ndarray, val_pct: np.ndarray, output_path: Path) -> None:
    """Plot Train vs Val distribution (linear + log y) as a single figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(CLASSES))
    width = 0.35

    # Linear scale
    bars1 = ax1.bar(x - width / 2, train_pct, width, label="Train", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, val_pct, width, label="Val", alpha=0.8)

    ax1.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Effective Class Distribution: Train vs Val", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    # Log scale
    ax2.bar(x - width / 2, train_pct, width, label="Train", alpha=0.8)
    ax2.bar(x + width / 2, val_pct, width, label="Val", alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Percentage (%) - Log Scale", fontsize=12, fontweight="bold")
    ax2.set_title("Effective Class Distribution (Log Scale)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def _stage_name_from_config_path(config_path: str) -> str:
    p = Path(config_path)
    return p.stem  # e.g. stage4_minority


def main():
    parser = argparse.ArgumentParser(description="Analyze effective class distribution for a stage config.")
    parser.add_argument("--config", type=str, required=True, help="Path to stage config .py file")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of samples to draw per split")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling/augmentation")
    parser.add_argument("--out-dir", type=str, default="analysis_outputs", help="Output directory for plots/weights")
    args = parser.parse_args()

    _seed_everything(args.seed)

    config_path = args.config
    stage = _stage_name_from_config_path(config_path)
    out_dir = Path(args.out_dir) / "class_distribution" / stage
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from: {config_path}")
    cfg = py2cfg(config_path)

    # Estimate effective distributions
    train_counts, train_pct, train_w = analyze_dataset_effective(
        cfg.train_dataset, name="Training Dataset", n_samples=args.n_samples
    )
    val_counts, val_pct, val_w = analyze_dataset_effective(
        cfg.val_dataset, name="Validation Dataset", n_samples=args.n_samples
    )

    # Plot
    plot_path = out_dir / f"class_distribution_{stage}_N{args.n_samples}_seed{args.seed}.png"
    plot_distribution(train_pct, val_pct, output_path=plot_path)

    # Save weights (based on train)
    weights_file = out_dir / f"recommended_class_weights_{stage}_N{args.n_samples}_seed{args.seed}.txt"
    with open(weights_file, "w") as f:
        f.write("Recommended Class Weights (based on TRAIN effective distribution estimate)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Stage: {stage}\n")
        f.write(f"N samples: {args.n_samples}\n")
        f.write(f"Seed: {args.seed}\n\n")

        f.write("For PyTorch CrossEntropyLoss:\n")
        f.write("weight = torch.FloatTensor([")
        f.write(", ".join([f"{w:.4f}" for w in train_w]))
        f.write("])\n\n")

        f.write("For config file:\n")
        f.write("class_weights = [")
        f.write(", ".join([f"{w:.4f}" for w in train_w]))
        f.write("]\n\n")

        f.write("Class mapping:\n")
        for i, (cls, w) in enumerate(zip(CLASSES, train_w)):
            f.write(f"{i}: {cls:<20} weight={w:.4f}\n")

    print(f"Weights saved to: {weights_file}")
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
