import os
import sys
from pathlib import Path
import json
import datetime

# Add project root to Python path (repo root = parent of evaluation/)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn as nn

from torch.utils.data import DataLoader

from geoseg.utils.metric import Evaluator
from geoseg.datasets.biodiversity_dataset import BiodiversityValDataset
from geoseg.models.UNetFormer import UNetFormer

# Optional: if you have DCSwin in your repo and want to evaluate it too
try:
    from geoseg.models.DCSwin import dcswin_base
except Exception:
    dcswin_base = None


CLASS_NAMES_6 = [
    "Background",
    "Forest land",
    "Grassland",
    "Cropland",
    "Settlement",
    "Seminatural Grassland",
]
CLASS_NAMES_5 = CLASS_NAMES_6[1:]


def build_model(model_name: str, num_classes: int = 6) -> torch.nn.Module:
    """
    model_name: "unetformer" or "dcswin"
    NOTE: We default to pretrained=False to avoid missing weight files.
    """
    model_name = model_name.lower()

    if model_name == "dcswin":
        if dcswin_base is None:
            raise RuntimeError("dcswin_base could not be imported. Is geoseg/models/DCSwin.py present?")
        return dcswin_base(num_classes=num_classes, pretrained=False, weight_path=None)

    # default
    return UNetFormer(num_classes=num_classes)


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)

    # lightning ckpt typically has 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # remove common lightning prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k.replace("model.", "", 1)] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    return model


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    model_name: str,
    val_root: Path,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 0,
):
    model = build_model(model_name=model_name, num_classes=6).to(device)
    model = load_checkpoint_into_model(model, ckpt_path, device)
    model.eval()

    val_dataset = BiodiversityValDataset(data_root=str(val_root))
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    evaluator = Evaluator(num_class=6)
    cm = np.zeros((6, 6), dtype=np.int64)

    softmax = nn.Softmax(dim=1)

    for batch in tqdm(val_loader, desc=f"Evaluating {ckpt_path.name}", leave=False):
        images = batch["img"].to(device)
        masks = batch["gt_semantic_seg"].cpu().numpy()  # (B,H,W)

        outputs = model(images)
        preds = softmax(outputs).argmax(dim=1).cpu().numpy()

        for true, pred in zip(masks, preds):
            cm += confusion_matrix(true.flatten(), pred.flatten(), labels=list(range(6)))
            evaluator.add_batch(true, pred)

    # Metrics excluding background
    iou_all = evaluator.Intersection_over_Union()
    f1_all = evaluator.F1()
    oa = float(evaluator.OA())

    iou_no_bg = iou_all[1:]
    f1_no_bg = f1_all[1:]
    miou = float(np.mean(iou_no_bg))
    mf1 = float(np.mean(f1_no_bg))

    metrics = {
        "checkpoint": str(ckpt_path),
        "model_name": model_name,
        "val_root": str(val_root),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "OA": oa,
        "mIoU_excluding_bg": miou,
        "mF1_excluding_bg": mf1,
        "per_class_iou": {CLASS_NAMES_6[i]: float(iou_all[i]) for i in range(6)},
        "per_class_f1": {CLASS_NAMES_6[i]: float(f1_all[i]) for i in range(6)},
    }
    return metrics, cm


def plot_confusion_matrix(cm: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_pct = cm / row_sums

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_pct, interpolation="nearest")
    plt.title("Confusion Matrix (row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(6), CLASS_NAMES_6, rotation=45, ha="right")
    plt.yticks(range(6), CLASS_NAMES_6)

    for i in range(6):
        for j in range(6):
            plt.text(j, i, f"{cm_pct[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_bars(values: list[float], labels: list[str], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(labels)), values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def infer_model_name_from_path(p: Path) -> str:
    s = str(p).lower()
    if "dcswin" in s:
        return "dcswin"
    return "unetformer"


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate checkpoints and write metrics.json + plots")
    ap.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Directory to search for .ckpt files (recursively). Example: model_weights/",
    )
    ap.add_argument(
        "--val-root",
        type=str,
        default="data/biodiversity_split/val",
        help="Validation dataset root (contains images/ and masks/).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="evaluation_results",
        help="Where to write evaluation outputs.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.ckpt",
        help="Checkpoint filename pattern (default: *.ckpt).",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers (start with 0 to avoid multiprocessing issues).",
    )
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    val_root = Path(args.val_root).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpts = sorted(base_dir.rglob(args.pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {base_dir} with pattern {args.pattern}")

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Found {len(ckpts)} checkpoints under {base_dir}")

    for ckpt in ckpts:
        model_name = infer_model_name_from_path(ckpt)

        # results per-checkpoint folder
        safe_name = ckpt.parent.name + "__" + ckpt.stem
        run_dir = out_root / safe_name
        run_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Evaluating: {ckpt} (model={model_name})")

        metrics, cm = evaluate_checkpoint(
            ckpt_path=ckpt,
            model_name=model_name,
            val_root=val_root,
            device=device,
            batch_size=1,
            num_workers=args.num_workers,
        )

        # write metrics.json
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # plots
        plot_confusion_matrix(cm, run_dir)
        iou_no_bg = [metrics["per_class_iou"][c] for c in CLASS_NAMES_5]
        f1_no_bg = [metrics["per_class_f1"][c] for c in CLASS_NAMES_5]

        plot_class_bars(
            iou_no_bg,
            CLASS_NAMES_5,
            title=f"Per-Class IoU (excl. Background)\n{ckpt.name}",
            ylabel="IoU",
            out_path=run_dir / "class_iou_scores.png",
        )
        plot_class_bars(
            f1_no_bg,
            CLASS_NAMES_5,
            title=f"Per-Class F1 (excl. Background)\n{ckpt.name}",
            ylabel="F1",
            out_path=run_dir / "class_f1_scores.png",
        )

        # text report
        with open(run_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write("=== Evaluation Report ===\n\n")
            f.write(f"Checkpoint: {metrics['checkpoint']}\n")
            f.write(f"Model: {metrics['model_name']}\n")
            f.write(f"Val root: {metrics['val_root']}\n\n")
            f.write(f"Overall Accuracy (OA): {metrics['OA']:.4f}\n")
            f.write(f"Mean IoU (excl. bg): {metrics['mIoU_excluding_bg']:.4f}\n")
            f.write(f"Mean F1 (excl. bg): {metrics['mF1_excluding_bg']:.4f}\n\n")
            f.write("Per-class IoU:\n")
            for c in CLASS_NAMES_6:
                f.write(f"  {c}: {metrics['per_class_iou'][c]:.4f}\n")

    logging.info(f"Done. Outputs written to {out_root}")


if __name__ == "__main__":
    main()
