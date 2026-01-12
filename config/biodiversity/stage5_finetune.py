"""
Stage 5B: Finetune after OEM pretraining.

Cumulative ablation:
  - replication
  - difficulty-weighted sampling
  - minority-aware cropping

Initialized from Stage 5A OEM-pretrained checkpoint.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    train_aug_minority,
    val_aug,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# -------------------
# Training hyperparams
# -------------------
max_epoch = 45
ignore_index = 0  # background is class 0 (pipeline decision)

train_batch_size = 4
val_batch_size = 4

lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# -------------------
# Logging / checkpoints
# -------------------
weights_name = "stage5_finetune_after_oem_ftunetformer"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_F1"
monitor_mode = "max"
save_top_k = 3
save_last = False
check_val_every_n_epoch = 1

# Stage 5A checkpoint (OEM-pretrained student)
pretrained_ckpt_path = (
    "model_weights/stage5_oem_pretrain_student/"
    "stage5_oem_pretrain_student.ckpt"
)

resume_ckpt_path = None
gpus = "auto"


# -------------------
# Model / loss
# -------------------
net = ft_unetformer(
    pretrained=False,
    weight_path=None,
    num_classes=num_classes,
    decoder_channels=256,
)

loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)

use_aux_loss = False


# -------------------
# Datasets
# -------------------
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train_rep",
    transform=train_aug_minority,   # minority-aware cropping
)

val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)

test_dataset = BiodiversityTestDataset(
    data_root="data/biodiversity_split/test",
)


# -------------------
# Difficulty weights -> sampler
# -------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sample_weights_path = REPO_ROOT / "artifacts" / "sample_weights.txt"

if not sample_weights_path.exists():
    raise FileNotFoundError(
        f"Missing sample weights: {sample_weights_path}\n"
        "Regenerate with: python evaluation/analyze_hard_samples.py"
    )

sample_weights = []
with open(sample_weights_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        _, w = line.split("\t")
        sample_weights.append(float(w))

if len(sample_weights) != len(train_dataset):
    raise ValueError(
        f"sample_weights length ({len(sample_weights)}) != "
        f"train_dataset length ({len(train_dataset)})."
    )

print(f"[stage5B] Loaded {len(sample_weights)} sample weights from {sample_weights_path}")

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)


# -------------------
# Loaders
# -------------------
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    sampler=sampler,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)


# -------------------
# Optimizer / scheduler
# -------------------
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(
    net_params, lr=lr, weight_decay=weight_decay
)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
