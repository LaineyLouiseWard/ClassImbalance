"""
Stage 6: Final KD consolidation stage (cumulative).

Target: Biodiversity TIFF only (your split)
Cumulative components kept ON:
- replication (train_rep split)
- difficulty-weighted sampling (artifacts/sample_weights.txt)
- minority-aware cropping (train_aug_minority)
PLUS:
- knowledge distillation (teacher -> student)

IMPORTANT:
- ignore_index = 0 everywhere
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
    train_aug_minority,
    val_aug,
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.models.unet import TeacherUNet
from geoseg.utils.kd_utils import KDHelper, create_mapping_matrix
from geoseg.utils.optim import Lookahead, process_model_params


# ======================
# Training hyperparams
# ======================
max_epoch = 60
ignore_index = 0

train_batch_size = 2
val_batch_size = 2

lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# ======================
# KD parameters
# ======================
kd_temperature = 2.0
kd_alpha = 0.3
rangeland_split_alpha = 0.7

teacher_checkpoint = "pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth"


# ======================
# Logging / checkpoints
# ======================
weights_name = "stage6_final_kd_ftunetformer"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_F1"
monitor_mode = "max"
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = "auto"

pretrained_ckpt_path = None
resume_ckpt_path = None


# ======================
# Models
# ======================
net = ft_unetformer(
    pretrained=False,
    weight_path=None,
    num_classes=num_classes,
    decoder_channels=256,
)

teacher = TeacherUNet(num_classes=9, pretrained=False)
teacher.load_checkpoint(teacher_checkpoint)
teacher.freeze()

mapping_matrix = create_mapping_matrix(alpha=rangeland_split_alpha)
kd_helper = KDHelper(mapping_matrix=mapping_matrix, temperature=kd_temperature)


# ======================
# Loss
# ======================
hard_loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)


class KDLoss(nn.Module):
    def __init__(self, hard_loss: nn.Module, kd_helper: KDHelper, alpha: float):
        super().__init__()
        self.hard_loss = hard_loss
        self.kd_helper = kd_helper
        self.alpha = alpha

    def forward(self, student_logits, targets, teacher_logits):
        loss_hard = self.hard_loss(student_logits, targets)
        loss_kd = self.kd_helper.compute_kd_loss(student_logits, teacher_logits)
        return (1.0 - self.alpha) * loss_hard + self.alpha * loss_kd


loss = KDLoss(hard_loss, kd_helper, alpha=kd_alpha)
use_aux_loss = False


# ======================
# Datasets
# ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train_rep",
    transform=train_aug_minority,
)

val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)

test_dataset = BiodiversityTestDataset(
    data_root="data/biodiversity_split/test",
)


# ======================
# Difficulty weights
# ======================
repo_root = Path(__file__).resolve().parents[2]
sample_weights_path = repo_root / "artifacts" / "sample_weights.txt"

sample_weights = []
with open(sample_weights_path, "r", encoding="utf-8") as f:
    for line in f:
        _, w = line.strip().split("\t")
        sample_weights.append(float(w))

if len(sample_weights) != len(train_dataset):
    raise ValueError(
        f"sample_weights ({len(sample_weights)}) != train_dataset ({len(train_dataset)})"
    )

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)


# ======================
# Loaders
# ======================
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


# ======================
# Optimizer
# ======================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
