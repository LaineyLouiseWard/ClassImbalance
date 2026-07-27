"""
Stage 2a (OEM transfer, pre-train): supervised training on the combined dataset
(Biodiversity + OEM, harmonised to 6 classes via the grounded OEM->student mapping).

First half of the Stage 2 OEM-transfer step; Stage 2b (stage2b_oem_finetune.py) fine-tunes
this checkpoint on Biodiversity alone.

Fair ablation rule (for the paper):
- Use the SAME core training hyperparams as Stage 1 (lr/backbone_lr/weight_decay/etc.)
- OEM data is used ONLY in this pretraining stage via the combined 6-class dataset.
- Validation is Biodiversity-only to avoid OEM leakage into reported val curves.

Run with:
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2a_oem_pretrain.py
"""

from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    train_aug_random,
    val_aug,
    BiodiversityValDataset,
)
from geoseg.datasets.biodiversity_oem_dataset import BiodiversityOEMTrainDataset
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# -------------------
# Training hyperparams (MATCH STAGE 1)
# -------------------
max_epoch = 45  # two complete CosineAnnealingWarmRestarts cycles (T_0=15, T_mult=2),
                # so training ends at an LR minimum. 50 would end mid-cycle and break the
                # save_last/checkpoint rationale. The next valid stop is 105.
ignore_index = 0  # background ignored in loss/metrics

# --- Batch/LR variant (env-gated): BATCH_VARIANT=b2 (default) | b4 — MUST match across all 5 cells ---
# --- Data split (env-gated): BIO_SPLIT selects which spatially blocked assignment to use.
#   Default is the legacy random-by-tile split, which LEAKS (see
#   notes/rebuild_2026-07/decisions/TILE_OVERLAP_LEAKAGE_2026-07-25.md); the campaign must set it explicitly. ---
_BIO_SPLIT = os.environ["BIO_SPLIT"]  # required: the old default was the LEAKY split
_BIO_OEM = os.environ["BIO_OEM_COMBINED"]  # required: the old default was the WITHDRAWN pool
# Was os.environ.get(..., "data/biodiversity_oem_combined") -- a SOFT default beside a hard
# BIO_SPLIT, pointing at the withdrawn pool. That pool's train set holds 239 of the 294 Test A
# tiles, 153 of the 191 Test B tiles and 141 of the 173 val tiles, so an unset variable would
# pre-train stage 2a on most of both test sets and inflate exactly the transfer contrast this
# paper measures -- plausibly, and at exit 0. Both launchers do export it; that is not a reason
# to keep a default that only ever resolves to withdrawn data.

_BV = os.environ.get("BATCH_VARIANT", "b2")
assert _BV in ("b2", "b4"), f"BATCH_VARIANT must be b2 or b4, got {_BV!r}"
_LR_SCALE = 2.0 if _BV == "b4" else 1.0
_BATCH = 4 if _BV == "b4" else 2

train_batch_size = _BATCH
val_batch_size = _BATCH

# IMPORTANT: Use Stage 1 values (your chosen "fair ablation" base)
lr = 3e-4 * _LR_SCALE
weight_decay = 2.5e-4
backbone_lr = 3e-5 * _LR_SCALE
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# -------------------
# Logging / checkpoints
# -------------------
# Fold tag in every output path. Without it f2 resumed from f1's last.ckpt and the evaluation
# directories collided, while f1's training set holds 100% of f3's test tiles.
FOLD_TAG = os.environ.get("SPLIT_TAG", "")
_SUF = (f"_{FOLD_TAG}" if FOLD_TAG else "") + os.environ.get("RUN_TAG", "")

weights_name = f"stage2a_oem_pretrain{_SUF}"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
# save_last kept alongside save_top_k so the last-epoch model can be reported as a
# selection-rule sensitivity. Best-val stays primary (matches the FT-UNetFormer reference
# implementation); 45 epochs = two full CosineAnnealingWarmRestarts cycles (T_0=15, T_mult=2),
# so the last epoch lands at a learning-rate minimum.
save_last = True
check_val_every_n_epoch = 1

# Stage 3a starts from scratch
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = "auto"


# -------------------
# Model / loss (ADE20K-pretrained Swin-B backbone via stseg_base.pth)
# -------------------
net = ft_unetformer(
    pretrained=True,
    weight_path="pretrain_weights/stseg_base.pth",
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
# Combined pretraining dataset (Biodiversity + OEM already harmonised to 0..5)
train_dataset = BiodiversityOEMTrainDataset(
    data_root=f"{_BIO_OEM}/train",
    transform=train_aug_random,
)

# Validation is Biodiversity-only (no OEM leakage)
val_dataset = BiodiversityValDataset(
    data_root=f"{_BIO_SPLIT}/val",
    transform=val_aug,
)


# -------------------
# Loaders
# -------------------
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
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

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
