"""
Stage 2b (OEM transfer, fine-tune): OEM->Biodiversity finetune on the Biodiversity train split.

Second half of the Stage 2 OEM-transfer step. Initialises from the Stage 2a OEM pre-train
checkpoint (stage2a_oem_pretrain) and fine-tunes on Biodiversity `train` at native class
frequency (no static minority duplication anywhere in the pipeline). The single imbalance
mechanism — the clsbal class-balanced sampler (Kang 2020) — is introduced in the sampler cells
(stage3_clsbal / stage_sampler_only), on top of this checkpoint.

Decoupling rationale (normal sampling here, rebalance later):
- Kang et al. 2020 (Decoupling) and Zhou et al. 2020 (BBN) show instance-balanced (normal) sampling
  learns the most generalisable representations; rebalancing is deferred to a later stage.
- Caveat (dense prediction): this is a classification finding. In segmentation the i.i.d. premise is
  weaker (Cui et al. 2022, Region Rebalance; Li et al. 2024, Frequency-based Matcher). We therefore
  treat normal-sampling transfer as the *intuition*, and let the sampler stage do the rebalancing.

Run:
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2b_oem_finetune.py --force
"""

from __future__ import annotations

import os
from torch.utils.data import DataLoader
import torch

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    train_aug_random,
    val_aug,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# -------------------
# Training hyperparams (match Stage 1 / Stage 2a)
# -------------------
max_epoch = 50  # raised from 45: the folds train on ~1.1k tiles, not 1706
ignore_index = 0

# --- Batch/LR variant (env-gated): BATCH_VARIANT=b2 (default) | b4 — MUST match across all 5 cells ---
# --- Data split (env-gated): BIO_SPLIT selects which spatially blocked assignment to use.
#   Default is the legacy random-by-tile split, which LEAKS (see
#   notes/TILE_OVERLAP_LEAKAGE_2026-07-25.md); the campaign must set it explicitly. ---
_BIO_SPLIT = os.environ["BIO_SPLIT"]  # required: the old default was the LEAKY split
_BIO_OEM = os.environ.get("BIO_OEM_COMBINED", "data/biodiversity_oem_combined")

_BV = os.environ.get("BATCH_VARIANT", "b2")
assert _BV in ("b2", "b4"), f"BATCH_VARIANT must be b2 or b4, got {_BV!r}"
_LR_SCALE = 2.0 if _BV == "b4" else 1.0
_BATCH = 4 if _BV == "b4" else 2

train_batch_size = _BATCH
val_batch_size = _BATCH

lr = 3e-4 * _LR_SCALE
weight_decay = 2.5e-4
backbone_lr = 3e-5 * _LR_SCALE
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# -------------------
# Logging / checkpoints  (ISOLATED name)
# -------------------
# Fold tag in every output path. Without it f2 resumed from f1's last.ckpt and the evaluation
# directories collided, while f1's training set holds 100% of f3's test tiles.
FOLD_TAG = os.environ.get("SPLIT_TAG", "")
_SUF = (f"_{FOLD_TAG}" if FOLD_TAG else "") + os.environ.get("RUN_TAG", "")

weights_name = f"stage2b_oem_finetune{_SUF}"
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

# init from the Stage 2a OEM pretrain checkpoint
pretrained_ckpt_path = (
    "model_weights/biodiversity/"
    "stage2a_oem_pretrain/"
    "stage2a_oem_pretrain.ckpt"
)

resume_ckpt_path = None
gpus = "auto"


# -------------------
# Model / loss (IDENTICAL)
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
# Datasets  -- Biodiversity train at native class frequency (no replication)
# -------------------
train_dataset = BiodiversityTrainDataset(
    data_root=f"{_BIO_SPLIT}/train",
    transform=train_aug_random,
)

val_dataset = BiodiversityValDataset(
    data_root=f"{_BIO_SPLIT}/val",
    transform=val_aug,
)

test_dataset = BiodiversityTestDataset(
    data_root=f"{_BIO_SPLIT}/test",
)


# -------------------
# Loaders (vanilla shuffle; the clsbal sampler is applied only in the sampler cells) -- IDENTICAL
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
# Optimizer / scheduler (IDENTICAL)
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
