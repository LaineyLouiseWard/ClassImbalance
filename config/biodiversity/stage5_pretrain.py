"""
Stage 5A: OEM pretraining on combined dataset (biodiversity + OEM).

- OEM labels already harmonised to 6-class taxonomy (0–5).
- OEM used ONLY here; removed for finetune/eval.
"""

from torch.utils.data import DataLoader
import torch

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    train_aug_random,
    val_aug,
    BiodiversityValDataset,
)
from geoseg.datasets.biodiversity_oem_dataset import (
    BiodiversityOEMTrainDataset,   # <-- CHANGED
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# -------------------
# Training hyperparams
# -------------------
max_epoch = 50
ignore_index = 0

train_batch_size = 2
val_batch_size = 2

lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# -------------------
# Logging / checkpoints
# -------------------
weights_name = "stage5_oem_pretrain_student"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_F1"
monitor_mode = "max"
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1

pretrained_ckpt_path = None
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
# Combined pretraining dataset (biodiversity + OEM)
train_dataset = BiodiversityOEMTrainDataset(     # <-- CHANGED
    data_root="data/biodiversity_oem_combined/train",
    transform=train_aug_random,
)

# Validation = biodiversity only (no OEM leakage)
val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)


# -------------------
# Loaders
# -------------------
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=0,      # OK for stability during pretrain
    pin_memory=False,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
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
