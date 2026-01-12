from torch.utils.data import DataLoader
import torch

from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    train_aug_random,
    val_aug,
    BiodiversityTiffTrainDataset,
    BiodiversityTiffTestDataset,
)
from geoseg.models.FTUNetFormer import ft_unetformer
from geoseg.utils.utils import Lookahead, process_model_params


# -----------------------
# Training hyperparams
# -----------------------
max_epoch = 45
ignore_index = 255

train_batch_size = 4
val_batch_size = 4

lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES

# -----------------------
# Logging / checkpoints
# -----------------------
weights_name = "stage1_baseline_ftunetformer"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_F1"
monitor_mode = "max"
save_top_k = 3
save_last = False
check_val_every_n_epoch = 1

pretrained_ckpt_path = None
gpus = "auto"
resume_ckpt_path = None


# -----------------------
# Model (NO PRETRAIN)
# -----------------------
net = ft_unetformer(
    pretrained=False,           # IMPORTANT: avoids looking for pretrain_weights/stseg_base.pth
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


# -----------------------
# Datasets (your split)
# -----------------------
train_dataset = BiodiversityTiffTrainDataset(
    data_root="data/biodiversity_split/train",
    img_dir="images",
    mask_dir="masks",
    img_suffix=".tif",
    mask_suffix=".png",
    mosaic_ratio=0.25,
    transform=train_aug_random,
)

val_dataset = BiodiversityTiffTrainDataset(
    data_root="data/biodiversity_split/val",
    img_dir="images",
    mask_dir="masks",
    img_suffix=".tif",
    mask_suffix=".png",
    mosaic_ratio=0.0,
    transform=val_aug,
)

# If your test split has no masks (recommended), this is correct:
test_dataset = BiodiversityTiffTestDataset(
    data_root="data/biodiversity_split/test",
    img_dir="images",
    img_suffix=".tif",
)


# -----------------------
# Loaders
# -----------------------
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


# -----------------------
# Optimizer / scheduler
# -----------------------
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
