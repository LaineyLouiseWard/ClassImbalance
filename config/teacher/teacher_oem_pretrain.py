"""
Teacher training: EfficientNet-B4 U-Net on OEM native taxonomy.

This produces the teacher checkpoint used later for KD.

Run:
  python train_supervision.py -c config/openearthmap/teacher_oem_pretrain.py
"""

from torch.utils.data import DataLoader
import torch

from geoseg.losses import *
from geoseg.models.unet import TeacherUNet
from geoseg.utils.utils import Lookahead, process_model_params
from geoseg.datasets.openearthmap_dataset import OpenEarthMapTeacherDataset, oem_train_aug, oem_val_aug

# -------------------------
# hparams
# -------------------------
max_epoch = 50
ignore_index = 255

train_batch_size = 4
val_batch_size = 4

lr = 6e-4
weight_decay = 2.5e-4

# IMPORTANT:
# OEM native is commonly 8 classes (0..7).
# If your dataset scan shows 0..8, set this to 9.
num_classes = 8
classes = tuple([f"OEM_{i}" for i in range(num_classes)])

weights_name = "teacher_unet_efficientnetb4_oem_pretrain"
weights_path = f"pretrain_weights/{weights_name}"
test_weights_name = weights_name
log_name = f"openearthmap/{weights_name}"

monitor = "val_F1"
monitor_mode = "max"
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = "auto"

# -------------------------
# model
# -------------------------
net = TeacherUNet(num_classes=num_classes, pretrained=True)

# -------------------------
# loss
# -------------------------
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0, 1.0
)
use_aux_loss = False

# -------------------------
# data
# -------------------------
train_dataset = OpenEarthMapTeacherDataset(
    data_root="data/openearthmap_teacher/train",
    transform=oem_train_aug,
)
val_dataset = OpenEarthMapTeacherDataset(
    data_root="data/openearthmap_teacher/val",
    transform=oem_val_aug,
)

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

# -------------------------
# optim
# -------------------------
# TeacherUNet doesn't have your "backbone.*" naming like FTUNetFormer,
# so keep it simple unless you inspect its parameter names.
net_params = process_model_params(net, layerwise_params=None)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
