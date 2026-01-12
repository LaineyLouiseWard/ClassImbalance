"""Stage 4: replication + difficulty-weighted sampling + minority-aware cropping."""

from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import torch

from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import *
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.utils import Lookahead, process_model_params

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

weights_name = "stage4_replication_difficulty_minoritycrop_ftunetformer"
weights_path = f"model_weights/{weights_name}"
test_weights_name = weights_name
log_name = f"{weights_name}"

monitor = "val_F1"
monitor_mode = "max"
save_top_k = 3
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = "auto"
resume_ckpt_path = None

net = ft_unetformer(num_classes=num_classes, decoder_channels=256)

loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0, 1.0
)
use_aux_loss = False

train_dataset = BiodiversityTiffTrainDataset(
    data_root="data/biodiversity_split/train_rep",
    img_dir="images",
    mask_dir="masks",
    img_suffix=".tif",
    mask_suffix=".png",
    mosaic_ratio=0.25,
    transform=train_aug_minority,  # <-- minority-aware cropping happens here
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

test_dataset = BiodiversityTiffTestDataset(
    data_root="data/biodiversity_split/test",
    img_dir="images",
    img_suffix=".tif",
)

# difficulty weights from Stage 2
repo_root = Path(__file__).resolve().parents[2]
sample_weights_path = repo_root / "artifacts" / "sample_weights.txt"

sample_weights = []
with open(sample_weights_path, "r") as f:
    for line in f:
        _, weight = line.strip().split("\t")
        sample_weights.append(float(weight))

print(f"Loaded {len(sample_weights)} sample weights from {sample_weights_path}")

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    sampler=sampler,       # sampler replaces shuffle
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

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
