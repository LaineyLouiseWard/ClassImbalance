#!/bin/bash
set -euo pipefail
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Training Reproduction
## Assumptions:
# - You are in the repo root
# - conda env is activated
# - raw datasets already exist in data/
# - artifacts/ (train_augmentation_list.json, sample_weights.txt) are writable

# Stage 1 — Baseline (Biodiversity only)
#echo "Running Stage 1: Baseline"
#python -m train.train_supervision -c config/biodiversity/stage1_baseline.py

# Stage 2 — Replication only (uses train_rep)
#echo "Running Stage 2: Replication"
#python -m train.train_supervision -c config/biodiversity/stage2_replication.py

# Step — Generate difficulty weights (required for Stage 3/4/5B)
echo "Running Step: Generate difficulty weights"
python -m scripts.analyze_hard_samples

# Stage 3 — Replication + difficulty-weighted sampling
echo "Running Stage 3: Replication + Difficulty-Weighted Sampling"
python -m train.train_supervision -c config/biodiversity/stage3_hardsampling.py

# Stage 4 — Replication + difficulty-weighted sampling + minority-aware cropping
echo "Running Stage 4: Replication + Difficulty-Weighted Sampling + Minority-Aware Cropping"
python -m train.train_supervision -c config/biodiversity/stage4_minoritycrop.py

# Step — Train OEM teacher (EfficientNet-B4 U-Net on OEM native taxonomy)
echo "Running Step: Train OEM Teacher"
python -m train.train_teacher -c config/teacher/unet_oem.py

# Step — Export teacher checkpoint for student pretraining
echo "Running Step: Export Teacher Checkpoint"
python -m scripts.export_teacher_checkpoint \
  --ckpt model_weights/teacher/teacher.ckpt \
  --out pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth

# Stage 5A — OEM pretraining (student, Biodiversity + OEM combined)
echo "Running Stage 5A: OEM Pretraining"
python -m train.train_supervision -c config/biodiversity/stage5_pretrain.py

# Stage 5B — Finetune after OEM pretraining (Biodiversity only)
echo "Running Stage 5B: Finetune After OEM Pretraining"
python -m train.train_supervision -c config/biodiversity/stage5_finetune.py

# Stage 6 — Final KD consolidation (student + teacher)
echo "Running Stage 6: Final KD Consolidation"
python -m train.train_kd -c config/biodiversity/stage6_kd.py
