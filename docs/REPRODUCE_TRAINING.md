# ============================================================
# Training reproduction (run in order)
# ============================================================

# Stage 1 — Baseline (Biodiversity only)
python train/train_supervision.py config/biodiversity/stage1_baseline.py

# Stage 2 — Replication only
python train/train_supervision.py config/biodiversity/stage2_replication.py

# Generate difficulty weights (from Stage 2 model)
python scripts/analyze_hard_samples.py

# Stage 3 — Replication + difficulty-weighted sampling
python train/train_supervision.py config/biodiversity/stage3_hardsampling.py

# Stage 4 — Replication + difficulty-weighted sampling + minority-aware cropping
python train/train_supervision.py config/biodiversity/stage4_minoritycrop.py

# Train OEM teacher
python train/train_teacher.py config/teacher/unet_oem.py

# Export teacher checkpoint
python scripts/export_teacher_checkpoint.py \
  --ckpt model_weights/teacher.ckpt \
  --out pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth

# Stage 5A — OEM pretraining (student, biodiversity + OEM combined)
python train/train_supervision.py config/biodiversity/stage5_pretrain.py

# Stage 5B — Finetune after OEM pretraining (biodiversity only)
python train/train_supervision.py config/biodiversity/stage5_finetune.py

# Stage 6 — Final KD consolidation (student + teacher)
python train/train_kd.py config/biodiversity/stage6_kd.py
