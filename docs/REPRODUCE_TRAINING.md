# Training Reproduction
## Assumptions:
- You are in the repo root
- conda env is activated
- raw datasets already exist in data/
- artifacts/ (train_augmentation_list.json, sample_weights.txt) are writable


## Stage 1 — Baseline (Biodiversity only)
python -m train.train_supervision -c config/biodiversity/stage1_baseline.py


## Stage 2 — Replication only (uses train_rep)
python -m train.train_supervision -c config/biodiversity/stage2_replication.py


## Step — Generate difficulty weights (required for Stage 3/4/5B)
python -m scripts.analyze_hard_samples


## Stage 3 — Replication + difficulty-weighted sampling
python -m train.train_supervision -c config/biodiversity/stage3_hardsampling.py


## Stage 4 — Replication + difficulty-weighted sampling + minority-aware cropping
python -m train.train_supervision -c config/biodiversity/stage4_minoritycrop.py


## Step — Train OEM teacher (EfficientNet-B4 U-Net on OEM native taxonomy)
python -m train.train_teacher -c config/teacher/unet_oem.py


## Step — Export teacher checkpoint for student pretraining
python -m scripts.export_teacher_checkpoint \
  --ckpt model_weights/teacher/teacher.ckpt \
  --out  pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth


## Stage 5A — OEM pretraining (student, Biodiversity + OEM combined)
python -m train.train_supervision -c config/biodiversity/stage5_pretrain.py


## Stage 5B — Finetune after OEM pretraining (Biodiversity only)
python -m train.train_supervision -c config/biodiversity/stage5_finetune.py


## Stage 6 — Final KD consolidation (student + teacher)
python -m train.train_kd -c config/biodiversity/stage6_kd.py
