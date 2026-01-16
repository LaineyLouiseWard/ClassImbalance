# Data reproduction
## Assumptions:
- You are in the repo root
- conda env is activated
- raw datasets already exist in data/


## Step 0 — Optional: inspect biodiversity class distribution
python -m scripts.analyze_class_distribution \
  --out artifacts/train_augmentation_list.json \
  --overwrite


## Step 1 — Split Biodiversity into train / val / test
python -m scripts.split_biodiversity_dataset \
  --in-root data/biodiversity_raw \
  --out-root data/biodiversity_split \
  --mode symlink \
  --seed 42 \
  --overwrite


## Step 2 — Replicate minority-rich samples (creates train_rep)
python -m scripts.replicate_minority_samples \
  --data-root data/biodiversity_split/train \
  --out-root  data/biodiversity_split/train_rep \
  --augmentation-list artifacts/train_augmentation_list.json \
  --replications 1 \
  --overwrite


## Step 3 — Filter OEM to rural subset  (RAW → filtered)
python -m scripts.filter_oem_rural \
  --raw-root data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD \
  --out-root data/openearthmap_filtered \
  --threshold 50 \
  --mode symlink \
  --overwrite


## Step 4 — Prepare OEM teacher split (filtered → teacher)
python -m scripts.prepare_oem_teacher_data \
  --raw-root data/openearthmap_filtered \
  --out-root data/openearthmap_teacher \
  --val-frac 0.1 \
  --seed 42 \
  --mode symlink \
  --overwrite


## Step 5 — Relabel OEM into 6-class taxonomy (filtered → relabelled)
python -m scripts.relabel_oem_taxonomy \
  --in-root  data/openearthmap_filtered \
  --out-root data/openearthmap_relabelled \
  --mode symlink \
  --overwrite


## Step 5.5 — Second filtering pass (relabelled → relabelled_filtered)
python -m scripts.filter_oem_settlement_postmap \
  --in-root  data/openearthmap_relabelled \
  --out-root data/openearthmap_relabelled_filtered \
  --threshold 50.0 \
  --mode symlink \
  --overwrite


## Step 6 — Create combined Biodiversity + OEM training set
python -m scripts.create_biodiversity_oem_combined \
  --mode symlink \
  --overwrite