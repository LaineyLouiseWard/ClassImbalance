# ============================================================
# Data reproduction (run in order)
# ============================================================
# Assumptions:
# - You are in the repo root
# - The Python environment is activated
# - Raw datasets already exist on disk
# - Paths below match your local setup
# ============================================================


# ------------------------------------------------------------
# Step 0 — Inspect raw class distributions (optional sanity)
# ------------------------------------------------------------
python scripts/analyze_class_distribution.py


# ------------------------------------------------------------
# Step 1 — Split Biodiversity dataset into train / val / test
# ------------------------------------------------------------
python scripts/split_biodiversity_dataset.py \
  --in-root data/biodiversity_raw \
  --out-root data/biodiversity_split


# ------------------------------------------------------------
# Step 2 — Replicate minority samples (train_rep split)
# ------------------------------------------------------------
python scripts/replicate_minority_samples.py \
  --in-root data/biodiversity_split/train \
  --out-root data/biodiversity_split/train_rep


# ------------------------------------------------------------
# Step 3 — Prepare OpenEarthMap (OEM) teacher dataset
#   - filter rural tiles
#   - keep raw OEM labels (with ignore_index=255)
# ------------------------------------------------------------
python scripts/filter_oem_rural.py \
  --raw-root data/openearthmap_raw \
  --out-root data/openearthmap_filtered

python scripts/prepare_oem_teacher_data.py \
  --raw-root data/openearthmap_filtered \
  --out-root data/openearthmap_teacher


# ------------------------------------------------------------
# Step 4 — Relabel OEM masks into 6-class taxonomy
# ------------------------------------------------------------
python scripts/relabel_oem_taxonomy.py \
  --in-root data/openearthmap_filtered \
  --out-root data/openearthmap_relabelled


# ------------------------------------------------------------
# Step 5 — Create combined Biodiversity + OEM training set
#   (used only for OEM pretraining)
# ------------------------------------------------------------
python scripts/create_biodiversity_oem_combined.py
