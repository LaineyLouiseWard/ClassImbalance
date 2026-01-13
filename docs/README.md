# ClassImbalance — Biodiversity Land-Cover Segmentation with Staged Imbalance Handling

This repository implements a **fully reproducible, stage-wise training pipeline** for 6-class land-cover semantic segmentation on a Biodiversity TIFF dataset, with explicit handling of class imbalance and optional knowledge distillation from an OpenEarthMap (OEM) teacher model.

The methodology is structured as **incremental ablations**, where each stage adds exactly one component, enabling clean attribution of performance gains.

---

## Training Stages

| Stage | Description |
|------:|-------------|
| 1 | Baseline supervised training (Biodiversity only) |
| 2 | Minority sample replication (`train_rep`) |
| 3 | + Difficulty-weighted sampling |
| 4 | + Minority-aware cropping |
| 5A | OEM pretraining (Biodiversity + OEM combined) |
| 5B | Finetune on Biodiversity only |
| 6 | Knowledge distillation (OEM teacher → student) |

**Evaluation protocol**
- Validation: all stages (Biodiversity validation split)
- Test: final model only (Stage 6, Biodiversity test split)

---

## Repository Structure

PUT_TREE_HERE

---

## Data Assumptions

### Biodiversity Dataset

Proprietary land-cover dataset provided by ODOS Tech.
This dataset cannot be redistributed and is not included in this repository.

Raw data:
data/biodiversity_raw/
images/*.tif
masks/*.png

Generated splits:
data/biodiversity_split/
train/
train_rep/
val/
test/

- 6 semantic classes (labels 0–5)
- Class `0` corresponds to Background / void
- Test masks exist **only for evaluation**, never for training

---

### OpenEarthMap (OEM)

Publicly available OpenEarthMap dataset:
https://zenodo.org/records/7223446
https://open-earth-map.org/overview.html

Usage in this repository:
- OEM tiles are filtered to rural areas
- OEM labels are relabelled to the 6-class Biodiversity taxonomy
- Used **only** in Stage 5A (pretraining) and as a **teacher** in Stage 6
- OEM data is **never** used during validation or testing.

---

## Environment Setup

conda env create -f environment.yaml
conda activate ClassImbalance
pip install -r requirements.txt

---

## Reproducing Results

All experiments in this repository are fully reproducible via scripted pipelines.

The workflow is strictly:

**data → training → evaluation**

There are no manual steps.

---

## 1. Data Reproduction

All dataset preparation steps are defined in:

REPRODUCE_DATA.md

This script:
- splits the Biodiversity dataset into `train / val / test`
- creates `train_rep` via minority replication
- filters and relabels OpenEarthMap (OEM)
- prepares OEM teacher data
- creates the combined Biodiversity + OEM dataset for pretraining

All steps must be run **in order** before training.

---

## 2. Training Reproduction

All training stages are executed sequentially using:

REPRODUCE_TRAINING.md

Stages are:

1. Baseline (Biodiversity only)  
2. Replication  
3. Replication + difficulty-weighted sampling  
4. Replication + difficulty sampling + minority-aware cropping  
5A. OEM pretraining (combined Biodiversity + OEM)  
5B. Finetuning (Biodiversity only)  
6. Knowledge distillation (OEM teacher → student)

Each stage writes checkpoints to:

model_weights/biodiversity/<stage_name>/

No stage uses validation or test data for training.

---

## 3. Evaluation Reproduction

All evaluation steps are defined in:

REPRODUCE_EVALUATION.md

### Validation (all stages)

- All checkpoints from all stages are evaluated on:
  data/biodiversity_split/val
- Validation is used **only** for ablation comparison.

Outputs:
evaluation/evaluation_results/val/

### Test (final model only)

- Only the final Stage 6 model is evaluated on:
  data/biodiversity_split/test
- Test images are held out during training.
- Test masks are used **only** for evaluation.

Outputs:
evaluation/evaluation_results/test/

---

## Evaluation Outputs

For each evaluated checkpoint, the following files are produced:

- `metrics.json`
- `evaluation_report.txt`
- `confusion_matrix.png`
- `class_iou_scores.png`
- `class_f1_scores.png`

An aggregate summary across checkpoints is generated via:

evaluation/aggregate_metrics.py

This produces ranked comparisons of:
- mean IoU (excluding background)
- mean F1 (excluding background)
- overall accuracy

---

## Important Notes

- Validation is used for **model selection and ablation analysis only**
- Testing is performed **once**, on the final model
- OEM data is **never** used in validation or testing
- Models are trained on images only; masks are used solely for loss computation and evaluation
- Knowledge distillation uses a fixed, explicit class-mapping scheme

This structure enforces a clean separation between training, validation, and testing and prevents data leakage across stages.
