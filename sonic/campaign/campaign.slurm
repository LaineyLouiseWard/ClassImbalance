#!/bin/bash
# ============================================================================
# The 10-seed 2x2 factorial campaign, one array task per seed.
#
# Submit through sonic/campaign/submit_campaign.sh, never directly: the submitter is what
# refuses a dirty working tree, records the commit and fills in the log paths. Every cluster-
# specific value arrives through the environment, so nothing here names a person or an account.
#
#   SONIC_SCRATCH  campaign root on scratch; holds seed<N>/, results/ and logs/   (required)
#   SPLIT_TAG      which spatially blocked split to run                          (default f1)
#   BATCH_VARIANT  b2 = batch 2 / lr 3e-4 (default), b4 = batch 4 / lr 6e-4
#
# Per seed this trains the five configs that make the four factorial cells, scores all four on
# validation, Test A and Test B, and dumps the per-seed softmax that rho is computed from.
# ============================================================================
#SBATCH --job-name=lqc-campaign
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --array=0-9
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
# --output/--error/--account/--mail-user are supplied by submit_campaign.sh, because hard-coding
# them is how the previous script ended up carrying one person's home directory into the repo.
# No --mem: Sonic GPU nodes are RealMemory=1, so memory is not scheduled.

set -euo pipefail

: "${SONIC_SCRATCH:?SONIC_SCRATCH must be set (submit through sonic/campaign/submit_campaign.sh)}"
SPLIT_TAG="${SPLIT_TAG:-f1}"
BATCH_VARIANT="${BATCH_VARIANT:-b2}"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
DIR="$SONIC_SCRATCH/seed${SEED}"
RESULTS="$SONIC_SCRATCH/results"

echo "=== task ${SLURM_ARRAY_TASK_ID}  seed=${SEED}  split=${SPLIT_TAG}  dir=${DIR}"
echo "===  node=$(hostname)  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module load anaconda3/2025.12-1
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SONIC_SCRATCH/env"
set -u

cd "$DIR"

# Every path the pipeline touches is tagged. SPLIT_TAG must be exported, not merely set: the cell
# configs build each checkpoint path from it, and an unexported tag makes training write untagged
# checkpoints while evaluation looks for tagged ones -- which is how Test B was once scored by
# nothing at all, exit code 0.
export SPLIT_TAG
export BATCH_VARIANT
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
SPLIT_ROOT="data/split_${SPLIT_TAG}"
export BIO_SPLIT="$SPLIT_ROOT"
export BIO_OEM_COMBINED="data/oem_combined_${SPLIT_TAG}"
export SAMPLER_TSV="artifacts/sampler_weights_clsbal_${SPLIT_TAG}.tsv"
export TEACHER_CONFUSION_NPZ="artifacts/teacher_oem_gt_confusion_${SPLIT_TAG}.npz"
export AUG_LIST="artifacts/train_augmentation_list_${SPLIT_TAG}.json"

echo "[preflight] full leakage gate over the split AND every derived artefact"
# Up front, before twenty hours of GPU time: the split, the pre-training pool, the sampler weights,
# the augmentation list and the teacher confusion checked against each other. RUNBOOK stage B4b
# runs the same gate, but only after the first three cells have already trained.
python scripts/data_prep/assert_no_split_leakage.py \
  --split-root        "$SPLIT_ROOT" \
  --oem-root          "$BIO_OEM_COMBINED" \
  --sampler-tsv       "$SAMPLER_TSV" \
  --augmentation-list "$AUG_LIST" \
  --confusion-npz     "$TEACHER_CONFUSION_NPZ"

echo "[train+eval] RUNBOOK stages B1..C2 for seed ${SEED}"
# The runbook is the single definition of what a seed does, so the cluster cannot drift away from
# what runs locally. B1..C2 covers the five training configs, the sampler build, the full preflight
# gate again, and evaluation on validation, Test B and Test A.
SEED="$SEED" bash RUNBOOK.sh --from B1 --to C2

echo "[dump] per-seed softmax for rho"
# rho is the pre-registered primary statistic and these dumps are its only input. Nothing else in
# the pipeline produces them, so a campaign that skips this step cannot answer the paper's question.
# Test A and Test B go to one directory per cell: tile ids are unique across splits, and
# seed_disagreement.load_seed_stack resolves by id.
for CELL in stage1_baseline stage2b_oem_finetune stage_sampler_only stage3_clsbal; do
  for SPLIT in test external_test; do
    python scripts/analysis/dump_seed_softmax.py \
      --ckpt    "model_weights/biodiversity/${CELL}_${SPLIT_TAG}/${CELL}_${SPLIT_TAG}.ckpt" \
      --data-root "${SPLIT_ROOT}/${SPLIT}" \
      --out-dir "analysis/seed_softmax/${CELL}/seed${SEED}" \
      --device cuda
  done
done

echo "[collect] copying this seed's metrics to the shared drop"
# One flat directory per split, which is what aggregate_seeds.py reads through
# --results-dir / --test-results-dir / --external-results-dir.
for CELL in stage1_baseline stage2b_oem_finetune stage_sampler_only stage3_clsbal; do
  for PAIR in "val:val" "test:test" "external:external_${SPLIT_TAG}"; do
    DROP="${PAIR%%:*}"
    SRC="evaluation/evaluation_results/${PAIR#*:}/${CELL}_${SPLIT_TAG}/metrics.json"
    [ -f "$SRC" ] || { echo "COLLECT FAIL seed ${SEED}: missing ${SRC}"; exit 1; }
    mkdir -p "${RESULTS}/${DROP}"
    cp "$SRC" "${RESULTS}/${DROP}/seed${SEED}_${CELL}.json"
  done
done

echo "[check] the four cells scored the right checkpoint, without TTA"
python - "$SEED" "$SPLIT_TAG" <<'PY'
import json, sys
from pathlib import Path
seed, tag = sys.argv[1], sys.argv[2]
cells = ("stage1_baseline", "stage2b_oem_finetune", "stage_sampler_only", "stage3_clsbal")
for split in ("val", "test", f"external_{tag}"):
    for cell in cells:
        p = Path(f"evaluation/evaluation_results/{split}/{cell}_{tag}/metrics.json")
        m = json.loads(p.read_text())
        assert m["tta"] is False, f"{split}/{cell}: TTA must be off"
        got = Path(m["checkpoint"]).name
        assert got == f"{cell}_{tag}.ckpt", f"{split}/{cell}: scored {got}"
        i = m["per_class_iou"]
        print(f"[seed {seed}] {split:12s} {cell:22s} Settl={i['Settlement']*100:5.2f}  "
              f"Semi={i['Seminatural Grassland']*100:5.2f}  mIoU={m['mIoU_excluding_bg']*100:5.2f}")
PY

echo "=== seed ${SEED} complete  $(date) ==="
