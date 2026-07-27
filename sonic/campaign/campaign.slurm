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

# What is this task actually running? The submitter checks ITS OWN checkout is clean, but every
# array task runs from a scratch copy staged separately, so forty runs could be attributed to a
# commit none of them contain. If the staged copy is a git checkout, it must match; if it is not
# (rsync, tarball), say so rather than implying a provenance that was never verified.
if git rev-parse --git-dir >/dev/null 2>&1; then
  HERE="$(git rev-parse HEAD)"
  if [ -n "${CAMPAIGN_COMMIT:-}" ] && [ "$HERE" != "$CAMPAIGN_COMMIT" ]; then
    echo "ABORT seed ${SEED}: $DIR is at $HERE, the campaign was submitted from $CAMPAIGN_COMMIT" >&2
    exit 1
  fi
  if [ -n "$(git status --porcelain)" ]; then
    echo "ABORT seed ${SEED}: $DIR has uncommitted changes; this run would not match the others" >&2
    exit 1
  fi
  echo "[provenance] seed ${SEED} running ${HERE} (clean)"
else
  echo "[provenance] seed ${SEED}: $DIR is not a git checkout, commit not verifiable"
fi

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

echo "[preflight] taxonomy consistency (class orders, grounded OEM->student mapping)"
# RUNBOOK stage A0 runs this, but the campaign window is B4..C5, so nothing in a seed's own run
# checks it. If the grounded pre-train map ever stops matching argmax(teacher confusion), stage 2a
# relabels OpenEarthMap against a mapping the artefact no longer supports and every transfer-arm run
# is quietly wrong. Cheap here, unavailable later.
python scripts/verify_taxonomy_consistency.py

echo "[preflight] full leakage gate over the split AND every derived artefact"
# Up front, before twenty hours of GPU time: the split, the pre-training pool, the sampler weights,
# the augmentation list and the teacher confusion checked against each other. RUNBOOK stage B4b runs
# the same gate at the start of the window below; this one fails before the sampler is even rebuilt.
python scripts/data_prep/assert_no_split_leakage.py \
  --split-root        "$SPLIT_ROOT" \
  --oem-root          "$BIO_OEM_COMBINED" \
  --sampler-tsv       "$SAMPLER_TSV" \
  --augmentation-list "$AUG_LIST" \
  --confusion-npz     "$TEACHER_CONFUSION_NPZ"

echo "[train+eval] RUNBOOK stages B4..C5 for seed ${SEED}"
# The runbook is the single definition of what a seed does, so the cluster cannot drift away from
# what runs locally. B4..C5 covers the sampler build, the full preflight gate, the five training
# configs, evaluation on validation, Test B and Test A, and the per-seed softmax dumps the boundary
# evidence is computed from. The dump used to be duplicated here; one definition, in the runbook.
SEED="$SEED" bash RUNBOOK.sh --from B4 --to C5

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

echo "[collect] copying this seed's softmax dumps to the shared drop"
# The metrics alone are not enough. rho and the trimap exclusion curve -- the primary evidence for
# the boundary claim -- are computed from the per-seed softmax, and stage C5 writes it inside THIS
# seed's tree. Nine of the ten seeds run in trees nobody afterwards looks in, so without this the
# dumps stay on scratch until it is purged and the evidence costs a re-run of inference over all
# forty checkpoints. Flat <root>/<cell>/seed<N> layout, which is what seed_disagreement.seed_dir
# resolves.
for CELL in stage1_baseline stage2b_oem_finetune stage_sampler_only stage3_clsbal; do
  SRC="analysis/seed_softmax/${CELL}/seed${SEED}"
  [ -d "$SRC" ] || { echo "COLLECT FAIL seed ${SEED}: missing softmax dir ${SRC}"; exit 1; }
  N="$(find "$SRC" -name '*.npy' | wc -l)"
  [ "$N" -gt 0 ] || { echo "COLLECT FAIL seed ${SEED}: ${SRC} holds no .npy dumps"; exit 1; }
  mkdir -p "${RESULTS}/softmax/${CELL}"
  cp -r "$SRC" "${RESULTS}/softmax/${CELL}/"
  echo "  ${CELL}: ${N} dumps -> ${RESULTS}/softmax/${CELL}/seed${SEED}"
done

echo "[collect] copying this seed's CHECKPOINTS to the shared drop"
# Without this the campaign is unrepeatable. Until 2026-07-27 the collect step took metrics.json and
# the softmax dumps and nothing else, so all fifty trained models stayed in each seed's scratch
# worktree and died with the purge. That forecloses, permanently and silently: re-evaluating on a
# corrected split, any new figure needing predictions, any analysis needing inference, and the
# step-matched control D12 names as its own reversal condition -- which branches from the per-seed
# stage1_baseline checkpoint and is cheap ONLY while that checkpoint exists.
#
# The four cell checkpoints are the default because they cover both re-evaluation and that control.
# stage2a is opt-in: it is needed only to train a NEW variant from the pre-train, not to reproduce
# anything already reported. Set COLLECT_PRETRAIN=1 to include it.
CKPT_CELLS=(stage1_baseline stage2b_oem_finetune stage_sampler_only stage3_clsbal)
[ "${COLLECT_PRETRAIN:-0}" = "1" ] && CKPT_CELLS+=(stage2a_oem_pretrain)

# Space is checked BEFORE the first copy. A half-copied 1.5 GB checkpoint that fails at seed 7 is
# worse than a clean refusal at seed 0: it looks like a collection that happened.
NEED_KB=0
for CELL in "${CKPT_CELLS[@]}"; do
  SRC="model_weights/biodiversity/${CELL}_${SPLIT_TAG}/${CELL}_${SPLIT_TAG}.ckpt"
  [ -f "$SRC" ] || { echo "COLLECT FAIL seed ${SEED}: missing checkpoint ${SRC}"; exit 1; }
  NEED_KB=$(( NEED_KB + $(du -k "$SRC" | cut -f1) ))
done
mkdir -p "${RESULTS}/checkpoints"
AVAIL_KB="$(df -Pk "${RESULTS}" | awk 'NR==2 {print $4}')"
if [ "$AVAIL_KB" -lt "$(( NEED_KB * 2 ))" ]; then
  echo "COLLECT FAIL seed ${SEED}: need $(( NEED_KB/1024 )) MB for checkpoints (x2 margin), only $(( AVAIL_KB/1024 )) MB free on ${RESULTS}." >&2
  echo "  Free space, or set COLLECT_PRETRAIN=0, or drop cells from CKPT_CELLS -- but understand that" >&2
  echo "  an uncollected checkpoint is gone for good once scratch is purged." >&2
  exit 1
fi
for CELL in "${CKPT_CELLS[@]}"; do
  SRC="model_weights/biodiversity/${CELL}_${SPLIT_TAG}/${CELL}_${SPLIT_TAG}.ckpt"
  mkdir -p "${RESULTS}/checkpoints/${CELL}"
  cp "$SRC" "${RESULTS}/checkpoints/${CELL}/seed${SEED}.ckpt"
  echo "  ${CELL}: $(du -h "$SRC" | cut -f1) -> ${RESULTS}/checkpoints/${CELL}/seed${SEED}.ckpt"
done

echo "[check] the four cells scored the right checkpoint, without TTA"
python - "$SEED" "$SPLIT_TAG" <<'PY'
import json, sys
from pathlib import Path
seed, tag = sys.argv[1], sys.argv[2]
cells = ("stage1_baseline", "stage2b_oem_finetune", "stage_sampler_only", "stage3_clsbal")
# (results directory, the split directory data_root must end in). They differ for Test B: the
# results land in external_<tag> while the data lives in data/split_<tag>/external_test.
for split, data_dir in (("val", "val"), ("test", "test"), (f"external_{tag}", "external_test")):
    for cell in cells:
        p = Path(f"evaluation/evaluation_results/{split}/{cell}_{tag}/metrics.json")
        m = json.loads(p.read_text())
        # Check the DATA, not only the model. Validating `checkpoint` and `tta` alone is what let a
        # withdrawn metrics.json -- one whose own fields read split: val, data_root:
        # .../biodiversity_split/val -- pass for a held-out test result at exit 0. Both fields were
        # in the file, unread. Mirrors scripts/analysis/utils.assert_metrics_provenance.
        assert m["tta"] is False, f"{split}/{cell}: TTA must be off"
        got = Path(m["checkpoint"]).name
        assert got == f"{cell}_{tag}.ckpt", f"{split}/{cell}: scored {got}"
        dr = Path(m["data_root"])
        assert dr.name == data_dir, f"{split}/{cell}: computed over {dr}, not a '{data_dir}' split"
        assert dr.parent.name == f"split_{tag}", \
            f"{split}/{cell}: computed over {dr}, not under split_{tag}"
        i = m["per_class_iou"]
        print(f"[seed {seed}] {split:12s} {cell:22s} Settl={i['Settlement']*100:5.2f}  "
              f"Semi={i['Seminatural Grassland']*100:5.2f}  mIoU={m['mIoU_excluding_bg']*100:5.2f}")
PY

echo "=== seed ${SEED} complete  $(date) ==="
