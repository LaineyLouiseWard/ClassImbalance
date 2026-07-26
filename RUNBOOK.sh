#!/bin/bash
set -euo pipefail
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Reduce CUDA fragmentation OOM on the 8 GB laptop GPU (pure stability; no result change).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail fast (with the fix) if the active 'python' lacks PyTorch — i.e. the conda env is not active.
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: active 'python' has no PyTorch — the label-quality-ceiling conda env is not activated."
  echo "  Fix:  conda activate label-quality-ceiling   (then re-run bash RUNBOOK.sh)"
  echo "  Or:   PATH=\"\$HOME/miniconda3/envs/label-quality-ceiling/bin:\$PATH\" bash RUNBOOK.sh"
  exit 1
fi

# ====================================================================
# Full reproducibility pipeline — 3-stage, no-replication ablation
#
#   Stage 1  baseline
#   Stage 2  OEM transfer        (2a pre-train on Bio+OEM -> 2b finetune on Bio)
#   Stage 3  clsbal class-balanced sampler  (FINAL shipped model)
#
# The teacher is built UPSTREAM of the student lineage (teacher -> confusion ->
# grounded OEM relabel -> student), because the OEM->student mappings are derived
# from the teacher's measured confusion.
#
# Usage:
#   bash RUNBOOK.sh                          # run everything from A0
#   bash RUNBOOK.sh --from B1                # resume from Stage 1 training onward
#   bash RUNBOOK.sh --from B1 --to C2        # run a stage WINDOW (Stage 1 .. test eval)
#   SEED=1 bash RUNBOOK.sh --from B1         # student lineage at seed 1 (teacher fixed at 42)
#   RESUME=1 bash RUNBOOK.sh --from B1       # resume training stages from their last.ckpt (no --force)
#
# Valid stages: A0 (taxonomy check), A1-A10 (data prep + teacher build),
#               B1-B5 (student training), C1-C4 (evaluation), D (analyses), E (figures)
#
# For the full 5-seed campaign as ONE resumable command, use ./run_campaign.sh.
#
# Overwrite flags:  --overwrite (data-prep)   --force (training/eval/export)
# Window flags:     --from <stage>  --to <stage>      Resume:  RESUME=1 (training stages)
# ====================================================================

# ---- Canonical paths ----
# --- Split selection -------------------------------------------------------
# The original random-by-tile split LEAKS: tiles are chipped on a 50% stride, so ~93% of each
# held-out tile's ground area also sat inside a training tile (notes/rebuild_2026-07/decisions/TILE_OVERLAP_LEAKAGE_2026-07-25.md).
# The campaign runs on spatially blocked splits built by scripts/data_prep/build_spatial_split.py.
# Three folds, f1 to f3, each cutting a different strip of the inland site. Select one with
#   SPLIT_TAG=f2 bash RUNBOOK.sh --from A2
# Every default below must name a directory build_spatial_split.py --materialise actually writes.
SPLIT_TAG="${SPLIT_TAG:-f1}"
SPLIT_ROOT="${SPLIT_ROOT:-data/split_${SPLIT_TAG}}"
OEM_COMBINED="${OEM_COMBINED:-data/oem_combined_${SPLIT_TAG}}"
SAMPLER_TSV="${SAMPLER_TSV:-artifacts/sampler_weights_clsbal_${SPLIT_TAG}.tsv}"
TEACHER_CONFUSION_NPZ="${TEACHER_CONFUSION_NPZ:-artifacts/teacher_oem_gt_confusion_${SPLIT_TAG}.npz}"
AUG_LIST="${AUG_LIST:-artifacts/train_augmentation_list_${SPLIT_TAG}.json}"
# Exported so the cell configs and the taxonomy guard pick the same split up.
# SPLIT_TAG must be exported: the cell configs build every checkpoint path from it, and without
# it they resolve untagged while C1b looks for tagged paths, so Test B printed "skip" for all
# four cells and the stage exited 0 -- the number the paper leads on produced by nothing.
export SPLIT_TAG BIO_SPLIT="$SPLIT_ROOT" BIO_OEM_COMBINED="$OEM_COMBINED" SAMPLER_TSV TEACHER_CONFUSION_NPZ AUG_LIST

BIO_RAW=data/biodiversity_raw
OEM_RAW=data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD
TEACHER_PTH=pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth

# ---- Parse --from / --to arguments ----
FROM_STAGE="A0"
TO_STAGE=""          # empty = run through the last stage
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM_STAGE="$2"; shift 2 ;;
    --to)   TO_STAGE="$2";   shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Ordered list of all stages
STAGES=(A0 A1 A1b A2 A3 A4 A5 A6 A7 A8 A9 A10 B1 B2 B3 B4 B4b B4c B5 N3 C1 C1b C2 C3 C4 D E)

# Validate --from value
valid=false
for s in "${STAGES[@]}"; do
  if [[ "$s" == "$FROM_STAGE" ]]; then valid=true; break; fi
done
if ! $valid; then
  echo "ERROR: Invalid stage '$FROM_STAGE'"
  echo "  Valid stages: ${STAGES[*]}"
  exit 1
fi

# Find index of FROM_STAGE
from_idx=0
for i in "${!STAGES[@]}"; do
  if [[ "${STAGES[$i]}" == "$FROM_STAGE" ]]; then from_idx=$i; break; fi
done

# Default TO = last stage; validate it; find its index
[ -z "$TO_STAGE" ] && TO_STAGE="${STAGES[$((${#STAGES[@]}-1))]}"
valid=false
for s in "${STAGES[@]}"; do
  if [[ "$s" == "$TO_STAGE" ]]; then valid=true; break; fi
done
if ! $valid; then
  echo "ERROR: Invalid stage '$TO_STAGE'"
  echo "  Valid stages: ${STAGES[*]}"
  exit 1
fi
to_idx=0
for i in "${!STAGES[@]}"; do
  if [[ "${STAGES[$i]}" == "$TO_STAGE" ]]; then to_idx=$i; break; fi
done
if [[ $from_idx -gt $to_idx ]]; then
  echo "ERROR: --from $FROM_STAGE is after --to $TO_STAGE"
  exit 1
fi

# Helper: should we run this stage? (within the [from, to] window)
run_stage() {
  local stage="$1"
  for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]}" == "$stage" ]]; then
      [[ $i -ge $from_idx && $i -le $to_idx ]] && return 0 || return 1
    fi
  done
  return 1
}

# Helper: check a directory has files (follows symlinks)
require_nonempty() {
  local dir="$1" stage="$2"
  if [ ! -d "$dir" ] || [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "ERROR: Required input '$dir' is missing or empty."
    echo "  Run stage $stage first (bash RUNBOOK.sh --from $stage)."
    exit 1
  fi
}

# Helper: check a file exists
require_file() {
  local f="$1" stage="$2"
  if [ ! -f "$f" ]; then
    echo "ERROR: Required file '$f' not found."
    echo "  Run stage $stage first (bash RUNBOOK.sh --from $stage)."
    exit 1
  fi
}

# ---- Resume mode -----------------------------------------------------------
# RESUME=1 drops --force from the heavy training stages so an interrupted run
# picks up from its per-stage last.ckpt (model+optimizer+scheduler+epoch, written
# every epoch via save_last=True). Completed stages no-op (last.ckpt already at
# max_epoch); the interrupted stage continues mid-training; later stages run fresh.
# Default (unset) keeps --force = always start each stage from scratch.
FORCE_TRAIN="--force"
if [ "${RESUME:-0}" = "1" ]; then
  FORCE_TRAIN=""
fi

echo "================================================================"
echo " PIPELINE -- stages $FROM_STAGE..$TO_STAGE  (SEED=${SEED:-42}, RESUME=${RESUME:-0})"
echo "================================================================"
echo ""

# ======================== A0. PRE-FLIGHT CHECK =======================

if run_stage A0; then
  echo "[A0] Verifying taxonomy consistency (class orders / grounded OEM->student mappings)"
  # Aborts the whole run (set -e) if any class order/index has drifted from geoseg/taxonomy.py,
  # or if the grounded pre-train map != argmax(teacher confusion) when the confusion artifact exists.
  PYTHONPATH=. python scripts/verify_taxonomy_consistency.py
fi

# ======================== A. DATA PREP + TEACHER BUILD ===============
# Order reflects the dependency teacher -> confusion -> grounded mappings -> OEM relabel -> student.

if run_stage A1; then
  # NOTE: this stage no longer decides the experimental split. It only unpacks the raw tiles into
  # the legacy three-directory layout, which downstream tooling reads as a flat POOL of all 2,143
  # tiles; the train/val/test assignment it writes is discarded and never trained on. The real
  # split is built in A1b. Splitting these overlap-chipped tiles at random is what leaked
  # (notes/rebuild_2026-07/decisions/TILE_OVERLAP_LEAKAGE_2026-07-25.md), so nothing may consume this assignment.
  echo "[A1] Unpacking Biodiversity tiles into the pool layout (assignment discarded; see A1b)"
  PYTHONPATH=. python scripts/data_prep/split_biodiversity_dataset.py \
    --in-root  "$BIO_RAW" \
    --out-root data/biodiversity_split \
    --seed 42 --mode copy --overwrite
fi

if run_stage A1b; then
  echo "[A1b] Building the spatially blocked split '$SPLIT_TAG' -> $SPLIT_ROOT"
  # Replays a committed manifest when one exists, so every machine materialises the identical
  # split; otherwise derives it. Both paths re-read the GeoTIFF geometry and abort on any
  # cross-split footprint overlap.
  MANIFEST="artifacts/spatial_split_manifest_${SPLIT_TAG}.json"
  # Replay only. Deriving here would be worse than failing: --three-region defaults to unset, so a
  # bare run falls through to the older greedy proposer and silently materialises a DIFFERENT design
  # under the same name. The three shipped manifests are committed; a missing one means the tag is
  # wrong, not that a new split should be invented.
  if [ ! -f "$MANIFEST" ]; then
    echo "[A1b] FATAL: no manifest at $MANIFEST" >&2
    echo "  SPLIT_TAG=$SPLIT_TAG is not one of the shipped folds. Available:" >&2
    ls artifacts/spatial_split_manifest_*.json 2>/dev/null | sed 's/^/    /' >&2
    echo "  To derive a genuinely new fold, name the design explicitly, e.g." >&2
    echo "    python scripts/data_prep/build_spatial_split.py --three-region biodiversity \\" >&2
    echo "      --external-sites ireland1 ireland2 --buffer-test-m 650 --buffer-val-m 256 \\" >&2
    echo "      --distinct-from artifacts/spatial_split_manifest_f*.json --out $MANIFEST" >&2
    exit 1
  fi
  PYTHONPATH=. python scripts/data_prep/build_spatial_split.py \
    --from-manifest "$MANIFEST" --materialise --mode symlink --out-root "$SPLIT_ROOT"
  require_nonempty "$SPLIT_ROOT"/train/masks A1b
  # Split-only at this point: the sampler, augmentation list and teacher confusion do not exist yet.
  # Stage B4b re-runs the gate over all of them once they do.
  echo "[A1b] Leakage preflight (split geometry only)"
  PYTHONPATH=. python scripts/data_prep/assert_no_split_leakage.py --split-root "$SPLIT_ROOT"
fi

if run_stage A2; then
  require_nonempty "$SPLIT_ROOT"/train/masks A1
  echo "[A2] Identifying minority-rich tiles (for the D-stage sampler-uplift analysis)"
  PYTHONPATH=. python scripts/data_prep/analyze_class_distribution.py \
    --data-root "$SPLIT_ROOT"/train \
    --out       "$AUG_LIST" \
    --overwrite
fi

if run_stage A3; then
  echo "[A3] Filtering OEM (pre-mapping, rural tiles only)"
  PYTHONPATH=. python scripts/data_prep/filter_oem_rural.py \
    --raw-root "$OEM_RAW" \
    --out-root data/openearthmap_filtered \
    --overwrite
fi

if run_stage A4; then
  require_nonempty "$OEM_RAW" A3
  echo "[A4] Preparing OEM teacher training split (FULL OEM, native 9-class taxonomy)"
  # Teacher trains on the FULL OEM (~3,500 tiles), NOT the rural-filtered subset:
  # the rural filter strips settlement-rich tiles, which would weaken the very
  # minority-class signal KD injects. Native labels 0..8 are preserved.
  PYTHONPATH=. python scripts/data_prep/prepare_oem_teacher_data.py \
    --raw-root "$OEM_RAW" \
    --out-root data/openearthmap_teacher \
    --official-split \
    --overwrite
fi

if run_stage A5; then
  require_nonempty data/openearthmap_teacher/train/images A4
  echo "[A5] Training OEM teacher (seed fixed at 42 — build-once, seed-invariant artifact)"
  # The teacher is held FIXED across the seed campaign (like the data), so it is NOT reseeded.
  PYTHONPATH=. python -m train.train_teacher \
    -c config/teacher/unet_oem.py --force
fi

if run_stage A6; then
  require_file model_weights/teacher/teacher.ckpt A5
  echo "[A6] Exporting teacher checkpoint + verifying native-A output channels"
  PYTHONPATH=. python -m scripts.data_prep.export_teacher_checkpoint \
    --ckpt model_weights/teacher/teacher.ckpt \
    --out  "$TEACHER_PTH" \
    --force
  # Aborts the run (set -e) if the teacher is not native-A — e.g. a stale 6-class checkpoint.
  PYTHONPATH=. python scripts/verify_teacher_channels.py \
    --ckpt "$TEACHER_PTH" \
    --data-root data/openearthmap_teacher/val
fi

if run_stage A7; then
  require_file "$TEACHER_PTH" A6
  require_nonempty "$SPLIT_ROOT"/train/masks A1
  echo "[A7] Measuring teacher->GT confusion on the training set (grounds the OEM->student mappings)"
  # Measured on THIS split's training set and written per split, because the matrix is fitted on
  # training labels: the committed base matrix was measured over the old random split, whose
  # training set held tiles that are now test. Its argmax IS taxonomy.OEM_TO_STUDENT_PRETRAIN,
  # which relabels the OEM tiles for pre-training, so a change here reaches the transfer arm.
  PYTHONPATH=. python scripts/analysis/teacher_oem_to_gt_confusion.py \
    --data-root "$SPLIT_ROOT"/train \
    --out       "$TEACHER_CONFUSION_NPZ"
  # Re-run the taxonomy guard against the matrix just written, not the stale committed one.
  TEACHER_CONFUSION_NPZ="$TEACHER_CONFUSION_NPZ" \
    PYTHONPATH=. python scripts/verify_taxonomy_consistency.py
fi

if run_stage A8; then
  require_nonempty data/openearthmap_filtered/masks A3
  echo "[A8] Relabelling OEM to the 6-class taxonomy (grounded argmax mapping)"
  PYTHONPATH=. python scripts/data_prep/relabel_oem_taxonomy.py \
    --in-root  data/openearthmap_filtered \
    --out-root data/openearthmap_relabelled \
    --overwrite
fi

if run_stage A9; then
  require_nonempty data/openearthmap_relabelled/masks A8
  echo "[A9] Filtering OEM (post-mapping, settlement-dominant removal)"
  PYTHONPATH=. python scripts/data_prep/filter_oem_settlement_postmap.py \
    --in-root  data/openearthmap_relabelled \
    --out-root data/openearthmap_relabelled_filtered \
    --overwrite
fi

if run_stage A10; then
  require_nonempty "$SPLIT_ROOT"/train/images A1
  require_nonempty data/openearthmap_relabelled_filtered/masks A9
  echo "[A10] Creating combined Biodiversity + OEM dataset (Stage 2a pre-training pool)"
  PYTHONPATH=. python scripts/data_prep/create_biodiversity_oem_combined.py \
    --bio-root "$SPLIT_ROOT" \
    --oem-root data/openearthmap_relabelled_filtered \
    --out-root "$OEM_COMBINED" \
    --overwrite
fi

# ======================== B. STUDENT LINEAGE =========================
# Seed-varying. Honours $SEED (default 42) in train_supervision.

if run_stage B1; then
  require_nonempty "$SPLIT_ROOT"/train/images A1
  echo "[B1] Stage 1: Baseline"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage1_baseline.py $FORCE_TRAIN
fi

if run_stage B2; then
  require_nonempty "$OEM_COMBINED"/train/images A10
  echo "[B2] Stage 2a: OEM pre-training (combined Bio + OEM)"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage2a_oem_pretrain.py $FORCE_TRAIN
fi

if run_stage B3; then
  require_file model_weights/biodiversity/stage2a_oem_pretrain_${SPLIT_TAG}/stage2a_oem_pretrain_${SPLIT_TAG}.ckpt B2
  echo "[B3] Stage 2b: OEM-transfer finetune on Biodiversity (init from 2a)"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage2b_oem_finetune.py $FORCE_TRAIN
fi

if run_stage B4; then
  require_nonempty "$SPLIT_ROOT"/train/images A1
  echo "[B4] Building class-balanced (clsbal) sampler weights"
  PYTHONPATH=. python scripts/data_prep/build_clsbal_sampler.py \
    --data_root "$SPLIT_ROOT"/train \
    --out       "$SAMPLER_TSV" \
    --aug-list  "$AUG_LIST" \
    --q 1.0 --settlement_target 1.27 \
    --force
fi

# The last gate before any weights are trained. A1b could only check the split geometry, because
# every derived artefact is built after it. This is where the split, the OEM pre-training pool, the
# sampler weights, the augmentation list and the teacher confusion are checked against each other --
# the combination that actually determines what the model sees.
if run_stage B4b; then
  echo "[B4b] Full leakage preflight over the split AND every derived artefact"
  PYTHONPATH=. python scripts/data_prep/assert_no_split_leakage.py \
    --split-root    "$SPLIT_ROOT" \
    --oem-root      "$OEM_COMBINED" \
    --sampler-tsv   "$SAMPLER_TSV" \
    --augmentation-list "$AUG_LIST" \
    --confusion-npz "$TEACHER_CONFUSION_NPZ"
fi

if run_stage B4c; then
  require_file "$SAMPLER_TSV" B4
  echo "[B4c] Sampler-only cell (4th factorial cell; was in NO runbook stage before 2026-07-26,"
  echo "      so an end-to-end run trained three of the four cells and the factorial could not be assembled)"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage_sampler_only.py $FORCE_TRAIN
fi

if run_stage B5; then
  require_file model_weights/biodiversity/stage2b_oem_finetune_${SPLIT_TAG}/stage2b_oem_finetune_${SPLIT_TAG}.ckpt B3
  require_file "$SAMPLER_TSV" B4
  echo "[B5] Stage 3: Class-balanced (clsbal) sampling — FINAL shipped model"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage3_clsbal.py $FORCE_TRAIN
fi

# (Former N3 sampler null control removed — the 2x2 factorial estimates the sampler effect
#  directly via the paired (sampler-only - baseline) / (full - transfer-only) contrasts, so the
#  separate uniform-draw null is redundant. The retired A0-era config is archived under
#  config/biodiversity/_archive/stage3null_nosampler.py.)

# ======================== C. EVALUATION ==============================
# The four factorial cells, in the order the paper reports them. Every evaluation stage names its
# checkpoints from this list rather than rglobbing a directory: `--base-dir` picks up `last.ckpt`
# alongside `<cell>_<tag>.ckpt` and evaluates BOTH into the same output directory, so which one
# survives in metrics.json is decided by filename sort order rather than by design.
CELLS=(stage1_baseline stage2b_oem_finetune stage_sampler_only stage3_clsbal)

# Path of one cell's selected checkpoint. Every checkpoint path is tagged; an untagged one is the
# withdrawn campaign's, still on disk under _archive/stale_checkpoints_pre_rebuild/.
cell_ckpt() { echo "model_weights/biodiversity/${1}_${SPLIT_TAG}/${1}_${SPLIT_TAG}.ckpt"; }

if run_stage C1; then
  require_nonempty "$SPLIT_ROOT"/val/images A1
  echo "[C1] Evaluating validation set (four factorial cells; checkpoint selection split)"
  for CELL in "${CELLS[@]}"; do
    CKPT="$(cell_ckpt "$CELL")"
    require_file "$CKPT" B5
    PYTHONPATH=. python evaluation/compute_metrics.py \
      --checkpoints "$CKPT" \
      --split val \
      --data-root "$SPLIT_ROOT"/val \
      --out-dir evaluation/evaluation_results/val \
      --force
  done
fi

if run_stage C1b; then
  echo "[C1b] Test B — the held-out upland sites. This is the generalisation number the paper leads on,"
  echo "      and until 2026-07-26 no runnable path scored it at all."
  for CELL in "${CELLS[@]}"; do
    CKPT="$(cell_ckpt "$CELL")"
    # A hard failure, not a skip. This stage silently succeeded over four missing checkpoints.
    require_file "$CKPT" B5
    # NO per-cell subdirectory: compute_metrics.py already names the run directory after the
    # checkpoint's parent, so adding ${CELL} here nested the metrics one level deeper than every
    # reader looks (external_f1/<cell>/<cell>_f1/metrics.json), and Test B aggregated to nothing.
    PYTHONPATH=. python evaluation/compute_metrics.py \
      --checkpoints "$CKPT" \
      --data-root  "$SPLIT_ROOT"/external_test \
      --split test \
      --out-dir    evaluation/evaluation_results/external_${SPLIT_TAG} \
      --ignore-index 0 \
      --force
  done
  echo "[C1b] Per-class support for Test B (a share is not a support; classes under 5 independent"
  echo "      950 m blocks are reported as unestimable, never as an estimate)"
  PYTHONPATH=. python scripts/analysis/report_class_support.py --split-root "$SPLIT_ROOT"
fi

if run_stage C2; then
  require_nonempty "$SPLIT_ROOT"/test/images A1
  echo "[C2] Evaluating held-out Test A — ALL FOUR factorial cells"
  # Two of four were scored here until 2026-07-26, so the transfer, sampler and interaction
  # contrasts could only be formed on validation: the split every checkpoint is selected on.
  for CELL in "${CELLS[@]}"; do
    CKPT="$(cell_ckpt "$CELL")"
    require_file "$CKPT" B5
    PYTHONPATH=. python evaluation/compute_metrics.py \
      --checkpoints "$CKPT" \
      --split test \
      --data-root "$SPLIT_ROOT"/test \
      --out-dir evaluation/evaluation_results/test \
      --force
  done
  # Final shipped model (Stage 3 clsbal): ALSO evaluate the test split WITH TTA so the paper
  # can report both. TTA = multi-scale + H+V flip, softmax-averaged (GeoSeg 'd4' minus 1.5 scale;
  # bilinear interp here vs GeoSeg bicubic — harmless, worth a one-line methods note). Written to
  # a SEPARATE out-dir so the no-TTA metrics.json above (the one C4 reads) is preserved alongside it.
  # NB: the val/ablation eval (C1) deliberately stays WITHOUT TTA — TTA lifts all stages ~equally
  # and must not be silently turned on there.
  PYTHONPATH=. python evaluation/compute_metrics.py \
    --checkpoints "$(cell_ckpt stage3_clsbal)" \
    --split test \
    --data-root "$SPLIT_ROOT"/test \
    --out-dir evaluation/evaluation_results/test_tta \
    --tta --tta-flips hv --tta-scales 0.75,1.0,1.25 \
    --force
fi

if run_stage C3; then
  require_nonempty evaluation/evaluation_results/val C1
  echo "[C3] Aggregating validation summary"
  PYTHONPATH=. python evaluation/aggregate_metrics.py \
    --eval-root evaluation/evaluation_results/val \
    --out-file  evaluation/evaluation_results/val/metrics_summary.txt
fi

if run_stage C4; then
  require_nonempty evaluation/evaluation_results/test C2
  echo "[C4] Exporting test-set LaTeX table"
  python evaluation/export_final_test_table.py
fi

# ======================== D. ANALYSES ================================

if run_stage D; then
  require_nonempty evaluation/evaluation_results/val C1
  require_nonempty model_weights/biodiversity B5   # fail fast before running a1-a6
  echo "[D] Running supplementary analyses (A1-A6)"
  PYTHONPATH=. python scripts/analysis/a1_minority_recall.py
  PYTHONPATH=. python scripts/analysis/a2_symmetric_confusion.py
  PYTHONPATH=. python scripts/analysis/a3_sampler_weight_uplift.py
  PYTHONPATH=. python scripts/analysis/a4_val_test_gap.py
  PYTHONPATH=. python scripts/analysis/a5_majority_stability.py
  PYTHONPATH=. python scripts/analysis/a6_weight_gini.py
  echo "[D] Bootstrap confidence intervals (per-tile resampling; prerequisite for Figure 10)"
  # --force re-runs inference instead of reusing stale analysis/per_tile_cms/*.npz from a prior run.
  PYTHONPATH=. python scripts/analysis/bootstrap_metrics.py --device cuda --force
fi

# ======================== E. FIGURES =================================

if run_stage E; then
  echo "[E] Generating all paper figures"
  python scripts/figures/build_all_figures.py --device cuda
fi

echo "================================================================"
echo " DONE -- pipeline finished"
echo "================================================================"
