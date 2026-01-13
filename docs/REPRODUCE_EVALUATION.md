# ------------------------------------------------------------
# Validation — all stages (Biodiversity val split only)
# ------------------------------------------------------------

python -m evaluation.compute_metrics \
  --split val \
  --base-dir model_weights \
  --data-root data/biodiversity_split/val \
  --out-dir evaluation/evaluation_results/val \
  --pattern "*.ckpt" \
  --device cuda \
  --ignore-index 0 \
  --num-workers 0


# ------------------------------------------------------------
# Test — final model only (Stage 6)
# ------------------------------------------------------------

python -m evaluation.compute_metrics \
  --split test \
  --base-dir model_weights/biodiversity/stage6_final_kd_ftunetformer \
  --data-root data/biodiversity_split/test \
  --out-dir evaluation/evaluation_results/test \
  --pattern "stage6_final_kd_ftunetformer.ckpt" \
  --device cuda \
  --ignore-index 0 \
  --num-workers 0


# ------------------------------------------------------------
# Aggregate validation results (ablation comparison)
# ------------------------------------------------------------

python -m evaluation.aggregate_metrics \
  --eval-root evaluation/evaluation_results/val \
  --out-file evaluation/evaluation_results/val/metrics_summary.txt


# ------------------------------------------------------------
# Aggregate test results (final model only; optional)
# ------------------------------------------------------------

python -m evaluation.aggregate_metrics \
  --eval-root evaluation/evaluation_results/test \
  --out-file evaluation/evaluation_results/test/metrics_summary.txt
