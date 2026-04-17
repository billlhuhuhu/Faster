#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

# Eval-only rerun for the 4 newly added baselines.
BASELINE_METHODS="${BASELINE_METHODS:-presel visa dataprophet dynamic_pruning}"
ABS_BUDGETS="${ABS_BUDGETS:-100 200 500}"
RATIOS="${RATIOS:-0.01 0.02 0.03}"
BASELINE_SEEDS="${BASELINE_SEEDS:-0}"
BASELINE_DEVICE="${BASELINE_DEVICE:-cuda:0}"
BASELINE_CONFIG="${BASELINE_CONFIG:-baselines/configs/main_aligned_flickr_nfnet_bert.yaml}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-artifacts/baselines}"
BASELINE_DATASET="${BASELINE_DATASET:-flickr}"
BASELINE_IMAGE_ENCODER="${BASELINE_IMAGE_ENCODER:-nfnet}"
BASELINE_TEXT_ENCODER="${BASELINE_TEXT_ENCODER:-bert}"
BASELINE_FEATURE_SOURCE="${BASELINE_FEATURE_SOURCE:-artifacts/feature_cache}"
BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-data/flickr30k}"
BASELINE_ANN_ROOT="${BASELINE_ANN_ROOT:-data/Flickr30k_ann}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
BASELINE_BATCH_TRAIN="${BASELINE_BATCH_TRAIN:-64}"
BASELINE_BATCH_TEST="${BASELINE_BATCH_TEST:-128}"
BASELINE_TEXT_BATCH="${BASELINE_TEXT_BATCH:-1024}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-4}"
NEW4_CLEAR_OLD_METRICS="${NEW4_CLEAR_OLD_METRICS:-1}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

echo "[new4-eval-only] methods=${BASELINE_METHODS}"
echo "[new4-eval-only] abs_budgets=${ABS_BUDGETS}"
echo "[new4-eval-only] ratios=${RATIOS}"
echo "[new4-eval-only] seeds=${BASELINE_SEEDS}"
echo "[new4-eval-only] output_root=${BASELINE_OUTPUT_ROOT}"
echo "[new4-eval-only] device=${BASELINE_DEVICE}"

if [[ "${NEW4_CLEAR_OLD_METRICS}" == "1" ]]; then
  echo "[new4-eval-only] removing old downstream_metrics.json for target methods..."
  find "${BASELINE_OUTPUT_ROOT}" -type f \
    \( -path "*/presel/*/downstream_metrics.json" -o \
       -path "*/visa/*/downstream_metrics.json" -o \
       -path "*/dataprophet/*/downstream_metrics.json" -o \
       -path "*/dynamic_pruning/*/downstream_metrics.json" \) -delete || true
fi

# 1) Absolute budget eval-only (no reselection).
BASELINE_BUDGETS="${ABS_BUDGETS}" \
BASELINE_METHODS="${BASELINE_METHODS}" \
BASELINE_SEEDS="${BASELINE_SEEDS}" \
BASELINE_DEVICE="${BASELINE_DEVICE}" \
BASELINE_ROOT="${BASELINE_OUTPUT_ROOT}" \
BASELINE_CONFIG="${BASELINE_CONFIG}" \
bash baselines/scripts/run_main_aligned_eval.sh

# 2) Ratio eval-only (no reselection).
for seed in ${BASELINE_SEEDS}; do
  for ratio in ${RATIOS}; do
    ratio_tag="$(python - <<PY
r = float("${ratio}")
print(f"ratio_{int(round(r*100)):02d}")
PY
)"
    for method in ${BASELINE_METHODS}; do
      run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}/${ratio_tag}/${method}/seed_${seed}"
      selected_path="${run_dir}/selected_indices.json"
      if [[ ! -f "${selected_path}" ]]; then
        echo "[new4-eval-only][skip] selection missing: ${selected_path}"
        continue
      fi
      echo "[new4-eval-only][ratio-eval] method=${method} ratio=${ratio} seed=${seed}"
      python -m baselines.runners.evaluate_baseline_subsets \
        --baseline_result_dir "${run_dir}" \
        --dataset_name "${BASELINE_DATASET}" \
        --image_encoder "${BASELINE_IMAGE_ENCODER}" \
        --text_encoder "${BASELINE_TEXT_ENCODER}" \
        --feature_source "${BASELINE_FEATURE_SOURCE}" \
        --image_root "${BASELINE_IMAGE_ROOT}" \
        --ann_root "${BASELINE_ANN_ROOT}" \
        --device "${BASELINE_DEVICE}" \
        --epochs "${BASELINE_EPOCHS}" \
        --batch_size_train "${BASELINE_BATCH_TRAIN}" \
        --batch_size_test "${BASELINE_BATCH_TEST}" \
        --text_batch_size "${BASELINE_TEXT_BATCH}" \
        --num_workers "${BASELINE_NUM_WORKERS}" \
        --eval_interval 1 \
        --no_aug
    done
  done
done

# 3) Refresh tables and compact final table.
python -m baselines.runners.export_baseline_tables \
  --root "${BASELINE_OUTPUT_ROOT}" \
  --output_dir "${BASELINE_OUTPUT_ROOT}"

python - <<PY
import csv
import os

src = os.path.join("${BASELINE_OUTPUT_ROOT}", "main_table_aligned.csv")
dst = os.path.join("${BASELINE_OUTPUT_ROOT}", "final_results_table.csv")
if os.path.exists(src):
    keep_cols = [
        "method", "budget", "ratio", "seed", "dataset", "image_encoder", "text_encoder",
        "sample_unit", "I2T_R1", "I2T_R5", "I2T_R10", "T2I_R1", "T2I_R5", "T2I_R10",
        "MeanRecall", "selection_time", "train_time", "eval_time", "output_dir"
    ]
    with open(src, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if not r.get("dataset") and r.get("dataset_name"):
            r["dataset"] = r["dataset_name"]
    with open(dst, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keep_cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keep_cols})
    print(dst)
PY

echo ""
echo "[new4-eval-only] done."
echo "[new4-eval-only] final table: ${BASELINE_OUTPUT_ROOT}/final_results_table.csv"
