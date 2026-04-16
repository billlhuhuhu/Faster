#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

BASELINE_CONFIG="${BASELINE_CONFIG:-baselines/configs/main_aligned_flickr_nfnet_bert.yaml}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-artifacts/baselines}"
BASELINE_FEATURE_SOURCE="${BASELINE_FEATURE_SOURCE:-artifacts/feature_cache}"
BASELINE_DEVICE="${BASELINE_DEVICE:-cuda:0}"
BASELINE_DATASET="${BASELINE_DATASET:-flickr}"
BASELINE_IMAGE_ENCODER="${BASELINE_IMAGE_ENCODER:-nfnet}"
BASELINE_TEXT_ENCODER="${BASELINE_TEXT_ENCODER:-bert}"
BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-data/flickr30k}"
BASELINE_ANN_ROOT="${BASELINE_ANN_ROOT:-data/Flickr30k_ann}"
BASELINE_SEEDS="${BASELINE_SEEDS:-0}"
BASELINE_METHODS="${BASELINE_METHODS:-entropy el2n grand gradmatch glister ccs-rand ccs-herd ccs-kcenter ccs-forget dq dfool nms adap_sne presel visa dataprophet dynamic_pruning}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-20}"

# Recommended eval-time memory-safe defaults for nfnet.
BASELINE_BATCH_TRAIN="${BASELINE_BATCH_TRAIN:-16}"
BASELINE_BATCH_TEST="${BASELINE_BATCH_TEST:-32}"
BASELINE_TEXT_BATCH="${BASELINE_TEXT_BATCH:-256}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-2}"

# Absolute budgets + ratio budgets.
ABS_BUDGETS="${ABS_BUDGETS:-100 200 500}"
RATIOS="${RATIOS:-0.01 0.02 0.03}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export LORS_CHECKPOINT_ROOT="${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}"

echo "[formal] project_root=${PROJECT_ROOT}"
echo "[formal] output_root=${BASELINE_OUTPUT_ROOT}"
echo "[formal] device=${BASELINE_DEVICE}"
echo "[formal] methods=${BASELINE_METHODS}"
echo "[formal] abs_budgets=${ABS_BUDGETS}"
echo "[formal] ratios=${RATIOS}"
echo "[formal] seeds=${BASELINE_SEEDS}"
echo "[formal] batch_train=${BASELINE_BATCH_TRAIN} batch_test=${BASELINE_BATCH_TEST} text_batch=${BASELINE_TEXT_BATCH}"
echo "[formal] checkpoints=${LORS_CHECKPOINT_ROOT}"

run_abs_job() {
  local method="$1"
  local budget="$2"
  local seed="$3"

  local run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}/${method}/budget_$(printf "%04d" "${budget}")/seed_${seed}"
  local selected_path="${run_dir}/selected_indices.json"
  local metrics_path="${run_dir}/downstream_metrics.json"

  if [[ -f "${metrics_path}" ]]; then
    echo "[skip][abs] done method=${method} budget=${budget} seed=${seed}"
    return 0
  fi

  if [[ ! -f "${selected_path}" ]]; then
    echo "[run][abs][selection] method=${method} budget=${budget} seed=${seed}"
    python -m baselines.runners.run_baseline_selection \
      --method "${method}" \
      --budget "${budget}" \
      --dataset_name "${BASELINE_DATASET}" \
      --split train \
      --image_encoder "${BASELINE_IMAGE_ENCODER}" \
      --text_encoder "${BASELINE_TEXT_ENCODER}" \
      --feature_source "${BASELINE_FEATURE_SOURCE}" \
      --output_dir "${BASELINE_OUTPUT_ROOT}" \
      --config "${BASELINE_CONFIG}" \
      --output_layout budget \
      --seed "${seed}" \
      --device "${BASELINE_DEVICE}"
  else
    echo "[skip][abs][selection] exists method=${method} budget=${budget} seed=${seed}"
  fi

  echo "[run][abs][eval] method=${method} budget=${budget} seed=${seed}"
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
}

run_ratio_job() {
  local method="$1"
  local ratio="$2"
  local seed="$3"

  local ratio_tag
  ratio_tag="$(python - <<PY
r = float("${ratio}")
print(f"ratio_{int(round(r*100)):02d}")
PY
)"
  local model_tag="${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}"
  local run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${ratio_tag}/${method}/seed_${seed}"
  local selected_path="${run_dir}/selected_indices.json"
  local metrics_path="${run_dir}/downstream_metrics.json"

  if [[ -f "${metrics_path}" ]]; then
    echo "[skip][ratio] done method=${method} ratio=${ratio} seed=${seed}"
    return 0
  fi

  if [[ ! -f "${selected_path}" ]]; then
    echo "[run][ratio][selection] method=${method} ratio=${ratio} seed=${seed}"
    python -m baselines.runners.run_baseline_selection \
      --method "${method}" \
      --ratio "${ratio}" \
      --dataset_name "${BASELINE_DATASET}" \
      --split train \
      --image_encoder "${BASELINE_IMAGE_ENCODER}" \
      --text_encoder "${BASELINE_TEXT_ENCODER}" \
      --feature_source "${BASELINE_FEATURE_SOURCE}" \
      --output_dir "${BASELINE_OUTPUT_ROOT}" \
      --config "${BASELINE_CONFIG}" \
      --output_layout ratio \
      --seed "${seed}" \
      --device "${BASELINE_DEVICE}"
  else
    echo "[skip][ratio][selection] exists method=${method} ratio=${ratio} seed=${seed}"
  fi

  echo "[run][ratio][eval] method=${method} ratio=${ratio} seed=${seed}"
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
}

# 1) Absolute budgets
for seed in ${BASELINE_SEEDS}; do
  for budget in ${ABS_BUDGETS}; do
    for method in ${BASELINE_METHODS}; do
      run_abs_job "${method}" "${budget}" "${seed}"
    done
  done
done

# 2) Ratio budgets
for seed in ${BASELINE_SEEDS}; do
  for ratio in ${RATIOS}; do
    for method in ${BASELINE_METHODS}; do
      run_ratio_job "${method}" "${ratio}" "${seed}"
    done
  done
done

# 3) Export merged tables
python -m baselines.runners.export_baseline_tables \
  --root "${BASELINE_OUTPUT_ROOT}" \
  --output_dir "${BASELINE_OUTPUT_ROOT}"

# 4) Keep only one final compact table
python - <<PY
import csv
import os

src = os.path.join("${BASELINE_OUTPUT_ROOT}", "main_table_aligned.csv")
dst = os.path.join("${BASELINE_OUTPUT_ROOT}", "final_results_table.csv")
if not os.path.exists(src):
    raise FileNotFoundError(src)

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
echo "[formal] done."
echo "[formal] final table: ${BASELINE_OUTPUT_ROOT}/final_results_table.csv"
