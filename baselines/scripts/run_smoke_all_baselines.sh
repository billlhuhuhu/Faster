#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

BASELINE_DEVICE="${BASELINE_DEVICE:-cuda:0}"
BASELINE_DATASET="${BASELINE_DATASET:-flickr}"
BASELINE_IMAGE_ENCODER="${BASELINE_IMAGE_ENCODER:-nfnet}"
BASELINE_TEXT_ENCODER="${BASELINE_TEXT_ENCODER:-bert}"
BASELINE_FEATURE_SOURCE="${BASELINE_FEATURE_SOURCE:-artifacts/feature_cache}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-artifacts/baselines}"
BASELINE_CONFIG="${BASELINE_CONFIG:-baselines/configs/main_aligned_flickr_nfnet_bert.yaml}"
BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-data/flickr30k}"
BASELINE_ANN_ROOT="${BASELINE_ANN_ROOT:-data/Flickr30k_ann}"

BASELINE_BUDGET="${BASELINE_BUDGET:-20}"
BASELINE_SEED="${BASELINE_SEED:-0}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-1}"
BASELINE_CANDIDATE_POOL_SIZE="${BASELINE_CANDIDATE_POOL_SIZE:-5000}"
BASELINE_CANDIDATE_POOL_MODE="${BASELINE_CANDIDATE_POOL_MODE:-head}"
BASELINE_BATCH_TRAIN="${BASELINE_BATCH_TRAIN:-32}"
BASELINE_BATCH_TEST="${BASELINE_BATCH_TEST:-64}"
BASELINE_TEXT_BATCH="${BASELINE_TEXT_BATCH:-256}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-2}"

BASELINE_METHODS="${BASELINE_METHODS:-entropy el2n grand gradmatch glister ccs-rand ccs-herd ccs-kcenter ccs-forget dq dfool nms adap_sne}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-8}"
export LORS_CHECKPOINT_ROOT="${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}"

echo "[smoke] root=${PROJECT_ROOT}"
echo "[smoke] device=${BASELINE_DEVICE} budget=${BASELINE_BUDGET} seed=${BASELINE_SEED} epochs=${BASELINE_EPOCHS}"
echo "[smoke] candidate_pool_size=${BASELINE_CANDIDATE_POOL_SIZE} mode=${BASELINE_CANDIDATE_POOL_MODE}"
echo "[smoke] methods=${BASELINE_METHODS}"
echo "[smoke] checkpoints=${LORS_CHECKPOINT_ROOT}"

ok_methods=()
failed_methods=()

for method in ${BASELINE_METHODS}; do
  echo ""
  echo "=============================="
  echo "[smoke] method=${method} (selection)"
  echo "=============================="

  if ! python -m baselines.runners.run_baseline_selection \
    --method "${method}" \
    --dataset_name "${BASELINE_DATASET}" \
    --image_encoder "${BASELINE_IMAGE_ENCODER}" \
    --text_encoder "${BASELINE_TEXT_ENCODER}" \
    --feature_source "${BASELINE_FEATURE_SOURCE}" \
    --budget "${BASELINE_BUDGET}" \
    --candidate_pool_size "${BASELINE_CANDIDATE_POOL_SIZE}" \
    --candidate_pool_mode "${BASELINE_CANDIDATE_POOL_MODE}" \
    --output_layout budget \
    --config "${BASELINE_CONFIG}" \
    --output_dir "${BASELINE_OUTPUT_ROOT}" \
    --device "${BASELINE_DEVICE}" \
    --seed "${BASELINE_SEED}"; then
    echo "[smoke][FAIL] selection failed: ${method}"
    failed_methods+=("${method}")
    continue
  fi

  run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}/${method}/budget_$(printf "%04d" "${BASELINE_BUDGET}")/seed_${BASELINE_SEED}"

  echo ""
  echo "=============================="
  echo "[smoke] method=${method} (downstream eval)"
  echo "=============================="

  if ! python -m baselines.runners.evaluate_baseline_subsets \
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
    --eval_interval 1; then
    echo "[smoke][FAIL] eval failed: ${method}"
    failed_methods+=("${method}")
    continue
  fi

  metrics_path="${run_dir}/downstream_metrics.json"
  if [[ -f "${metrics_path}" ]]; then
    echo "[smoke][OK] ${method} -> ${metrics_path}"
    ok_methods+=("${method}")
  else
    echo "[smoke][FAIL] missing downstream_metrics: ${method}"
    failed_methods+=("${method}")
  fi
done

echo ""
echo "=============================="
echo "[smoke] summary"
echo "=============================="
echo "[smoke] OK (${#ok_methods[@]}): ${ok_methods[*]:-none}"
echo "[smoke] FAIL (${#failed_methods[@]}): ${failed_methods[*]:-none}"

if [[ "${#failed_methods[@]}" -gt 0 ]]; then
  exit 1
fi

echo "[smoke] all methods passed."
