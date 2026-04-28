#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

# Unimodal-style baseline set (exclude newly added multimodal counterexample/dynamic methods).
BASELINE_METHODS="${BASELINE_METHODS:-entropy el2n grand gradmatch glister ccs-rand ccs-herd ccs-kcenter ccs-forget dq dfool nms adap_sne}"
ABS_BUDGETS="${ABS_BUDGETS:-100 200 500}"
RATIOS="${RATIOS:-0.01 0.02 0.03}"
BASELINE_SEEDS="${BASELINE_SEEDS:-0}"
# Device inside each subprocess after CUDA_VISIBLE_DEVICES remap.
BASELINE_DEVICE="${BASELINE_DEVICE:-cuda:0}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-artifacts/baselines_coco_unimodal}"

BASELINE_DATASET="${BASELINE_DATASET:-coco}"
BASELINE_IMAGE_ENCODER="${BASELINE_IMAGE_ENCODER:-nfnet}"
BASELINE_TEXT_ENCODER="${BASELINE_TEXT_ENCODER:-bert}"
BASELINE_FEATURE_SOURCE="${BASELINE_FEATURE_SOURCE:-artifacts/feature_cache}"
BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-data/coco}"
BASELINE_ANN_ROOT="${BASELINE_ANN_ROOT:-data/COCO}"
BASELINE_CONFIG="${BASELINE_CONFIG:-baselines/configs/main_aligned_flickr_nfnet_bert.yaml}"

# Align with main experiment defaults.
BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
BASELINE_BATCH_TRAIN="${BASELINE_BATCH_TRAIN:-64}"
BASELINE_BATCH_TEST="${BASELINE_BATCH_TEST:-128}"
BASELINE_TEXT_BATCH="${BASELINE_TEXT_BATCH:-1024}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-4}"
# Method-specific candidate pool to avoid OOM/kill on very large COCO train set.
BASELINE_CANDIDATE_POOL_DFOOL="${BASELINE_CANDIDATE_POOL_DFOOL:-5000}"
BASELINE_CANDIDATE_POOL_ADAPSNE="${BASELINE_CANDIDATE_POOL_ADAPSNE:-20000}"
BASELINE_CANDIDATE_POOL_ENTROPY="${BASELINE_CANDIDATE_POOL_ENTROPY:-20000}"
BASELINE_CANDIDATE_POOL_MODE="${BASELINE_CANDIDATE_POOL_MODE:-head}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

# Multi-GPU scheduling: round-robin jobs across GPU ids.
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_LIST="${GPU_LIST//,/ }"
read -r -a GPU_ARRAY <<< "${GPU_LIST}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  GPU_ARRAY=("0")
fi
GPU_COUNT="${#GPU_ARRAY[@]}"
MAX_PARALLEL="${MAX_PARALLEL:-${GPU_COUNT}}"

echo "[coco-unimodal] dataset=${BASELINE_DATASET}"
echo "[coco-unimodal] methods=${BASELINE_METHODS}"
echo "[coco-unimodal] abs_budgets=${ABS_BUDGETS}"
echo "[coco-unimodal] ratios=${RATIOS}"
echo "[coco-unimodal] seeds=${BASELINE_SEEDS}"
echo "[coco-unimodal] output_root=${BASELINE_OUTPUT_ROOT}"
echo "[coco-unimodal] device=${BASELINE_DEVICE}"
echo "[coco-unimodal] gpus=${GPU_ARRAY[*]} max_parallel=${MAX_PARALLEL}"
echo "[coco-unimodal] pool_dfool=${BASELINE_CANDIDATE_POOL_DFOOL} pool_adapsne=${BASELINE_CANDIDATE_POOL_ADAPSNE} pool_entropy=${BASELINE_CANDIDATE_POOL_ENTROPY} pool_mode=${BASELINE_CANDIDATE_POOL_MODE}"

method_pool_args() {
  local method="$1"
  case "${method}" in
    dfool)
      echo "--candidate_pool_size ${BASELINE_CANDIDATE_POOL_DFOOL} --candidate_pool_mode ${BASELINE_CANDIDATE_POOL_MODE}"
      ;;
    adap_sne)
      echo "--candidate_pool_size ${BASELINE_CANDIDATE_POOL_ADAPSNE} --candidate_pool_mode ${BASELINE_CANDIDATE_POOL_MODE}"
      ;;
    entropy)
      echo "--candidate_pool_size ${BASELINE_CANDIDATE_POOL_ENTROPY} --candidate_pool_mode ${BASELINE_CANDIDATE_POOL_MODE}"
      ;;
    *)
      echo ""
      ;;
  esac
}

pick_gpu() {
  local idx="$1"
  echo "${GPU_ARRAY[$(( idx % GPU_COUNT ))]}"
}

throttle_jobs() {
  while [[ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]]; do
    wait -n
  done
}

wait_all_jobs() {
  while [[ "$(jobs -rp | wc -l)" -gt 0 ]]; do
    wait -n
  done
}

run_abs_job() {
  local method="$1"
  local budget="$2"
  local seed="$3"
  local gpu_id="$4"

  local run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}/${method}/budget_$(printf "%04d" "${budget}")/seed_${seed}"
  local selected_path="${run_dir}/selected_indices.json"
  local metrics_path="${run_dir}/downstream_metrics.json"

  if [[ -f "${metrics_path}" ]]; then
    echo "[skip][abs] done method=${method} budget=${budget} seed=${seed}"
    return 0
  fi

  if [[ ! -f "${selected_path}" ]]; then
    echo "[run][abs][selection] method=${method} budget=${budget} seed=${seed} gpu=${gpu_id}"
    local extra_args
    extra_args="$(method_pool_args "${method}")"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m baselines.runners.run_baseline_selection \
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
      --device "${BASELINE_DEVICE}" \
      ${extra_args}
  else
    echo "[skip][abs][selection] exists method=${method} budget=${budget} seed=${seed}"
  fi

  echo "[run][abs][eval] method=${method} budget=${budget} seed=${seed} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python -m baselines.runners.evaluate_baseline_subsets \
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
  local gpu_id="$4"

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
    echo "[run][ratio][selection] method=${method} ratio=${ratio} seed=${seed} gpu=${gpu_id}"
    local extra_args
    extra_args="$(method_pool_args "${method}")"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m baselines.runners.run_baseline_selection \
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
      --device "${BASELINE_DEVICE}" \
      ${extra_args}
  else
    echo "[skip][ratio][selection] exists method=${method} ratio=${ratio} seed=${seed}"
  fi

  echo "[run][ratio][eval] method=${method} ratio=${ratio} seed=${seed} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python -m baselines.runners.evaluate_baseline_subsets \
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

job_idx=0
for seed in ${BASELINE_SEEDS}; do
  for budget in ${ABS_BUDGETS}; do
    for method in ${BASELINE_METHODS}; do
      throttle_jobs
      gpu_id="$(pick_gpu "${job_idx}")"
      run_abs_job "${method}" "${budget}" "${seed}" "${gpu_id}" &
      job_idx=$((job_idx + 1))
    done
  done
done
wait_all_jobs

for seed in ${BASELINE_SEEDS}; do
  for ratio in ${RATIOS}; do
    for method in ${BASELINE_METHODS}; do
      throttle_jobs
      gpu_id="$(pick_gpu "${job_idx}")"
      run_ratio_job "${method}" "${ratio}" "${seed}" "${gpu_id}" &
      job_idx=$((job_idx + 1))
    done
  done
done
wait_all_jobs

python -m baselines.runners.export_baseline_tables \
  --root "${BASELINE_OUTPUT_ROOT}" \
  --output_dir "${BASELINE_OUTPUT_ROOT}"

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
echo "[coco-unimodal] done."
echo "[coco-unimodal] final table: ${BASELINE_OUTPUT_ROOT}/final_results_table.csv"
