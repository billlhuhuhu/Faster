#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

# Default: the 4 newly added multimodal baselines (override if needed).
BASELINE_METHODS="${BASELINE_METHODS:-presel visa dataprophet dynamic_pruning}"
ABS_BUDGETS="${ABS_BUDGETS:-100 200 500}"
RATIOS="${RATIOS:-0.01 0.02 0.03}"
DATAPROPHET_MASTER_RATIO="${DATAPROPHET_MASTER_RATIO:-0.05}"
BASELINE_SEEDS="${BASELINE_SEEDS:-0}"
BASELINE_DEVICE="${BASELINE_DEVICE:-cuda:0}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-artifacts/baselines_coco}"

BASELINE_DATASET="${BASELINE_DATASET:-coco}"
BASELINE_IMAGE_ENCODER="${BASELINE_IMAGE_ENCODER:-nfnet}"
BASELINE_TEXT_ENCODER="${BASELINE_TEXT_ENCODER:-bert}"
BASELINE_FEATURE_SOURCE="${BASELINE_FEATURE_SOURCE:-artifacts/feature_cache}"
BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-data/coco}"
BASELINE_ANN_ROOT="${BASELINE_ANN_ROOT:-data/COCO}"

# Align with main experiment defaults.
BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
BASELINE_BATCH_TRAIN="${BASELINE_BATCH_TRAIN:-64}"
BASELINE_BATCH_TEST="${BASELINE_BATCH_TEST:-128}"
BASELINE_TEXT_BATCH="${BASELINE_TEXT_BATCH:-1024}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-4}"
BASELINE_BATCH_TRAIN_DYNAMIC="${BASELINE_BATCH_TRAIN_DYNAMIC:-8}"
BASELINE_BATCH_TEST_DYNAMIC="${BASELINE_BATCH_TEST_DYNAMIC:-16}"
BASELINE_TEXT_BATCH_DYNAMIC="${BASELINE_TEXT_BATCH_DYNAMIC:-128}"
BASELINE_NUM_WORKERS_DYNAMIC="${BASELINE_NUM_WORKERS_DYNAMIC:-2}"
BASELINE_CONFIG="${BASELINE_CONFIG:-baselines/configs/main_aligned_flickr_nfnet_bert.yaml}"
BASELINE_POOL_DYNAMIC_PRUNING="${BASELINE_POOL_DYNAMIC_PRUNING:-50000}"
BASELINE_POOL_DFOOL="${BASELINE_POOL_DFOOL:-50000}"
BASELINE_POOL_MODE="${BASELINE_POOL_MODE:-head}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

# Multi-GPU scheduling (round-robin).
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_LIST="${GPU_LIST//,/ }"
read -r -a GPU_ARRAY <<< "${GPU_LIST}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  GPU_ARRAY=("0")
fi
GPU_COUNT="${#GPU_ARRAY[@]}"
MAX_PARALLEL="${MAX_PARALLEL:-${GPU_COUNT}}"

echo "[coco-formal] dataset=${BASELINE_DATASET}"
echo "[coco-formal] methods=${BASELINE_METHODS}"
echo "[coco-formal] abs_budgets=${ABS_BUDGETS}"
echo "[coco-formal] ratios=${RATIOS:-<disabled>}"
echo "[coco-formal] dataprophet_master_ratio=${DATAPROPHET_MASTER_RATIO}"
echo "[coco-formal] seeds=${BASELINE_SEEDS}"
echo "[coco-formal] output_root=${BASELINE_OUTPUT_ROOT}"
echo "[coco-formal] device=${BASELINE_DEVICE}"
echo "[coco-formal] image_root=${BASELINE_IMAGE_ROOT}"
echo "[coco-formal] ann_root=${BASELINE_ANN_ROOT}"
echo "[coco-formal] pool_dynamic_pruning=${BASELINE_POOL_DYNAMIC_PRUNING} pool_dfool=${BASELINE_POOL_DFOOL} pool_mode=${BASELINE_POOL_MODE}"
echo "[coco-formal] eval_batch_default=${BASELINE_BATCH_TRAIN}/${BASELINE_BATCH_TEST}/${BASELINE_TEXT_BATCH}"
echo "[coco-formal] eval_batch_dynamic=${BASELINE_BATCH_TRAIN_DYNAMIC}/${BASELINE_BATCH_TEST_DYNAMIC}/${BASELINE_TEXT_BATCH_DYNAMIC}"
echo "[coco-formal] gpus=${GPU_ARRAY[*]} max_parallel=${MAX_PARALLEL}"

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

method_selection_pool_args() {
  local method="$1"
  case "${method}" in
    dynamic_pruning|infobatch)
      echo "--candidate_pool_size ${BASELINE_POOL_DYNAMIC_PRUNING} --candidate_pool_mode ${BASELINE_POOL_MODE}"
      ;;
    dfool)
      echo "--candidate_pool_size ${BASELINE_POOL_DFOOL} --candidate_pool_mode ${BASELINE_POOL_MODE}"
      ;;
    *)
      echo ""
      ;;
  esac
}

method_eval_batch_args() {
  local method="$1"
  if [[ "${method}" == "dynamic_pruning" || "${method}" == "infobatch" ]]; then
    echo "${BASELINE_BATCH_TRAIN_DYNAMIC} ${BASELINE_BATCH_TEST_DYNAMIC} ${BASELINE_TEXT_BATCH_DYNAMIC} ${BASELINE_NUM_WORKERS_DYNAMIC}"
  else
    echo "${BASELINE_BATCH_TRAIN} ${BASELINE_BATCH_TEST} ${BASELINE_TEXT_BATCH} ${BASELINE_NUM_WORKERS}"
  fi
}

run_eval_with_oom_retry() {
  local method="$1"
  local run_dir="$2"
  local dataset_name="$3"
  local image_encoder="$4"
  local text_encoder="$5"
  local feature_source="$6"
  local image_root="$7"
  local ann_root="$8"
  local device="$9"
  local epochs="${10}"
  local eval_interval="${11}"
  local batch_train="${12}"
  local batch_test="${13}"
  local text_batch="${14}"
  local num_workers="${15}"
  local gpu_id="${16}"

  set +e
  CUDA_VISIBLE_DEVICES="${gpu_id}" python -m baselines.runners.evaluate_baseline_subsets \
    --baseline_result_dir "${run_dir}" \
    --dataset_name "${dataset_name}" \
    --image_encoder "${image_encoder}" \
    --text_encoder "${text_encoder}" \
    --feature_source "${feature_source}" \
    --image_root "${image_root}" \
    --ann_root "${ann_root}" \
    --device "${device}" \
    --epochs "${epochs}" \
    --batch_size_train "${batch_train}" \
    --batch_size_test "${batch_test}" \
    --text_batch_size "${text_batch}" \
    --num_workers "${num_workers}" \
    --eval_interval "${eval_interval}" \
    --no_aug
  local rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    return 0
  fi

  local log_path="${run_dir}/train_eval_log.txt"
  if [[ -f "${log_path}" ]] && grep -q "OutOfMemoryError" "${log_path}"; then
    local retry_train=$(( batch_train > 1 ? batch_train / 2 : 1 ))
    local retry_test=$(( batch_test > 1 ? batch_test / 2 : 1 ))
    local retry_text=$(( text_batch > 1 ? text_batch / 2 : 1 ))
    local retry_workers=$(( num_workers > 1 ? num_workers - 1 : 1 ))
    echo "[oom-retry] method=${method} run_dir=${run_dir}"
    echo "[oom-retry] retry batch_train=${retry_train} batch_test=${retry_test} text_batch=${retry_text} workers=${retry_workers}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m baselines.runners.evaluate_baseline_subsets \
      --baseline_result_dir "${run_dir}" \
      --dataset_name "${dataset_name}" \
      --image_encoder "${image_encoder}" \
      --text_encoder "${text_encoder}" \
      --feature_source "${feature_source}" \
      --image_root "${image_root}" \
      --ann_root "${ann_root}" \
      --device "${device}" \
      --epochs "${epochs}" \
      --batch_size_train "${retry_train}" \
      --batch_size_test "${retry_test}" \
      --text_batch_size "${retry_text}" \
      --num_workers "${retry_workers}" \
      --eval_interval "${eval_interval}" \
      --no_aug
    return $?
  fi
  return "${rc}"
}

max_abs_budget() {
  local max_b=0
  local b
  for b in ${ABS_BUDGETS}; do
    if [[ "${b}" -gt "${max_b}" ]]; then
      max_b="${b}"
    fi
  done
  echo "${max_b}"
}

ratio_to_tag() {
  local ratio="$1"
  python - <<PY
r = float("${ratio}")
print(f"ratio_{int(round(r*100)):02d}")
PY
}

ensure_dataprophet_master_selection() {
  local seed="$1"
  local gpu_id="${2:-0}"
  run_ratio_selection_only "dataprophet" "${DATAPROPHET_MASTER_RATIO}" "${seed}" "${gpu_id}"
}

materialize_dataprophet_budget_from_master_ratio() {
  local seed="$1"
  local target_budget="$2"
  local master_ratio="$3"

  local method="dataprophet"
  local master_tag
  master_tag="$(ratio_to_tag "${master_ratio}")"
  local model_tag="${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}"
  local src_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${master_tag}/${method}/seed_${seed}"
  local dst_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}/${method}/budget_$(printf "%04d" "${target_budget}")/seed_${seed}"
  local src_selected="${src_dir}/selected_indices.json"
  local src_summary="${src_dir}/baseline_summary.json"
  local src_scores="${src_dir}/selection_scores.npz"
  local dst_selected="${dst_dir}/selected_indices.json"
  local dst_summary="${dst_dir}/baseline_summary.json"
  local dst_scores="${dst_dir}/selection_scores.npz"

  if [[ -f "${dst_selected}" ]]; then
    echo "[dataprophet-reuse][skip] selection exists budget=${target_budget} seed=${seed}"
    return 0
  fi
  if [[ ! -f "${src_selected}" ]]; then
    echo "[dataprophet-reuse][warn] source selection missing, cannot derive budget=${target_budget} from ratio=${master_ratio}"
    return 1
  fi

  mkdir -p "${dst_dir}"
  python - <<PY
import json
import os
import shutil

src_selected = "${src_selected}"
src_summary = "${src_summary}"
src_scores = "${src_scores}"
dst_selected = "${dst_selected}"
dst_summary = "${dst_summary}"
dst_scores = "${dst_scores}"
target_budget = int("${target_budget}")
master_ratio = float("${master_ratio}")

with open(src_selected, "r", encoding="utf-8") as f:
    payload = json.load(f)
src_idx = [int(x) for x in payload.get("selected_indices", [])]
selected = src_idx[:target_budget]
with open(dst_selected, "w", encoding="utf-8") as f:
    json.dump({"selected_indices": selected}, f, ensure_ascii=False, indent=2)

summary = {}
if os.path.exists(src_summary):
    with open(src_summary, "r", encoding="utf-8") as f:
        summary = json.load(f)
total = int(summary.get("total_train_size", max(len(selected), 1)))
summary["budget"] = int(target_budget)
summary["subset_size"] = int(len(selected))
summary["ratio"] = float(len(selected)) / max(float(total), 1.0)
summary["derived_from_ratio"] = float(master_ratio)
summary["derived_from"] = src_selected
with open(dst_summary, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

if os.path.exists(src_scores):
    shutil.copy2(src_scores, dst_scores)
PY
  echo "[dataprophet-reuse] derived budget=${target_budget} from ratio=${master_ratio} seed=${seed}"
}

materialize_dataprophet_ratio_from_master_ratio() {
  local seed="$1"
  local target_ratio="$2"
  local master_ratio="$3"

  local method="dataprophet"
  local target_tag
  local master_tag
  target_tag="$(ratio_to_tag "${target_ratio}")"
  master_tag="$(ratio_to_tag "${master_ratio}")"

  if [[ "${target_tag}" == "${master_tag}" ]]; then
    return 0
  fi

  local model_tag="${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}"
  local src_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${master_tag}/${method}/seed_${seed}"
  local dst_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${target_tag}/${method}/seed_${seed}"
  local src_selected="${src_dir}/selected_indices.json"
  local src_summary="${src_dir}/baseline_summary.json"
  local src_scores="${src_dir}/selection_scores.npz"
  local dst_selected="${dst_dir}/selected_indices.json"
  local dst_summary="${dst_dir}/baseline_summary.json"
  local dst_scores="${dst_dir}/selection_scores.npz"

  if [[ -f "${dst_selected}" ]]; then
    echo "[dataprophet-reuse][skip] ratio selection exists ratio=${target_ratio} seed=${seed}"
    return 0
  fi
  if [[ ! -f "${src_selected}" ]]; then
    echo "[dataprophet-reuse][warn] source ratio selection missing, cannot derive ratio=${target_ratio} from ratio=${master_ratio}"
    return 1
  fi

  mkdir -p "${dst_dir}"
  python - <<PY
import json
import math
import os
import shutil

src_selected = "${src_selected}"
src_summary = "${src_summary}"
src_scores = "${src_scores}"
dst_selected = "${dst_selected}"
dst_summary = "${dst_summary}"
dst_scores = "${dst_scores}"
target_ratio = float("${target_ratio}")
master_ratio = float("${master_ratio}")

with open(src_selected, "r", encoding="utf-8") as f:
    payload = json.load(f)
src_idx = [int(x) for x in payload.get("selected_indices", [])]

summary = {}
if os.path.exists(src_summary):
    with open(src_summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

total = int(summary.get("total_train_size", max(len(src_idx), 1)))
target_k = int(round(target_ratio * total))
target_k = max(1, min(target_k, len(src_idx)))
selected = src_idx[:target_k]

with open(dst_selected, "w", encoding="utf-8") as f:
    json.dump({"selected_indices": selected}, f, ensure_ascii=False, indent=2)

summary["ratio"] = float(len(selected)) / max(float(total), 1.0)
summary["budget"] = int(len(selected))
summary["subset_size"] = int(len(selected))
summary["derived_from_ratio"] = float(master_ratio)
summary["derived_from"] = src_selected
with open(dst_summary, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

if os.path.exists(src_scores):
    shutil.copy2(src_scores, dst_scores)
PY
  echo "[dataprophet-reuse] derived ratio=${target_ratio} from ratio=${master_ratio} seed=${seed}"
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
    extra_args="$(method_selection_pool_args "${method}")"
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
  read -r eval_train eval_test eval_text eval_workers <<< "$(method_eval_batch_args "${method}")"
  run_eval_with_oom_retry \
    "${method}" "${run_dir}" "${BASELINE_DATASET}" "${BASELINE_IMAGE_ENCODER}" "${BASELINE_TEXT_ENCODER}" \
    "${BASELINE_FEATURE_SOURCE}" "${BASELINE_IMAGE_ROOT}" "${BASELINE_ANN_ROOT}" "${BASELINE_DEVICE}" \
    "${BASELINE_EPOCHS}" "1" "${eval_train}" "${eval_test}" "${eval_text}" "${eval_workers}" "${gpu_id}"
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
    extra_args="$(method_selection_pool_args "${method}")"
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
  read -r eval_train eval_test eval_text eval_workers <<< "$(method_eval_batch_args "${method}")"
  run_eval_with_oom_retry \
    "${method}" "${run_dir}" "${BASELINE_DATASET}" "${BASELINE_IMAGE_ENCODER}" "${BASELINE_TEXT_ENCODER}" \
    "${BASELINE_FEATURE_SOURCE}" "${BASELINE_IMAGE_ROOT}" "${BASELINE_ANN_ROOT}" "${BASELINE_DEVICE}" \
    "${BASELINE_EPOCHS}" "1" "${eval_train}" "${eval_test}" "${eval_text}" "${eval_workers}" "${gpu_id}"
}

run_ratio_selection_only() {
  local method="$1"
  local ratio="$2"
  local seed="$3"
  local gpu_id="$4"

  local ratio_tag
  ratio_tag="$(ratio_to_tag "${ratio}")"
  local model_tag="${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}"
  local run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${ratio_tag}/${method}/seed_${seed}"
  local selected_path="${run_dir}/selected_indices.json"

  if [[ -f "${selected_path}" ]]; then
    echo "[skip][ratio][selection-only] exists method=${method} ratio=${ratio} seed=${seed}"
    return 0
  fi

  echo "[run][ratio][selection-only] method=${method} ratio=${ratio} seed=${seed} gpu=${gpu_id}"
  local extra_args
  extra_args="$(method_selection_pool_args "${method}")"
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
}

# 1) Absolute budgets
job_idx=0
for seed in ${BASELINE_SEEDS}; do
  # dataprophet: run one master ratio selection, then derive all budgets.
  if [[ " ${BASELINE_METHODS} " == *" dataprophet "* ]]; then
    gpu_id="$(pick_gpu "${job_idx}")"
    ensure_dataprophet_master_selection "${seed}" "${gpu_id}"
    job_idx=$((job_idx + 1))
    for budget in ${ABS_BUDGETS}; do
      materialize_dataprophet_budget_from_master_ratio "${seed}" "${budget}" "${DATAPROPHET_MASTER_RATIO}" || true
      gpu_id="$(pick_gpu "${job_idx}")"
      run_abs_job "dataprophet" "${budget}" "${seed}" "${gpu_id}"
      job_idx=$((job_idx + 1))
    done
  fi

  for budget in ${ABS_BUDGETS}; do
    for method in ${BASELINE_METHODS}; do
      if [[ "${method}" == "dataprophet" ]]; then
        continue
      fi
      throttle_jobs
      gpu_id="$(pick_gpu "${job_idx}")"
      run_abs_job "${method}" "${budget}" "${seed}" "${gpu_id}" &
      job_idx=$((job_idx + 1))
    done
  done
done
wait_all_jobs

# 2) Ratio budgets (optional)
if [[ -n "${RATIOS}" ]]; then
  for seed in ${BASELINE_SEEDS}; do
    # dataprophet: derive all target ratios from the same master ratio run.
    if [[ " ${BASELINE_METHODS} " == *" dataprophet "* ]]; then
      gpu_id="$(pick_gpu "${job_idx}")"
      ensure_dataprophet_master_selection "${seed}" "${gpu_id}"
      job_idx=$((job_idx + 1))
      for ratio in ${RATIOS}; do
        materialize_dataprophet_ratio_from_master_ratio "${seed}" "${ratio}" "${DATAPROPHET_MASTER_RATIO}" || true
        gpu_id="$(pick_gpu "${job_idx}")"
        run_ratio_job "dataprophet" "${ratio}" "${seed}" "${gpu_id}"
        job_idx=$((job_idx + 1))
      done
    fi

    for ratio in ${RATIOS}; do
      for method in ${BASELINE_METHODS}; do
        if [[ "${method}" == "dataprophet" ]]; then
          continue
        fi
        throttle_jobs
        gpu_id="$(pick_gpu "${job_idx}")"
        run_ratio_job "${method}" "${ratio}" "${seed}" "${gpu_id}" &
        job_idx=$((job_idx + 1))
      done
    done
  done
fi
wait_all_jobs

# 3) Export merged tables
python -m baselines.runners.export_baseline_tables \
  --root "${BASELINE_OUTPUT_ROOT}" \
  --output_dir "${BASELINE_OUTPUT_ROOT}"

# 4) Keep one compact final table
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
echo "[coco-formal] done."
echo "[coco-formal] final table: artifacts/baselines_coco/final_results_table.csv"
