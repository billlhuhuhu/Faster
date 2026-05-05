#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${OURS_RESNET10_DATASET:-flickr}"
RATIO="${OURS_RESNET10_RATIO:-0.03}"
RATIO_TAG="$(python - "${RATIO}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
)"
SOURCE_BACKBONE="${OURS_RESNET10_SOURCE_BACKBONE:-nfnet}"
TEXT_ENCODER="${OURS_RESNET10_TEXT_ENCODER:-bert}"
EVAL_BACKBONE="${OURS_RESNET10_EVAL_BACKBONE:-resnet10}"
SEED="${OURS_RESNET10_SEED:-0}"
SELECTION_ROOT="${OURS_RESNET10_SELECTION_ROOT:-artifacts/subset_selection_dense_sift_bovw}"
SELECTION_TAG="${OURS_RESNET10_SELECTION_TAG:-proxy_opt_lsrc}"
OUTPUT_ROOT="${OURS_RESNET10_OUTPUT_ROOT:-artifacts/arch_bias_energy_3pct/ours_resnet10_reuse_selection}"
RUN_TAG="${OURS_RESNET10_RUN_TAG:-ours_resnet10_3pct_$(date '+%Y%m%d_%H%M%S')}"
TRAIN_ROOT="${OUTPUT_ROOT}/subset_train/${RUN_TAG}"
REPORT_DIR="${OUTPUT_ROOT}/reports/${RUN_TAG}"
LOG_DIR="${OUTPUT_ROOT}/logs/${RUN_TAG}"
MEASURE_DIR="${OUTPUT_ROOT}/measurements/${RUN_TAG}"
MANIFEST_PATH="${REPORT_DIR}/manifest.jsonl"
IMAGE_ROOT="$(get_image_root "${DATASET}")"
SOURCE_MODEL_TAG="$(sanitize_component "${SOURCE_BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
EVAL_MODEL_TAG="$(sanitize_component "${EVAL_BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
SELECTED_INDICES_PATH="${OURS_RESNET10_SELECTED_INDICES_PATH:-${SELECTION_ROOT}/${DATASET}/train/${SOURCE_MODEL_TAG}/${RATIO_TAG}/${SELECTION_TAG}/seed_${SEED}/selected_indices.json}"
SUBSET_TAG="${OURS_RESNET10_SUBSET_TAG:-ours_resnet10_reuse_selection}"
METRICS_PATH="${TRAIN_ROOT}/${DATASET}/${EVAL_MODEL_TAG}/${RATIO_TAG}/${SUBSET_TAG}/seed_${SEED}/metrics.json"
LOG_PATH="${LOG_DIR}/ours_${RATIO_TAG}_${EVAL_BACKBONE}_seed${SEED}_train.log"
MEASURE_PATH="${MEASURE_DIR}/ours_${RATIO_TAG}_${EVAL_BACKBONE}_seed${SEED}_train.json"
ENERGY_PREFER_ZEUS="${ENERGY_PREFER_ZEUS:-1}"
ENERGY_GPU_SAMPLER_INTERVAL="${ENERGY_GPU_SAMPLER_INTERVAL:-1.0}"
GPU_COUNT="${OURS_RESNET10_GPU_COUNT:-1}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "${MEASURE_DIR}" "${TRAIN_ROOT}"
: > "${MANIFEST_PATH}"

if [[ ! -f "${SELECTED_INDICES_PATH}" ]]; then
  echo "Missing Ours selected indices: ${SELECTED_INDICES_PATH}" >&2
  echo "Set OURS_RESNET10_SELECTED_INDICES_PATH=/path/to/selected_indices.json if your selection root is custom." >&2
  exit 1
fi

train_extra=()
if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
  train_extra+=(--no_aug)
fi

zeus_args=()
if [[ "${ENERGY_PREFER_ZEUS}" == "1" ]]; then
  zeus_args+=(--prefer_zeus)
fi

stage_log "Ours resnet10 reuse-selection training/eval start"
stage_log "  dataset=${DATASET} source=${SOURCE_MODEL_TAG} eval=${EVAL_MODEL_TAG} ratio=${RATIO_TAG} seed=${SEED}"
stage_log "  selected=${SELECTED_INDICES_PATH}"
stage_log "  output=${TRAIN_ROOT}"

python "${PROJECT_ROOT}/tools/measure_command_energy.py" \
  --label "ours_${RATIO_TAG}_${EVAL_BACKBONE}_seed${SEED}_train" \
  --output_json "${MEASURE_PATH}" \
  --working_dir "${PROJECT_ROOT}" \
  --gpu_sampler_interval "${ENERGY_GPU_SAMPLER_INTERVAL}" \
  --tee_log "${LOG_PATH}" \
  "${zeus_args[@]}" \
  -- \
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python "${PROJECT_ROOT}/run_subset_train.py" \
    --dataset "${DATASET}" \
    --image_root "${IMAGE_ROOT}" \
    --ann_root "${ANN_ROOT}" \
    --selected_indices_path "${SELECTED_INDICES_PATH}" \
    --subset_ratio "${RATIO}" \
    --subset_tag "${SUBSET_TAG}" \
    --image_encoder "${EVAL_BACKBONE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --output_root "${TRAIN_ROOT}" \
    --batch_size_train "${OURS_RESNET10_BATCH_TRAIN:-32}" \
    --batch_size_test "${OURS_RESNET10_BATCH_TEST:-64}" \
    --text_batch_size "${OURS_RESNET10_TEXT_BATCH_SIZE:-512}" \
    --num_workers "${NUM_WORKERS}" \
    --epochs "${OURS_RESNET10_EPOCHS:-50}" \
    --eval_interval "${OURS_RESNET10_EVAL_INTERVAL:-1}" \
    --lr_teacher_img "${OURS_RESNET10_LR_IMG:-0.001}" \
    --lr_teacher_txt "${OURS_RESNET10_LR_TXT:-0.05}" \
    --image_trainable "${OURS_RESNET10_IMAGE_TRAINABLE:-true}" \
    --text_trainable "${OURS_RESNET10_TEXT_TRAINABLE:-false}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    "${train_extra[@]}"

python - "${MANIFEST_PATH}" "${DATASET}" "${RATIO}" "${RATIO_TAG}" "${EVAL_BACKBONE}" "${SEED}" "${SELECTED_INDICES_PATH}" "${METRICS_PATH}" "${LOG_PATH}" "${MEASURE_PATH}" "${GPU_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
row = {
    "method": "ours",
    "dataset": sys.argv[2],
    "budget_type": "ratio",
    "budget_value": sys.argv[3],
    "budget_tag": sys.argv[4],
    "eval_backbone": sys.argv[5],
    "seed": sys.argv[6],
    "stage": "training_eval",
    "selected_indices_path": sys.argv[7],
    "metrics_path": sys.argv[8],
    "log_path": sys.argv[9],
    "measurement_path": sys.argv[10],
    "gpu_count": int(sys.argv[11]),
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${MANIFEST_PATH}" \
  --output_dir "${REPORT_DIR}"

stage_log "Ours resnet10 reuse-selection training/eval done"
stage_log "  metrics=${METRICS_PATH}"
stage_log "  detail=${REPORT_DIR}/supplemental_detail.csv"
stage_log "  architecture_bias=${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${REPORT_DIR}/energy_efficiency.csv"
