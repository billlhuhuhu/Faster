#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

LORS_BUFFER_ENERGY_DATASET="${LORS_BUFFER_ENERGY_DATASET:-flickr}"
LORS_BUFFER_ENERGY_IMAGE_ENCODER="${LORS_BUFFER_ENERGY_IMAGE_ENCODER:-resnet10}"
LORS_BUFFER_ENERGY_TEXT_ENCODER="${LORS_BUFFER_ENERGY_TEXT_ENCODER:-bert}"
LORS_BUFFER_ENERGY_LOSS_TYPE="${LORS_BUFFER_ENERGY_LOSS_TYPE:-InfoNCE}"
LORS_BUFFER_ENERGY_NUM_EXPERTS="${LORS_BUFFER_ENERGY_NUM_EXPERTS:-1}"
LORS_BUFFER_ENERGY_TRAIN_EPOCHS="${LORS_BUFFER_ENERGY_TRAIN_EPOCHS:-50}"
LORS_BUFFER_ENERGY_EVAL_FREQ="${LORS_BUFFER_ENERGY_EVAL_FREQ:-5}"
LORS_BUFFER_ENERGY_BATCH_TRAIN="${LORS_BUFFER_ENERGY_BATCH_TRAIN:-128}"
LORS_BUFFER_ENERGY_BATCH_TEST="${LORS_BUFFER_ENERGY_BATCH_TEST:-128}"
LORS_BUFFER_ENERGY_NO_AUG="${LORS_BUFFER_ENERGY_NO_AUG:-1}"
LORS_BUFFER_ENERGY_DEVICE="${LORS_BUFFER_ENERGY_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
LORS_BUFFER_ENERGY_ROOT="${LORS_BUFFER_ENERGY_ROOT:-artifacts/arch_bias_energy_3pct/lors/buffer_energy_probe}"
LORS_BUFFER_ENERGY_CHECKPOINT_ROOT="${LORS_BUFFER_ENERGY_CHECKPOINT_ROOT:-${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}}"
ENERGY_GPU_SAMPLER_INTERVAL="${ENERGY_GPU_SAMPLER_INTERVAL:-1.0}"
ENERGY_PREFER_ZEUS="${ENERGY_PREFER_ZEUS:-1}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_TAG="${LORS_BUFFER_ENERGY_RUN_TAG:-single_buffer_${RUN_TIMESTAMP}}"
LOG_DIR="${LORS_BUFFER_ENERGY_ROOT}/logs/${RUN_TAG}"
MEASURE_DIR="${LORS_BUFFER_ENERGY_ROOT}/measurements/${RUN_TAG}"
BUFFER_ROOT="${LORS_BUFFER_ENERGY_ROOT}/buffers/${RUN_TAG}"
mkdir -p "${LOG_DIR}" "${MEASURE_DIR}" "${BUFFER_ROOT}"

IMAGE_ROOT="$(get_image_root "${LORS_BUFFER_ENERGY_DATASET}")"
MODEL_TAG="$(sanitize_component "${LORS_BUFFER_ENERGY_IMAGE_ENCODER}")_$(sanitize_component "${LORS_BUFFER_ENERGY_TEXT_ENCODER}")"
LOSS_TAG="${LORS_BUFFER_ENERGY_LOSS_TYPE}"
if [[ "${LORS_BUFFER_ENERGY_NO_AUG}" == "1" ]]; then
  LOSS_TAG="${LOSS_TAG}_NoAug"
fi

measure_command() {
  local label="$1"
  local measurement_path="$2"
  local log_path="$3"
  shift 3
  local zeus_args=()
  if [[ "${ENERGY_PREFER_ZEUS}" == "1" ]]; then
    zeus_args+=(--prefer_zeus)
  fi
  python "${PROJECT_ROOT}/tools/measure_command_energy.py" \
    --label "${label}" \
    --output_json "${measurement_path}" \
    --working_dir "${PROJECT_ROOT}" \
    --gpu_sampler_interval "${ENERGY_GPU_SAMPLER_INTERVAL}" \
    --tee_log "${log_path}" \
    "${zeus_args[@]}" \
    -- "$@"
}

stage_log "LoRS single-buffer energy measurement start"
stage_log "  dataset=${LORS_BUFFER_ENERGY_DATASET}"
stage_log "  upstream=${MODEL_TAG} loss=${LOSS_TAG}"
stage_log "  num_experts=${LORS_BUFFER_ENERGY_NUM_EXPERTS} train_epochs=${LORS_BUFFER_ENERGY_TRAIN_EPOCHS}"
stage_log "  buffer_root=${BUFFER_ROOT}"
stage_log "  logs=${LOG_DIR}"
stage_log "  measurements=${MEASURE_DIR}"

BUFFER_LOG="${LOG_DIR}/lors_single_buffer.log"
BUFFER_MEASURE="${MEASURE_DIR}/lors_single_buffer_energy.json"
buffer_extra_args=()
if [[ "${LORS_BUFFER_ENERGY_NO_AUG}" == "1" ]]; then
  buffer_extra_args+=(--no_aug)
fi

measure_command "lors_single_buffer_${MODEL_TAG}" "${BUFFER_MEASURE}" "${BUFFER_LOG}" \
  env CUDA_VISIBLE_DEVICES="${LORS_BUFFER_ENERGY_DEVICE}" \
    python "${PROJECT_ROOT}/buffer.py" \
      --dataset "${LORS_BUFFER_ENERGY_DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --model_checkpoint_root "${LORS_BUFFER_ENERGY_CHECKPOINT_ROOT}" \
      --buffer_path "${BUFFER_ROOT}" \
      --image_encoder "${LORS_BUFFER_ENERGY_IMAGE_ENCODER}" \
      --text_encoder "${LORS_BUFFER_ENERGY_TEXT_ENCODER}" \
      --loss_type "${LORS_BUFFER_ENERGY_LOSS_TYPE}" \
      --num_experts "${LORS_BUFFER_ENERGY_NUM_EXPERTS}" \
      --train_epochs "${LORS_BUFFER_ENERGY_TRAIN_EPOCHS}" \
      --eval_freq "${LORS_BUFFER_ENERGY_EVAL_FREQ}" \
      --batch_size_train "${LORS_BUFFER_ENERGY_BATCH_TRAIN}" \
      --batch_size_test "${LORS_BUFFER_ENERGY_BATCH_TEST}" \
      --disabled_wandb True \
      "${buffer_extra_args[@]}"

cat > "${MEASURE_DIR}/buffer_energy_summary.json" <<JSON
{
  "dataset": "${LORS_BUFFER_ENERGY_DATASET}",
  "image_encoder": "${LORS_BUFFER_ENERGY_IMAGE_ENCODER}",
  "text_encoder": "${LORS_BUFFER_ENERGY_TEXT_ENCODER}",
  "loss_type": "${LOSS_TAG}",
  "num_experts": ${LORS_BUFFER_ENERGY_NUM_EXPERTS},
  "train_epochs": ${LORS_BUFFER_ENERGY_TRAIN_EPOCHS},
  "buffer_root": "${BUFFER_ROOT}/${LORS_BUFFER_ENERGY_DATASET}/${MODEL_TAG}/${LOSS_TAG}",
  "log_path": "${BUFFER_LOG}",
  "measurement_path": "${BUFFER_MEASURE}"
}
JSON

stage_log "LoRS single-buffer energy measurement done"
stage_log "  measurement=${BUFFER_MEASURE}"
stage_log "  summary=${MEASURE_DIR}/buffer_energy_summary.json"
