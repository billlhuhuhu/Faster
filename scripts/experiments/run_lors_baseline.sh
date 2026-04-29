#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

LORS_DATASET="${LORS_DATASET:-flickr}"
LORS_IMAGE_ENCODER="${LORS_IMAGE_ENCODER:-nfnet}"
LORS_TEXT_ENCODER="${LORS_TEXT_ENCODER:-bert}"
LORS_LOSS_TYPE="${LORS_LOSS_TYPE:-InfoNCE}"
LORS_MODEL_CHECKPOINT_ROOT="${LORS_MODEL_CHECKPOINT_ROOT:-${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}}"
LORS_BUFFER_ROOT="${LORS_BUFFER_ROOT:-buffers}"
LORS_LOG_ROOT="${LORS_LOG_ROOT:-logged_files}"
LORS_FORCE_REBUILD_BUFFER="${LORS_FORCE_REBUILD_BUFFER:-0}"
LORS_NUM_EXPERTS="${LORS_NUM_EXPERTS:-100}"
LORS_TRAIN_EPOCHS="${LORS_TRAIN_EPOCHS:-50}"
LORS_EVAL_FREQ="${LORS_EVAL_FREQ:-5}"
LORS_NUM_QUERIES="${LORS_NUM_QUERIES:-100}"
LORS_MINI_BATCH_SIZE="${LORS_MINI_BATCH_SIZE:-100}"
LORS_ITERATION="${LORS_ITERATION:-3000}"
LORS_EVAL_IT="${LORS_EVAL_IT:-50}"
LORS_NUM_EVAL="${LORS_NUM_EVAL:-1}"
LORS_EPOCH_EVAL_TRAIN="${LORS_EPOCH_EVAL_TRAIN:-100}"
LORS_EXPERT_EPOCHS="${LORS_EXPERT_EPOCHS:-3}"
LORS_SYN_STEPS="${LORS_SYN_STEPS:-20}"
LORS_MAX_START_EPOCH="${LORS_MAX_START_EPOCH:-25}"
LORS_BATCH_TRAIN="${LORS_BATCH_TRAIN:-128}"
LORS_BATCH_TEST="${LORS_BATCH_TEST:-128}"
LORS_SIM_TYPE="${LORS_SIM_TYPE:-full}"
LORS_NO_AUG="${LORS_NO_AUG:-1}"
LORS_DISABLED_WANDB="${LORS_DISABLED_WANDB:-True}"
LORS_PIX_INIT="${LORS_PIX_INIT:-real}"
LORS_TXT_INIT="${LORS_TXT_INIT:-real}"
LORS_RUN_TAG="${LORS_RUN_TAG:-}"
LORS_FORCE_REDISTILL="${LORS_FORCE_REDISTILL:-0}"
LORS_MAX_FILES="${LORS_MAX_FILES:-}"
LORS_MAX_EXPERTS="${LORS_MAX_EXPERTS:-}"
LORS_RUN_EVALUATE="${LORS_RUN_EVALUATE:-1}"

IMAGE_ROOT="$(get_image_root "${LORS_DATASET}")"
MODEL_TAG="$(sanitize_component "${LORS_IMAGE_ENCODER}")_$(sanitize_component "${LORS_TEXT_ENCODER}")"
LOSS_TAG="${LORS_LOSS_TYPE}"
if [[ "${LORS_NO_AUG}" == "1" ]]; then
  LOSS_TAG="${LOSS_TAG}_NoAug"
fi
BUFFER_LEAF_DIR="${LORS_BUFFER_ROOT}/${LORS_DATASET}/${MODEL_TAG}/${LOSS_TAG}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_NAME_DEFAULT="lors_${LORS_DATASET}_${MODEL_TAG}_${LOSS_TAG}_${RUN_TIMESTAMP}"
LORS_RUN_NAME="${LORS_RUN_NAME:-${RUN_NAME_DEFAULT}}"
RUN_TAG_SUFFIX=""
if [[ -n "${LORS_RUN_TAG}" ]]; then
  RUN_TAG_SUFFIX="_$(sanitize_component "${LORS_RUN_TAG}")"
fi
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/lors_baseline_${LORS_DATASET}${RUN_TAG_SUFFIX}_${RUN_TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}" "${LORS_BUFFER_ROOT}" "${LORS_LOG_ROOT}"

require_checkpoint_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Required checkpoint path does not exist: ${path}" >&2
    exit 1
  fi
}

normalize_checkpoint_path() {
  local raw_path="$1"
  if [[ -z "${raw_path}" ]]; then
    return 1
  fi
  if [[ "${raw_path}" = ./* ]]; then
    echo "${PROJECT_ROOT}/${raw_path#./}"
  elif [[ "${raw_path}" = /* ]]; then
    echo "${raw_path}"
  else
    echo "${PROJECT_ROOT}/${raw_path}"
  fi
}

extract_checkpoint_dir_from_log() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    return 1
  fi
  python - "${log_path}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"Saving to (.+)", text)
if matches:
    print(matches[-1].strip())
PY
}

find_existing_checkpoint_from_logs() {
  local iteration="$1"
  local candidate_dir
  for candidate_dir in $(find "${EXPERIMENT_LOG_ROOT}" -maxdepth 1 -type d -name "lors_baseline_${LORS_DATASET}${RUN_TAG_SUFFIX}_*" | sort -r); do
    if [[ "${candidate_dir}" == "${RUN_LOG_DIR}" ]]; then
      continue
    fi
    local distill_log="${candidate_dir}/distill.log"
    local saved_dir_raw
    saved_dir_raw="$(extract_checkpoint_dir_from_log "${distill_log}" || true)"
    if [[ -z "${saved_dir_raw}" ]]; then
      continue
    fi
    local saved_dir
    saved_dir="$(normalize_checkpoint_path "${saved_dir_raw}")"
    local ckpt="${saved_dir}/distilled_${iteration}.pt"
    if [[ -f "${ckpt}" ]]; then
      echo "${ckpt}"
      return 0
    fi
  done
  return 1
}

find_latest_distilled_checkpoint() {
  local dataset="$1"
  local iteration="$2"
  local current_log_candidate
  current_log_candidate="$(extract_checkpoint_dir_from_log "${RUN_LOG_DIR}/distill.log" || true)"
  if [[ -n "${current_log_candidate}" ]]; then
    local normalized_current
    normalized_current="$(normalize_checkpoint_path "${current_log_candidate}")"
    if [[ -f "${normalized_current}/distilled_${iteration}.pt" ]]; then
      echo "${normalized_current}/distilled_${iteration}.pt"
      return 0
    fi
  fi

  local exact_candidate="${LORS_LOG_ROOT}/${dataset}/${LORS_RUN_NAME}/distilled_${iteration}.pt"
  if [[ -f "${exact_candidate}" ]]; then
    echo "${exact_candidate}"
    return 0
  fi

  local existing_from_logs
  existing_from_logs="$(find_existing_checkpoint_from_logs "${iteration}" || true)"
  if [[ -n "${existing_from_logs}" ]]; then
    echo "${existing_from_logs}"
    return 0
  fi

  local search_root
  for search_root in "${LORS_LOG_ROOT}/${dataset}" "${PROJECT_ROOT}/logged_files/${dataset}"; do
    if [[ ! -d "${search_root}" ]]; then
      continue
    fi
    local found
    found="$(find "${search_root}" -type f -name "distilled_${iteration}.pt" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -n "${found}" ]]; then
      echo "${found}"
      return 0
    fi
  done
  return 1
}

run_buffer_stage() {
  local log_path="${RUN_LOG_DIR}/buffer.log"
  if [[ "${LORS_FORCE_REBUILD_BUFFER}" != "1" ]] && compgen -G "${BUFFER_LEAF_DIR}/img_replay_buffer_*.pt" > /dev/null && compgen -G "${BUFFER_LEAF_DIR}/txt_replay_buffer_*.pt" > /dev/null; then
    stage_log "Skip buffer: existing replay buffers found in ${BUFFER_LEAF_DIR}"
    return 0
  fi

  stage_log "LoRS buffer start: dataset=${LORS_DATASET} model=${MODEL_TAG} loss=${LORS_LOSS_TYPE}"
  buffer_extra_args=()
  if [[ "${LORS_NO_AUG}" == "1" ]]; then
    buffer_extra_args+=(--no_aug)
  fi

  python "${PROJECT_ROOT}/buffer.py" \
    --dataset "${LORS_DATASET}" \
    --image_root "${IMAGE_ROOT}" \
    --ann_root "${ANN_ROOT}" \
    --model_checkpoint_root "${LORS_MODEL_CHECKPOINT_ROOT}" \
    --buffer_path "${LORS_BUFFER_ROOT}" \
    --image_encoder "${LORS_IMAGE_ENCODER}" \
    --text_encoder "${LORS_TEXT_ENCODER}" \
    --loss_type "${LORS_LOSS_TYPE}" \
    --num_experts "${LORS_NUM_EXPERTS}" \
    --train_epochs "${LORS_TRAIN_EPOCHS}" \
    --eval_freq "${LORS_EVAL_FREQ}" \
    --batch_size_train "${LORS_BATCH_TRAIN}" \
    --batch_size_test "${LORS_BATCH_TEST}" \
    --disabled_wandb "${LORS_DISABLED_WANDB}" \
    "${buffer_extra_args[@]}" \
    > "${log_path}" 2>&1
  stage_log "LoRS buffer done: ${BUFFER_LEAF_DIR}"
}

run_distill_stage() {
  local log_path="${RUN_LOG_DIR}/distill.log"
  if [[ "${LORS_FORCE_REDISTILL}" != "1" ]]; then
    local existing_ckpt
    existing_ckpt="$(find_existing_checkpoint_from_logs "${LORS_ITERATION}" || true)"
    if [[ -n "${existing_ckpt}" ]]; then
      stage_log "Skip distill: existing checkpoint found at ${existing_ckpt}"
      printf 'Reusing existing distilled checkpoint: %s\n' "${existing_ckpt}" > "${log_path}"
      return 0
    fi
  fi

  stage_log "LoRS distill start: run_name=${LORS_RUN_NAME}"
  distill_extra_args=()
  if [[ "${LORS_NO_AUG}" == "1" ]]; then
    distill_extra_args+=(--no_aug)
  fi
  if [[ -n "${LORS_MAX_FILES}" ]]; then
    distill_extra_args+=(--max_files "${LORS_MAX_FILES}")
  fi
  if [[ -n "${LORS_MAX_EXPERTS}" ]]; then
    distill_extra_args+=(--max_experts "${LORS_MAX_EXPERTS}")
  fi

  python "${PROJECT_ROOT}/distill_tesla_lors.py" \
    --dataset "${LORS_DATASET}" \
    --image_root "${IMAGE_ROOT}" \
    --ann_root "${ANN_ROOT}" \
    --model_checkpoint_root "${LORS_MODEL_CHECKPOINT_ROOT}" \
    --buffer_path "${BUFFER_LEAF_DIR}" \
    --image_encoder "${LORS_IMAGE_ENCODER}" \
    --text_encoder "${LORS_TEXT_ENCODER}" \
    --loss_type "${LORS_LOSS_TYPE}" \
    --num_queries "${LORS_NUM_QUERIES}" \
    --mini_batch_size "${LORS_MINI_BATCH_SIZE}" \
    --Iteration "${LORS_ITERATION}" \
    --eval_it "${LORS_EVAL_IT}" \
    --num_eval "${LORS_NUM_EVAL}" \
    --epoch_eval_train "${LORS_EPOCH_EVAL_TRAIN}" \
    --expert_epochs "${LORS_EXPERT_EPOCHS}" \
    --syn_steps "${LORS_SYN_STEPS}" \
    --max_start_epoch "${LORS_MAX_START_EPOCH}" \
    --batch_train "${LORS_BATCH_TRAIN}" \
    --batch_size_train "${LORS_BATCH_TRAIN}" \
    --batch_size_test "${LORS_BATCH_TEST}" \
    --pix_init "${LORS_PIX_INIT}" \
    --txt_init "${LORS_TXT_INIT}" \
    --sim_type "${LORS_SIM_TYPE}" \
    --name "${LORS_RUN_NAME}" \
    --disabled_wandb "${LORS_DISABLED_WANDB}" \
    "${distill_extra_args[@]}" \
    > "${log_path}" 2>&1
  stage_log "LoRS distill done"
}

run_evaluate_stage() {
  local ckpt_path="$1"
  local log_path="${RUN_LOG_DIR}/evaluate.log"
  stage_log "LoRS evaluate start: ckpt=${ckpt_path}"
  eval_extra_args=()
  if [[ "${LORS_NO_AUG}" == "1" ]]; then
    eval_extra_args+=(--no_aug)
  fi

  python "${PROJECT_ROOT}/evaluate_only.py" \
    --dataset "${LORS_DATASET}" \
    --image_root "${IMAGE_ROOT}" \
    --ann_root "${ANN_ROOT}" \
    --model_checkpoint_root "${LORS_MODEL_CHECKPOINT_ROOT}" \
    --image_encoder "${LORS_IMAGE_ENCODER}" \
    --text_encoder "${LORS_TEXT_ENCODER}" \
    --loss_type "${LORS_LOSS_TYPE}" \
    --ckpt_path "${ckpt_path}" \
    --num_eval "${LORS_NUM_EVAL}" \
    --batch_train "${LORS_BATCH_TRAIN}" \
    --batch_size_train "${LORS_BATCH_TRAIN}" \
    --batch_size_test "${LORS_BATCH_TEST}" \
    --disabled_wandb "${LORS_DISABLED_WANDB}" \
    "${eval_extra_args[@]}" \
    > "${log_path}" 2>&1
  stage_log "LoRS evaluate done"
}

stage_log "LoRS baseline pipeline start: dataset=${LORS_DATASET} model=${MODEL_TAG} loss=${LORS_LOSS_TYPE}"
require_checkpoint_file "${LORS_MODEL_CHECKPOINT_ROOT}/bert-base-uncased"
if [[ "${LORS_IMAGE_ENCODER}" == "nfnet" ]]; then
  require_checkpoint_file "${LORS_MODEL_CHECKPOINT_ROOT}/nfnet_l0_ra2-45c6688d.pth"
fi

run_buffer_stage
run_distill_stage

LATEST_CKPT="$(find_latest_distilled_checkpoint "${LORS_DATASET}" "${LORS_ITERATION}" || true)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "Unable to locate distilled_${LORS_ITERATION}.pt under ${LORS_LOG_ROOT}/${LORS_DATASET}" >&2
  echo "Check distill log: ${RUN_LOG_DIR}/distill.log" >&2
  exit 1
fi

if [[ "${LORS_RUN_EVALUATE}" == "1" ]]; then
  run_evaluate_stage "${LATEST_CKPT}"
else
  stage_log "Skip LoRS evaluate stage because LORS_RUN_EVALUATE=0"
fi

stage_log "LoRS baseline pipeline completed"
stage_log "Buffer dir: ${BUFFER_LEAF_DIR}"
stage_log "Checkpoint: ${LATEST_CKPT}"
stage_log "Logs: ${RUN_LOG_DIR}"
