#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${ARCH5_DATASET:-flickr}"
RATIO="${ARCH5_RATIO:-0.05}"
RATIO_TAG="$(python - "${RATIO}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
)"
SOURCE_BACKBONE="${ARCH5_SOURCE_BACKBONE:-nfnet}"
TEXT_ENCODER="${ARCH5_TEXT_ENCODER:-bert}"
EVAL_BACKBONES="${ARCH5_EVAL_BACKBONES:-nfnet resnet50 vit_b16}"
SEEDS_STR="${ARCH5_SEEDS:-0}"
read -r -a SEEDS <<< "${SEEDS_STR}"

METHODS="${ARCH5_METHODS:-ours random repblend}"
OUTPUT_ROOT="${ARCH5_OUTPUT_ROOT:-artifacts/arch_bias_energy_5pct_rerun}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_TAG="${ARCH5_RUN_TAG:-${DATASET}_${RATIO_TAG}_${RUN_TIMESTAMP}}"
REPORT_DIR="${ARCH5_REPORT_DIR:-${OUTPUT_ROOT}/reports/${RUN_TAG}}"
LOG_DIR="${ARCH5_LOG_DIR:-${OUTPUT_ROOT}/logs/${RUN_TAG}}"
MEASURE_DIR="${ARCH5_MEASURE_DIR:-${OUTPUT_ROOT}/measurements/${RUN_TAG}}"
MANIFEST_PATH="${REPORT_DIR}/manifest.jsonl"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "${MEASURE_DIR}"
: > "${MANIFEST_PATH}"

MODEL_TAG="$(sanitize_component "${SOURCE_BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
IMAGE_ROOT="$(get_image_root "${DATASET}")"

OURS_FEATURE_CACHE_ROOT="${ARCH5_OURS_FEATURE_CACHE_ROOT:-${OUTPUT_ROOT}/feature_cache_dense_sift_bovw/${RUN_TAG}}"
OURS_TOPOLOGY_ROOT="${ARCH5_OURS_TOPOLOGY_ROOT:-${OUTPUT_ROOT}/topology_graph_dense_sift_bovw/${RUN_TAG}}"
OURS_CROSS_ROOT="${ARCH5_OURS_CROSS_ROOT:-${OUTPUT_ROOT}/cross_modal_topology_dense_sift_bovw/${RUN_TAG}}"
OURS_SELECTION_ROOT="${ARCH5_OURS_SELECTION_ROOT:-${OUTPUT_ROOT}/subset_selection_dense_sift_bovw/${RUN_TAG}}"
OURS_TRAIN_ROOT="${ARCH5_OURS_TRAIN_ROOT:-${OUTPUT_ROOT}/subset_train_ours_crossarch/${RUN_TAG}}"
RANDOM_SELECTION_ROOT="${ARCH5_RANDOM_SELECTION_ROOT:-${OUTPUT_ROOT}/subset_selection_random/${RUN_TAG}}"
RANDOM_TRAIN_ROOT="${ARCH5_RANDOM_TRAIN_ROOT:-${OUTPUT_ROOT}/subset_train_random_crossarch/${RUN_TAG}}"

REPBLEND_ROOT="${REPBLEND_ROOT:-${PROJECT_ROOT}/RepBlend}"
REPBLEND_BUFFER_ROOT="${REPBLEND_BUFFER_ROOT:-${REPBLEND_ROOT}/buffer}"
REPBLEND_ITERATION="${REPBLEND_ITERATION:-3000}"

ENERGY_PREFER_ZEUS="${ENERGY_PREFER_ZEUS:-1}"
ENERGY_GPU_SAMPLER_INTERVAL="${ENERGY_GPU_SAMPLER_INTERVAL:-1.0}"
GPU_COUNT="${ARCH5_GPU_COUNT:-$(python - <<PY
devices = "${CUDA_VISIBLE_DEVICES:-}".strip()
print(max(len([x for x in devices.split(",") if x.strip()]), 1))
PY
)}"
ARCH5_SELECTION_USE_TORCHRUN="${ARCH5_SELECTION_USE_TORCHRUN:-$(python - <<PY
print(1 if int("${GPU_COUNT}") > 1 else 0)
PY
)}"
ARCH5_SELECTION_NPROC_PER_NODE="${ARCH5_SELECTION_NPROC_PER_NODE:-${GPU_COUNT}}"
ARCH5_ENABLE_TRAIN_DATA_PARALLEL="${ARCH5_ENABLE_TRAIN_DATA_PARALLEL:-$(python - <<PY
print(1 if int("${GPU_COUNT}") > 1 else 0)
PY
)}"
ARCH5_TRAIN_DP_DEVICE_IDS="${ARCH5_TRAIN_DP_DEVICE_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
REPBLEND_CUDA_VISIBLE_DEVICES="${REPBLEND_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

method_enabled() {
  local name="$1"
  [[ " ${METHODS} " == *" ${name} "* ]]
}

append_manifest() {
  python - "$MANIFEST_PATH" "$@" <<'PY'
import json
import sys
path = sys.argv[1]
keys = sys.argv[2::2]
values = sys.argv[3::2]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(dict(zip(keys, values)), ensure_ascii=False) + "\n")
PY
}

measure_command() {
  local label="$1"
  local measurement_path="$2"
  local log_path="$3"
  local working_dir="$4"
  shift 4
  local zeus_args=()
  if [[ "${ENERGY_PREFER_ZEUS}" == "1" ]]; then
    zeus_args+=(--prefer_zeus)
  fi
  if [[ "${ARCH5_TEE_PROGRESS:-0}" == "1" ]]; then
    python "${PROJECT_ROOT}/tools/measure_command_energy.py" \
      --label "${label}" \
      --output_json "${measurement_path}" \
      --working_dir "${working_dir}" \
      --gpu_sampler_interval "${ENERGY_GPU_SAMPLER_INTERVAL}" \
      --tee_log "${log_path}" \
      "${zeus_args[@]}" \
      -- "$@"
  else
    python "${PROJECT_ROOT}/tools/measure_command_energy.py" \
      --label "${label}" \
      --output_json "${measurement_path}" \
      --working_dir "${working_dir}" \
      --gpu_sampler_interval "${ENERGY_GPU_SAMPLER_INTERVAL}" \
      "${zeus_args[@]}" \
      -- "$@" > "${log_path}" 2>&1
  fi
}

compute_train_size() {
  python - "${DATASET}" "${IMAGE_ROOT}" "${ANN_ROOT}" <<'PY'
import sys
from types import SimpleNamespace
from data import create_dataset
args = SimpleNamespace(
    dataset=sys.argv[1],
    image_root=sys.argv[2],
    ann_root=sys.argv[3],
    image_size=224,
    no_aug=True,
    return_sample_idx=False,
)
train_dataset, _, _ = create_dataset(args)
print(len(train_dataset))
PY
}

ratio_to_count() {
  local total_count="$1"
  local ratio="$2"
  python - "${total_count}" "${ratio}" <<'PY'
import sys
total = int(sys.argv[1])
ratio = float(sys.argv[2])
print(max(1, int(round(total * ratio))))
PY
}

run_ours_selection() {
  local seed="$1"
  local log_path="${LOG_DIR}/ours_${RATIO_TAG}_seed${seed}_selection.log"
  local measurement_path="${MEASURE_DIR}/ours_${RATIO_TAG}_seed${seed}_selection.json"
  local selected_indices_path="${OURS_SELECTION_ROOT}/${DATASET}/train/${MODEL_TAG}/${RATIO_TAG}/proxy_opt_lsrc/seed_${seed}/selected_indices.json"

  if [[ -f "${selected_indices_path}" ]]; then
    stage_log "Skip Ours selection: existing selected_indices found at ${selected_indices_path}"
    append_manifest method "ours" dataset "${DATASET}" budget_type "ratio" budget_value "${RATIO}" budget_tag "${RATIO_TAG}" \
      eval_backbone "" seed "${seed}" stage "selection" gpu_count "${GPU_COUNT}" \
      selected_indices_path "${selected_indices_path}" log_path "${log_path}" measurement_path "${measurement_path}" skipped "1"
    return 0
  fi

  stage_log "Measure Ours full selection pipeline: ${RATIO_TAG} seed=${seed}"
  measure_command "ours_${RATIO_TAG}_seed${seed}_selection" "${measurement_path}" "${log_path}" "${PROJECT_ROOT}" \
    env \
      FEATURE_CACHE_ROOT="${OURS_FEATURE_CACHE_ROOT}" \
      TOPOLOGY_ROOT="${OURS_TOPOLOGY_ROOT}" \
      WAVELET_MAIN_LATEST_CROSS_OUTPUT_ROOT="${OURS_CROSS_ROOT}" \
      WAVELET_MAIN_LATEST_SELECTION_OUTPUT_ROOT="${OURS_SELECTION_ROOT}" \
      WAVELET_MAIN_LATEST_TRAIN_OUTPUT_ROOT="${OUTPUT_ROOT}/unused_ours_train/${RUN_TAG}" \
      WAVELET_MAIN_LATEST_DATASET="${DATASET}" \
      WAVELET_MAIN_LATEST_BACKBONE="${SOURCE_BACKBONE}" \
      WAVELET_MAIN_LATEST_TEXT_ENCODER="${TEXT_ENCODER}" \
      WAVELET_MAIN_LATEST_VARIANT="wavelet_main_dense_sift_bovw_5pct_energy" \
      WAVELET_MAIN_LATEST_BUDGETS="" \
      WAVELET_MAIN_LATEST_RATIOS="${RATIO}" \
      WAVELET_MAIN_LATEST_SEEDS="${seed}" \
      WAVELET_MAIN_LATEST_RUN_SELECTION=1 \
      WAVELET_MAIN_LATEST_RUN_TRAIN=0 \
      WAVELET_MAIN_LATEST_SELECTION_USE_TORCHRUN="${ARCH5_SELECTION_USE_TORCHRUN}" \
      WAVELET_MAIN_LATEST_SELECTION_NPROC_PER_NODE="${ARCH5_SELECTION_NPROC_PER_NODE}" \
      WAVELET_MAIN_LATEST_SELECTION_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
      WAVELET_MAIN_LATEST_REPORT_NAME="ours_5pct_energy_selection" \
      SELECTION_IMAGE_REPR_METHOD="dense_sift_bovw" \
      BOVW_CODEBOOK_SIZE="${BOVW_CODEBOOK_SIZE}" \
      DENSE_SIFT_STEP="${DENSE_SIFT_STEP}" \
      DENSE_SIFT_PATCH="${DENSE_SIFT_PATCH}" \
      BOVW_MAX_FIT_DESCRIPTORS="${BOVW_MAX_FIT_DESCRIPTORS}" \
      BOVW_DESCRIPTORS_PER_IMAGE="${BOVW_DESCRIPTORS_PER_IMAGE}" \
      bash "${SCRIPT_DIR}/run_wavelet_main_latest_combo.sh"

  append_manifest method "ours" dataset "${DATASET}" budget_type "ratio" budget_value "${RATIO}" budget_tag "${RATIO_TAG}" \
    eval_backbone "" seed "${seed}" stage "selection" gpu_count "${GPU_COUNT}" \
    selected_indices_path "${selected_indices_path}" log_path "${log_path}" measurement_path "${measurement_path}" skipped "0"
}

run_random_selection() {
  local seed="$1"
  local log_path="${LOG_DIR}/random_${RATIO_TAG}_seed${seed}_selection.log"
  local measurement_path="${MEASURE_DIR}/random_${RATIO_TAG}_seed${seed}_selection.json"
  local selected_indices_path="${RANDOM_SELECTION_ROOT}/${DATASET}/train/${MODEL_TAG}/${RATIO_TAG}/random/seed_${seed}/selected_indices.json"

  stage_log "Measure Random selection: ${RATIO_TAG} seed=${seed}"
  measure_command "random_${RATIO_TAG}_seed${seed}_selection" "${measurement_path}" "${log_path}" "${PROJECT_ROOT}" \
    python "${PROJECT_ROOT}/run_random_subset_selection.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${SOURCE_BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --feature_cache_root "${OURS_FEATURE_CACHE_ROOT}" \
      --output_root "${RANDOM_SELECTION_ROOT}" \
      --budget_ratio "${RATIO}" \
      --selection_method random \
      --random_state "${seed}"

  append_manifest method "random" dataset "${DATASET}" budget_type "ratio" budget_value "${RATIO}" budget_tag "${RATIO_TAG}" \
    eval_backbone "" seed "${seed}" stage "selection" gpu_count "${GPU_COUNT}" \
    selected_indices_path "${selected_indices_path}" log_path "${log_path}" measurement_path "${measurement_path}" skipped "0"
}

run_real_training() {
  local method="$1"
  local selected_indices_path="$2"
  local train_root="$3"
  local subset_tag="$4"
  local eval_backbone="$5"
  local seed="$6"
  local eval_model_tag
  eval_model_tag="$(sanitize_component "${eval_backbone}")_$(sanitize_component "${TEXT_ENCODER}")"
  local metrics_path="${train_root}/${DATASET}/${eval_model_tag}/${RATIO_TAG}/${subset_tag}/seed_${seed}/metrics.json"
  local log_path="${LOG_DIR}/${method}_${RATIO_TAG}_${eval_backbone}_seed${seed}_train.log"
  local measurement_path="${MEASURE_DIR}/${method}_${RATIO_TAG}_${eval_backbone}_seed${seed}_train.json"
  local train_extra=()
  if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
    train_extra+=(--no_aug)
  fi
  if [[ "${ARCH5_ENABLE_TRAIN_DATA_PARALLEL}" == "1" ]]; then
    train_extra+=(--enable_image_encoder_data_parallel)
    if [[ -n "${ARCH5_TRAIN_DP_DEVICE_IDS}" ]]; then
      train_extra+=(--image_encoder_data_parallel_device_ids "${ARCH5_TRAIN_DP_DEVICE_IDS}")
    fi
  fi

  stage_log "Measure ${method} training/eval: ${RATIO_TAG} eval_backbone=${eval_backbone} seed=${seed}"
  measure_command "${method}_${RATIO_TAG}_${eval_backbone}_seed${seed}_train" "${measurement_path}" "${log_path}" "${PROJECT_ROOT}" \
    python "${PROJECT_ROOT}/run_subset_train.py" \
      --dataset "${DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --selected_indices_path "${selected_indices_path}" \
      --subset_ratio "${RATIO}" \
      --subset_tag "${subset_tag}" \
      --image_encoder "${eval_backbone}" \
      --text_encoder "${TEXT_ENCODER}" \
      --output_root "${train_root}" \
      --batch_size_train "${BATCH_TRAIN}" \
      --batch_size_test "${BATCH_TEST}" \
      --text_batch_size "${TEXT_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      "${train_extra[@]}"

  append_manifest method "${method}" dataset "${DATASET}" budget_type "ratio" budget_value "${RATIO}" budget_tag "${RATIO_TAG}" \
    eval_backbone "${eval_backbone}" seed "${seed}" stage "training_eval" gpu_count "${GPU_COUNT}" \
    selected_indices_path "${selected_indices_path}" metrics_path "${metrics_path}" log_path "${log_path}" measurement_path "${measurement_path}" skipped "0"
}

repblend_params_for_count() {
  local count="$1"
  REPBLEND_NUM_QUERIES="$((count - 1))"
  REPBLEND_EXTRA_ARGS=(
    --syn_steps "${REPBLEND_SYN_STEPS:-8}"
    --expert_epochs "${REPBLEND_EXPERT_EPOCHS:-1}"
    --max_start_epoch "${REPBLEND_MAX_START_EPOCH:-2}"
    --lr_img "${REPBLEND_LR_IMG:-1000}"
    --lr_txt "${REPBLEND_LR_TXT:-1000}"
    --lr_lr "${REPBLEND_LR_LR:-1e-2}"
    --lr_sim "${REPBLEND_LR_SIM:-10.0}"
    --sim_type "${REPBLEND_SIM_TYPE:-lowrank}"
    --sim_rank "${REPBLEND_SIM_RANK:-20}"
    --alpha "${REPBLEND_ALPHA:-1.0}"
    --mini_batch_size "${REPBLEND_MINI_BATCH_SIZE:-20}"
    --Iteration "${REPBLEND_ITERATION}"
    --loss_type "${REPBLEND_LOSS_TYPE:-WBCE}"
    --epoch_eval_train "${REPBLEND_EPOCH_EVAL_TRAIN:-100}"
  )
  if [[ "${DATASET}" == "coco" ]]; then
    REPBLEND_EXTRA_ARGS+=(--merge_loss_branches)
  fi
}

extract_repblend_checkpoint() {
  local log_path="$1"
  python - "${log_path}" "${REPBLEND_ROOT}" "${REPBLEND_ITERATION}" <<'PY'
import re
import sys
from pathlib import Path
log_path = Path(sys.argv[1])
root = Path(sys.argv[2])
iteration = sys.argv[3]
text = log_path.read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"Saving to (.+)", text)
if not matches:
    raise SystemExit(1)
save_dir = Path(matches[-1].strip())
if not save_dir.is_absolute():
    save_dir = root / save_dir
print(save_dir / f"distilled_{iteration}.pt")
PY
}

run_repblend() {
  local train_count
  local budget_count
  train_count="$(compute_train_size)"
  budget_count="$(ratio_to_count "${train_count}" "${RATIO}")"
  repblend_params_for_count "${budget_count}"

  local run_name="repblend_${DATASET}_${RATIO_TAG}_${RUN_TIMESTAMP}"
  local buffer_dir="${REPBLEND_BUFFER_ROOT}/${DATASET}/${MODEL_TAG}/InfoNCE"
  local buffer_log="${LOG_DIR}/repblend_${RATIO_TAG}_buffer.log"
  local buffer_measure="${MEASURE_DIR}/repblend_${RATIO_TAG}_buffer.json"
  local distill_log="${LOG_DIR}/repblend_${RATIO_TAG}_distill.log"
  local distill_measure="${MEASURE_DIR}/repblend_${RATIO_TAG}_distill.json"

  if compgen -G "${buffer_dir}/img_replay_buffer_*.pt" >/dev/null && compgen -G "${buffer_dir}/txt_replay_buffer_*.pt" >/dev/null; then
    stage_log "Skip RepBlend buffer generation: existing replay buffers found at ${buffer_dir}"
  else
    stage_log "Measure RepBlend buffer generation as selection setup: source=${MODEL_TAG}"
    measure_command "repblend_${RATIO_TAG}_buffer" "${buffer_measure}" "${buffer_log}" "${REPBLEND_ROOT}" \
      env CUDA_VISIBLE_DEVICES="${REPBLEND_CUDA_VISIBLE_DEVICES}" WANDB_MODE=disabled python buffer.py \
        --dataset "${DATASET}" \
        --buffer_path "${REPBLEND_BUFFER_ROOT}" \
        --image_root "${IMAGE_ROOT}" \
        --ann_root "${ANN_ROOT}" \
        --image_encoder "${SOURCE_BACKBONE}" \
        --text_encoder "${TEXT_ENCODER}" \
        --loss_type InfoNCE \
        --num_experts "${REPBLEND_BUFFER_NUM_EXPERTS:-20}" \
        --train_epochs "${REPBLEND_BUFFER_TRAIN_EPOCHS:-10}" \
        --eval_freq "${REPBLEND_BUFFER_EVAL_FREQ:-5}" \
        --batch_train "${REPBLEND_BUFFER_BATCH_TRAIN:-128}" \
        --batch_size_train "${REPBLEND_BUFFER_BATCH_TRAIN:-128}" \
        --batch_size_test "${REPBLEND_BUFFER_BATCH_TEST:-128}" \
        --disabled_wandb True
    append_manifest method "repblend" dataset "${DATASET}" budget_type "ratio" budget_value "${RATIO}" budget_tag "${RATIO_TAG}" \
      eval_backbone "" seed "0" stage "selection_buffer" gpu_count "${GPU_COUNT}" \
      log_path "${buffer_log}" measurement_path "${buffer_measure}" skipped "0"
  fi

  stage_log "Measure RepBlend distillation as selection: ${RATIO_TAG} count=${budget_count} num_queries=${REPBLEND_NUM_QUERIES}"
  measure_command "repblend_${RATIO_TAG}_distill" "${distill_measure}" "${distill_log}" "${REPBLEND_ROOT}" \
    env CUDA_VISIBLE_DEVICES="${REPBLEND_CUDA_VISIBLE_DEVICES}" WANDB_MODE=disabled python distill_repblend.py \
      --dataset "${DATASET}" \
      --buffer_path "${buffer_dir}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --image_encoder "${SOURCE_BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --lr_teacher_img 0.1 \
      --lr_teacher_txt 0.1 \
      --num_queries "${REPBLEND_NUM_QUERIES}" \
      --name "${run_name}" \
      --disabled_wandb True \
      "${REPBLEND_EXTRA_ARGS[@]}"

  local checkpoint_path
  checkpoint_path="$(extract_repblend_checkpoint "${distill_log}")"
  append_manifest method "repblend" dataset "${DATASET}" budget_type "ratio" budget_value "${RATIO}" budget_tag "${RATIO_TAG}" \
    eval_backbone "" seed "0" stage "distill_selection" gpu_count "${GPU_COUNT}" checkpoint_path "${checkpoint_path}" \
    log_path "${distill_log}" measurement_path "${distill_measure}" skipped "0"

  for eval_backbone in ${EVAL_BACKBONES}; do
    local eval_log="${LOG_DIR}/repblend_${RATIO_TAG}_${eval_backbone}_evaluate.log"
    local eval_measure="${MEASURE_DIR}/repblend_${RATIO_TAG}_${eval_backbone}_evaluate.json"
    stage_log "Measure RepBlend training/eval: ${RATIO_TAG} eval_backbone=${eval_backbone}"
    measure_command "repblend_${RATIO_TAG}_${eval_backbone}_evaluate" "${eval_measure}" "${eval_log}" "${PROJECT_ROOT}" \
      env CUDA_VISIBLE_DEVICES="${REPBLEND_CUDA_VISIBLE_DEVICES}" python "${PROJECT_ROOT}/evaluate_only.py" \
        --dataset "${DATASET}" \
        --image_root "${IMAGE_ROOT}" \
        --ann_root "${ANN_ROOT}" \
        --model_checkpoint_root "${REPBLEND_MODEL_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}" \
        --image_encoder "${eval_backbone}" \
        --text_encoder "${TEXT_ENCODER}" \
        --loss_type WBCE \
        --ckpt_path "${checkpoint_path}" \
        --num_eval "${REPBLEND_NUM_EVAL:-1}" \
        --batch_train "${REPBLEND_EVAL_BATCH_TRAIN:-128}" \
        --batch_size_train "${REPBLEND_EVAL_BATCH_TRAIN:-128}" \
        --batch_size_test "${REPBLEND_EVAL_BATCH_TEST:-128}" \
        --disabled_wandb True \
        --no_aug
    append_manifest method "repblend" dataset "${DATASET}" budget_type "ratio" budget_value "${RATIO}" budget_tag "${RATIO_TAG}" \
      eval_backbone "${eval_backbone}" seed "0" stage "training_eval" gpu_count "${GPU_COUNT}" checkpoint_path "${checkpoint_path}" \
      evaluate_log "${eval_log}" log_path "${eval_log}" measurement_path "${eval_measure}" skipped "0"
  done
}

stage_log "Architecture-bias + energy rerun start"
stage_log "  methods=${METHODS}"
stage_log "  dataset=${DATASET} ratio=${RATIO} tag=${RATIO_TAG}"
stage_log "  source=${MODEL_TAG} eval_backbones=${EVAL_BACKBONES}"
stage_log "  output=${OUTPUT_ROOT}/${RUN_TAG}"
stage_log "  multi-gpu: visible=${CUDA_VISIBLE_DEVICES:-<unset>} gpu_count=${GPU_COUNT} ours_selection_torchrun=${ARCH5_SELECTION_USE_TORCHRUN} nproc=${ARCH5_SELECTION_NPROC_PER_NODE} train_dp=${ARCH5_ENABLE_TRAIN_DATA_PARALLEL}"
stage_log "  repblend_visible=${REPBLEND_CUDA_VISIBLE_DEVICES} note=RepBlend upstream distillation is single-process; CUDA_VISIBLE_DEVICES controls placement, not DDP."
stage_log "  energy: GPU=Zeus preferred with nvidia-smi fallback, CPU=Intel RAPL"

if method_enabled ours; then
  for seed in "${SEEDS[@]}"; do
    run_ours_selection "${seed}"
    ours_indices="${OURS_SELECTION_ROOT}/${DATASET}/train/${MODEL_TAG}/${RATIO_TAG}/proxy_opt_lsrc/seed_${seed}/selected_indices.json"
    for eval_backbone in ${EVAL_BACKBONES}; do
      run_real_training "ours" "${ours_indices}" "${OURS_TRAIN_ROOT}" "ours_5pct_energy" "${eval_backbone}" "${seed}"
    done
  done
fi

if method_enabled random; then
  for seed in "${SEEDS[@]}"; do
    run_random_selection "${seed}"
    random_indices="${RANDOM_SELECTION_ROOT}/${DATASET}/train/${MODEL_TAG}/${RATIO_TAG}/random/seed_${seed}/selected_indices.json"
    for eval_backbone in ${EVAL_BACKBONES}; do
      run_real_training "random" "${random_indices}" "${RANDOM_TRAIN_ROOT}" "random_5pct_energy" "${eval_backbone}" "${seed}"
    done
  done
fi

if method_enabled repblend; then
  run_repblend
fi

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${MANIFEST_PATH}" \
  --output_dir "${REPORT_DIR}"

stage_log "5% architecture-bias + energy rerun done"
stage_log "  architecture_bias=${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${REPORT_DIR}/energy_efficiency.csv"
stage_log "  detail=${REPORT_DIR}/supplemental_detail.csv"
stage_log "  logs=${LOG_DIR}"
stage_log "  measurements=${MEASURE_DIR}"
