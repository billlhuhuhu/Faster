#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${ARCH_ENERGY_DATASET:-flickr}"
TEXT_ENCODER="${ARCH_ENERGY_TEXT_ENCODER:-bert}"
SOURCE_BACKBONE="${ARCH_ENERGY_SOURCE_BACKBONE:-nfnet}"
EVAL_BACKBONES="${ARCH_ENERGY_EVAL_BACKBONES:-resnet50 vit_b16}"
METHODS="${ARCH_ENERGY_METHODS:-ours random repblend}"
BUDGETS_STR="${ARCH_ENERGY_BUDGETS:-100 200 500}"
RATIOS_STR="${ARCH_ENERGY_RATIOS:-}"
SEEDS_STR="${ARCH_ENERGY_SEEDS:-0}"
read -r -a BUDGETS <<< "${BUDGETS_STR}"
read -r -a RATIOS <<< "${RATIOS_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"

SOURCE_MODEL_TAG="$(sanitize_component "${SOURCE_BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
IMAGE_ROOT="$(get_image_root "${DATASET}")"

OUTPUT_ROOT="${ARCH_ENERGY_OUTPUT_ROOT:-artifacts/arch_bias_energy}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
REPORT_DIR="${ARCH_ENERGY_REPORT_DIR:-${OUTPUT_ROOT}/reports/${DATASET}_${RUN_TIMESTAMP}}"
LOG_DIR="${ARCH_ENERGY_LOG_DIR:-${OUTPUT_ROOT}/logs/${DATASET}_${RUN_TIMESTAMP}}"
MEASURE_DIR="${ARCH_ENERGY_MEASURE_DIR:-${OUTPUT_ROOT}/measurements/${DATASET}_${RUN_TIMESTAMP}}"
MANIFEST_PATH="${REPORT_DIR}/manifest.jsonl"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "${MEASURE_DIR}"
: > "${MANIFEST_PATH}"

OURS_SELECTION_ROOT="${ARCH_ENERGY_OURS_SELECTION_ROOT:-artifacts/subset_selection_dense_sift_bovw}"
RANDOM_SELECTION_ROOT="${ARCH_ENERGY_RANDOM_SELECTION_ROOT:-artifacts/subset_selection_random_baseline}"
OURS_TRAIN_ROOT="${ARCH_ENERGY_OURS_TRAIN_ROOT:-${OUTPUT_ROOT}/subset_train_ours_crossarch}"
RANDOM_TRAIN_ROOT="${ARCH_ENERGY_RANDOM_TRAIN_ROOT:-${OUTPUT_ROOT}/subset_train_random_crossarch}"
OURS_SELECTION_TAG="${ARCH_ENERGY_OURS_SELECTION_TAG:-proxy_opt_lsrc}"
RANDOM_SELECTION_TAG="${ARCH_ENERGY_RANDOM_SELECTION_TAG:-random}"

REPBLEND_ROOT="${REPBLEND_ROOT:-${PROJECT_ROOT}/RepBlend}"
REPBLEND_BUFFER_ROOT="${REPBLEND_BUFFER_ROOT:-${REPBLEND_ROOT}/buffer}"
REPBLEND_LOGGED_FILES_ROOT="${REPBLEND_LOGGED_FILES_ROOT:-${REPBLEND_ROOT}/logged_files}"
REPBLEND_FORCE_REDISTILL="${REPBLEND_FORCE_REDISTILL:-1}"

ENERGY_PREFER_ZEUS="${ENERGY_PREFER_ZEUS:-1}"
ENERGY_GPU_SAMPLER_INTERVAL="${ENERGY_GPU_SAMPLER_INTERVAL:-1.0}"
GPU_COUNT="${ARCH_ENERGY_GPU_COUNT:-$(python - <<PY
devices = "${CUDA_VISIBLE_DEVICES:-}".strip()
print(max(len([x for x in devices.split(",") if x.strip()]), 1))
PY
)}"

method_enabled() {
  local name="$1"
  [[ " ${METHODS} " == *" ${name} "* ]]
}

ratio_to_tag() {
  local ratio="$1"
  python - "${ratio}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
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
  python "${PROJECT_ROOT}/tools/measure_command_energy.py" \
    --label "${label}" \
    --output_json "${measurement_path}" \
    --working_dir "${working_dir}" \
    --gpu_sampler_interval "${ENERGY_GPU_SAMPLER_INTERVAL}" \
    "${zeus_args[@]}" \
    -- "$@" > "${log_path}" 2>&1
}

append_skipped_selection_row() {
  local method="$1"
  local budget_type="$2"
  local budget_value="$3"
  local budget_tag="$4"
  append_manifest \
    method "${method}" dataset "${DATASET}" budget_type "${budget_type}" budget_value "${budget_value}" budget_tag "${budget_tag}" \
    eval_backbone "" stage "selection_skipped_reused_indices" seconds "0" gpu_count "${GPU_COUNT}" skipped "1" \
    note "selected_indices were reused; selection energy cannot be reconstructed unless rerun under the energy wrapper"
}

run_real_subset_training() {
  local method="$1"
  local selection_root="$2"
  local selection_tag="$3"
  local train_root="$4"
  local subset_tag="$5"
  local budget_type="$6"
  local budget_value="$7"
  local budget_tag="$8"
  local eval_backbone="$9"
  local seed="${10}"

  local selected_indices_path="${selection_root}/${DATASET}/train/${SOURCE_MODEL_TAG}/${budget_tag}/${selection_tag}/seed_${seed}/selected_indices.json"
  local eval_model_tag
  eval_model_tag="$(sanitize_component "${eval_backbone}")_$(sanitize_component "${TEXT_ENCODER}")"
  local metrics_path="${train_root}/${DATASET}/${eval_model_tag}/${budget_tag}/${subset_tag}/seed_${seed}/metrics.json"
  local log_path="${LOG_DIR}/${method}_${budget_tag}_${eval_backbone}_seed${seed}_train.log"
  local measurement_path="${MEASURE_DIR}/${method}_${budget_tag}_${eval_backbone}_seed${seed}_train.json"

  if [[ ! -f "${selected_indices_path}" ]]; then
    stage_log "Missing selected indices, skip ${method}/${budget_tag}/${eval_backbone}: ${selected_indices_path}"
    append_manifest method "${method}" dataset "${DATASET}" budget_type "${budget_type}" budget_value "${budget_value}" budget_tag "${budget_tag}" \
      eval_backbone "${eval_backbone}" seed "${seed}" stage "missing_selected_indices" seconds "0" gpu_count "${GPU_COUNT}" \
      selected_indices_path "${selected_indices_path}" metrics_path "${metrics_path}" log_path "${log_path}" skipped "1"
    return 0
  fi

  if [[ -f "${metrics_path}" && "${ARCH_ENERGY_FORCE_RETRAIN:-0}" != "1" ]]; then
    stage_log "Skip ${method} training: existing ${metrics_path}"
    append_manifest method "${method}" dataset "${DATASET}" budget_type "${budget_type}" budget_value "${budget_value}" budget_tag "${budget_tag}" \
      eval_backbone "${eval_backbone}" seed "${seed}" stage "training_eval_skipped_existing_metrics" seconds "0" gpu_count "${GPU_COUNT}" \
      selected_indices_path "${selected_indices_path}" metrics_path "${metrics_path}" log_path "${log_path}" skipped "1"
    return 0
  fi

  local budget_args=()
  if [[ "${budget_type}" == "size" ]]; then
    budget_args+=(--subset_size "${budget_value}")
  else
    budget_args+=(--subset_ratio "${budget_value}")
  fi
  local train_extra=()
  if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
    train_extra+=(--no_aug)
  fi

  stage_log "Measure ${method} training: ${budget_tag} eval_backbone=${eval_backbone} seed=${seed}"
  measure_command "${method}_${budget_tag}_${eval_backbone}_train" "${measurement_path}" "${log_path}" "${PROJECT_ROOT}" \
    python "${PROJECT_ROOT}/run_subset_train.py" \
      --dataset "${DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --selected_indices_path "${selected_indices_path}" \
      "${budget_args[@]}" \
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

  append_manifest method "${method}" dataset "${DATASET}" budget_type "${budget_type}" budget_value "${budget_value}" budget_tag "${budget_tag}" \
    eval_backbone "${eval_backbone}" seed "${seed}" stage "training_eval" gpu_count "${GPU_COUNT}" \
    selected_indices_path "${selected_indices_path}" metrics_path "${metrics_path}" log_path "${log_path}" measurement_path "${measurement_path}" skipped "0"
}

run_real_subset_method() {
  local method="$1"
  local selection_root="$2"
  local selection_tag="$3"
  local train_root="$4"
  local subset_tag="$5"

  for budget in "${BUDGETS[@]}"; do
    local budget_tag
    budget_tag="$(format_budget_tag "${budget}")"
    append_skipped_selection_row "${method}" "size" "${budget}" "${budget_tag}"
    for eval_backbone in ${EVAL_BACKBONES}; do
      for seed in "${SEEDS[@]}"; do
        run_real_subset_training "${method}" "${selection_root}" "${selection_tag}" "${train_root}" "${subset_tag}" \
          "size" "${budget}" "${budget_tag}" "${eval_backbone}" "${seed}"
      done
    done
  done

  for ratio in "${RATIOS[@]}"; do
    [[ -z "${ratio}" ]] && continue
    local ratio_tag
    ratio_tag="$(ratio_to_tag "${ratio}")"
    append_skipped_selection_row "${method}" "ratio" "${ratio}" "${ratio_tag}"
    for eval_backbone in ${EVAL_BACKBONES}; do
      for seed in "${SEEDS[@]}"; do
        run_real_subset_training "${method}" "${selection_root}" "${selection_tag}" "${train_root}" "${subset_tag}" \
          "ratio" "${ratio}" "${ratio_tag}" "${eval_backbone}" "${seed}"
      done
    done
  done
}

repblend_params_for_budget() {
  local budget="$1"
  local dataset="$2"
  REPBLEND_NUM_QUERIES="$((budget - 1))"
  REPBLEND_EXTRA_ARGS=()
  if [[ "${dataset}" == "coco" ]]; then
    REPBLEND_EXTRA_ARGS+=(--merge_loss_branches)
  fi
  case "${dataset}:${budget}" in
    flickr:100)
      REPBLEND_EXTRA_ARGS+=(--syn_steps 8 --expert_epochs 1 --max_start_epoch 2 --lr_img 100 --lr_txt 100 --lr_lr 1e-2 --lr_sim 10.0 --sim_type lowrank --sim_rank 10 --alpha 3 --mini_batch_size 20 --Iteration 3000 --loss_type WBCE --epoch_eval_train 100)
      ;;
    flickr:200)
      REPBLEND_EXTRA_ARGS+=(--syn_steps 8 --expert_epochs 1 --max_start_epoch 2 --lr_img 1000 --lr_txt 1000 --lr_lr 1e-2 --lr_sim 10.0 --sim_type lowrank --sim_rank 5 --alpha 1.0 --mini_batch_size 20 --Iteration 3000 --loss_type WBCE --epoch_eval_train 100)
      ;;
    flickr:500)
      REPBLEND_EXTRA_ARGS+=(--syn_steps 8 --expert_epochs 1 --max_start_epoch 3 --lr_img 1000 --lr_txt 1000 --lr_lr 1e-2 --lr_sim 100 --sim_type lowrank --sim_rank 20 --alpha 0.01 --mini_batch_size 20 --Iteration 3000 --loss_type WBCE --eval_it 300 --epoch_eval_train 100)
      ;;
    coco:100)
      REPBLEND_EXTRA_ARGS+=(--syn_steps 8 --expert_epochs 1 --max_start_epoch 2 --lr_img 1000 --lr_txt 1000 --lr_lr 1e-2 --lr_sim 5.0 --sim_type lowrank --sim_rank 10 --alpha 1.0 --mini_batch_size 20 --Iteration 3000 --loss_type WBCE --epoch_eval_train 100)
      ;;
    coco:200)
      REPBLEND_EXTRA_ARGS+=(--syn_steps 8 --expert_epochs 1 --max_start_epoch 2 --lr_img 1000 --lr_txt 1000 --lr_lr 1e-2 --lr_sim 50 --sim_type lowrank --sim_rank 20 --alpha 1.0 --mini_batch_size 20 --Iteration 3000 --loss_type WBCE --epoch_eval_train 100)
      ;;
    coco:500)
      REPBLEND_EXTRA_ARGS+=(--syn_steps 8 --expert_epochs 1 --max_start_epoch 2 --lr_img 5000 --lr_txt 5000 --lr_lr 1e-2 --lr_sim 500 --sim_type lowrank --sim_rank 40 --alpha 1.0 --mini_batch_size 20 --Iteration 3000 --temperature 0.1 --no_aug --loss_type WBCE --epoch_eval_train 100)
      ;;
    *)
      REPBLEND_EXTRA_ARGS+=(--syn_steps "${REPBLEND_SYN_STEPS:-8}" --expert_epochs "${REPBLEND_EXPERT_EPOCHS:-1}" --max_start_epoch "${REPBLEND_MAX_START_EPOCH:-2}" --lr_img "${REPBLEND_LR_IMG:-1000}" --lr_txt "${REPBLEND_LR_TXT:-1000}" --lr_lr "${REPBLEND_LR_LR:-1e-2}" --lr_sim "${REPBLEND_LR_SIM:-10.0}" --sim_type "${REPBLEND_SIM_TYPE:-lowrank}" --sim_rank "${REPBLEND_SIM_RANK:-10}" --alpha "${REPBLEND_ALPHA:-1.0}" --mini_batch_size "${REPBLEND_MINI_BATCH_SIZE:-20}" --Iteration "${REPBLEND_ITERATION:-3000}" --loss_type "${REPBLEND_LOSS_TYPE:-WBCE}" --epoch_eval_train "${REPBLEND_EPOCH_EVAL_TRAIN:-100}")
      ;;
  esac
}

extract_repblend_checkpoint() {
  local log_path="$1"
  local iteration="${2:-3000}"
  python - "${log_path}" "${REPBLEND_ROOT}" "${iteration}" <<'PY'
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

run_repblend_budget() {
  local budget="$1"
  local budget_tag
  budget_tag="$(format_budget_tag "${budget}")"
  repblend_params_for_budget "${budget}" "${DATASET}"

  local run_name="repblend_${DATASET}_${budget_tag}_${RUN_TIMESTAMP}"
  local distill_log="${LOG_DIR}/repblend_${budget_tag}_distill.log"
  local distill_measure="${MEASURE_DIR}/repblend_${budget_tag}_distill.json"
  local checkpoint_path=""

  if [[ "${REPBLEND_FORCE_REDISTILL}" == "1" ]]; then
    stage_log "Measure RepBlend distillation: ${budget_tag}"
    measure_command "repblend_${budget_tag}_distill" "${distill_measure}" "${distill_log}" "${REPBLEND_ROOT}" \
      env WANDB_MODE=disabled python distill_repblend.py \
        --dataset "${DATASET}" \
        --buffer_path "${REPBLEND_BUFFER_ROOT}/${DATASET}/${SOURCE_MODEL_TAG}/InfoNCE" \
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
  else
    stage_log "RepBlend redistill disabled; expecting checkpoint discovery from previous logs for ${budget_tag}"
  fi

  checkpoint_path="$(extract_repblend_checkpoint "${distill_log}" "${REPBLEND_ITERATION:-3000}" || true)"
  append_manifest method "repblend" dataset "${DATASET}" budget_type "size" budget_value "${budget}" budget_tag "${budget_tag}" \
    eval_backbone "" stage "distill_selection" gpu_count "${GPU_COUNT}" checkpoint_path "${checkpoint_path}" \
    log_path "${distill_log}" measurement_path "${distill_measure}" skipped "0"

  if [[ -z "${checkpoint_path}" || ! -f "${checkpoint_path}" ]]; then
    stage_log "Missing RepBlend checkpoint for ${budget_tag}; skip cross-arch evaluation"
    return 0
  fi

  for eval_backbone in ${EVAL_BACKBONES}; do
    local eval_log="${LOG_DIR}/repblend_${budget_tag}_${eval_backbone}_evaluate.log"
    local eval_measure="${MEASURE_DIR}/repblend_${budget_tag}_${eval_backbone}_evaluate.json"
    stage_log "Measure RepBlend train/eval: ${budget_tag} eval_backbone=${eval_backbone}"
    measure_command "repblend_${budget_tag}_${eval_backbone}_evaluate" "${eval_measure}" "${eval_log}" "${PROJECT_ROOT}" \
      env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${PROJECT_ROOT}/evaluate_only.py" \
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
    append_manifest method "repblend" dataset "${DATASET}" budget_type "size" budget_value "${budget}" budget_tag "${budget_tag}" \
      eval_backbone "${eval_backbone}" stage "training_eval" gpu_count "${GPU_COUNT}" checkpoint_path "${checkpoint_path}" \
      evaluate_log "${eval_log}" log_path "${eval_log}" measurement_path "${eval_measure}" skipped "0"
  done
}

stage_log "Architecture-bias + energy experiment start"
stage_log "  methods=${METHODS}"
stage_log "  dataset=${DATASET} source=${SOURCE_MODEL_TAG} eval_backbones=${EVAL_BACKBONES}"
stage_log "  budgets=${BUDGETS[*]} ratios=${RATIOS[*]:-<none>}"
stage_log "  energy: Zeus preferred=${ENERGY_PREFER_ZEUS}, GPU sampler interval=${ENERGY_GPU_SAMPLER_INTERVAL}s, CPU=Intel RAPL"

if method_enabled ours; then
  run_real_subset_method "ours" "${OURS_SELECTION_ROOT}" "${OURS_SELECTION_TAG}" "${OURS_TRAIN_ROOT}" "ours_crossarch_energy"
fi

if method_enabled random; then
  run_real_subset_method "random" "${RANDOM_SELECTION_ROOT}" "${RANDOM_SELECTION_TAG}" "${RANDOM_TRAIN_ROOT}" "random_crossarch_energy"
fi

if method_enabled repblend; then
  if [[ ! -d "${REPBLEND_ROOT}" ]]; then
    stage_log "RepBlend root not found: ${REPBLEND_ROOT}"
  else
    for budget in "${BUDGETS[@]}"; do
      run_repblend_budget "${budget}"
    done
  fi
fi

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${MANIFEST_PATH}" \
  --output_dir "${REPORT_DIR}"

stage_log "Architecture-bias + energy experiment done"
stage_log "  detail=${REPORT_DIR}/supplemental_detail.csv"
stage_log "  architecture_bias=${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${REPORT_DIR}/energy_efficiency.csv"
stage_log "  measurements=${MEASURE_DIR}"
stage_log "  logs=${LOG_DIR}"
