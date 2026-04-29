#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Supplement only the downstream cross-architecture retrieval evaluation for
# Ours/Random. This script intentionally never re-runs subset sampling.
DATASET="${CROSSARCH_SUPP_DATASET:-flickr}"
SOURCE_BACKBONE="${CROSSARCH_SUPP_SOURCE_BACKBONE:-nfnet}"
TEXT_ENCODER="${CROSSARCH_SUPP_TEXT_ENCODER:-bert}"
EVAL_BACKBONES="${CROSSARCH_SUPP_EVAL_BACKBONES:-resnet50 vit_b16}"
BUDGETS_STR="${CROSSARCH_SUPP_BUDGETS:-100 200 500}"
RATIOS_STR="${CROSSARCH_SUPP_RATIOS:-0.01 0.02 0.05}"
SEEDS_STR="${CROSSARCH_SUPP_SEEDS:-0}"

read -r -a BUDGETS <<< "${BUDGETS_STR}"
read -r -a RATIOS <<< "${RATIOS_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"

SOURCE_MODEL_TAG="$(sanitize_component "${SOURCE_BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
IMAGE_ROOT="$(get_image_root "${DATASET}")"

OURS_SELECTION_ROOT="${CROSSARCH_SUPP_OURS_SELECTION_ROOT:-artifacts/subset_selection_dense_sift_bovw}"
RANDOM_SELECTION_ROOT="${CROSSARCH_SUPP_RANDOM_SELECTION_ROOT:-artifacts/subset_selection_random_baseline}"
OURS_TRAIN_ROOT="${CROSSARCH_SUPP_OURS_TRAIN_ROOT:-artifacts/supplemental_arch_energy/subset_train_ours_crossarch}"
RANDOM_TRAIN_ROOT="${CROSSARCH_SUPP_RANDOM_TRAIN_ROOT:-artifacts/supplemental_arch_energy/subset_train_random_crossarch}"

OURS_SELECTION_METHOD_TAG="${CROSSARCH_SUPP_OURS_SELECTION_METHOD_TAG:-proxy_opt_lsrc}"
RANDOM_SELECTION_METHOD_TAG="${CROSSARCH_SUPP_RANDOM_SELECTION_METHOD_TAG:-random}"
OURS_SUBSET_TAG="${CROSSARCH_SUPP_OURS_SUBSET_TAG:-ours_crossarch}"
RANDOM_SUBSET_TAG="${CROSSARCH_SUPP_RANDOM_SUBSET_TAG:-random_crossarch}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_ROOT="${CROSSARCH_SUPP_OUTPUT_ROOT:-artifacts/supplemental_arch_energy}"
REPORT_DIR="${CROSSARCH_SUPP_REPORT_DIR:-${OUTPUT_ROOT}/reports/${DATASET}_ours_random_resnet_vit_${RUN_TIMESTAMP}}"
LOG_DIR="${CROSSARCH_SUPP_LOG_DIR:-${OUTPUT_ROOT}/logs/${DATASET}_ours_random_resnet_vit_${RUN_TIMESTAMP}}"
MANIFEST_PATH="${REPORT_DIR}/manifest.jsonl"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
: > "${MANIFEST_PATH}"

GPU_COUNT="${CROSSARCH_SUPP_GPU_COUNT:-$(python - <<PY
devices = "${CUDA_VISIBLE_DEVICES:-}".strip()
print(max(len([x for x in devices.split(",") if x.strip()]), 1))
PY
)}"

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
row = dict(zip(keys, values))
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

run_train_eval_if_needed() {
  local method="$1"
  local selection_root="$2"
  local selection_method_tag="$3"
  local train_root="$4"
  local subset_tag="$5"
  local budget_type="$6"
  local budget_value="$7"
  local budget_tag="$8"
  local eval_backbone="$9"
  local seed="${10}"

  local selected_indices_path="${selection_root}/${DATASET}/train/${SOURCE_MODEL_TAG}/${budget_tag}/${selection_method_tag}/seed_${seed}/selected_indices.json"
  local eval_model_tag
  eval_model_tag="$(sanitize_component "${eval_backbone}")_$(sanitize_component "${TEXT_ENCODER}")"
  local metrics_path="${train_root}/${DATASET}/${eval_model_tag}/${budget_tag}/${subset_tag}/seed_${seed}/metrics.json"
  local log_path="${LOG_DIR}/${method}_${budget_tag}_${eval_backbone}_seed${seed}.log"

  if [[ ! -f "${selected_indices_path}" ]]; then
    stage_log "Missing selected indices, skip ${method}/${budget_tag}/seed_${seed}: ${selected_indices_path}"
    append_manifest \
      method "${method}" dataset "${DATASET}" budget_type "${budget_type}" budget_value "${budget_value}" budget_tag "${budget_tag}" \
      eval_backbone "${eval_backbone}" seed "${seed}" stage "missing_selected_indices" seconds "0" gpu_count "${GPU_COUNT}" \
      selected_indices_path "${selected_indices_path}" metrics_path "${metrics_path}" log_path "${log_path}" skipped "1"
    return 0
  fi

  if [[ -f "${metrics_path}" ]]; then
    stage_log "Skip train/eval: existing ${metrics_path}"
    append_manifest \
      method "${method}" dataset "${DATASET}" budget_type "${budget_type}" budget_value "${budget_value}" budget_tag "${budget_tag}" \
      eval_backbone "${eval_backbone}" seed "${seed}" stage "train_eval" seconds "0" gpu_count "${GPU_COUNT}" \
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
  if [[ "${ENABLE_IMAGE_ENCODER_DATA_PARALLEL}" == "1" ]]; then
    train_extra+=(--enable_image_encoder_data_parallel)
    if [[ -n "${IMAGE_ENCODER_DATA_PARALLEL_DEVICE_IDS}" ]]; then
      train_extra+=(--image_encoder_data_parallel_device_ids "${IMAGE_ENCODER_DATA_PARALLEL_DEVICE_IDS}")
    fi
  fi

  stage_log "Train/eval start: method=${method} budget=${budget_tag} eval_backbone=${eval_backbone} seed=${seed}"
  local start end elapsed
  start="$(date +%s)"
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
    "${train_extra[@]}" \
    > "${log_path}" 2>&1
  end="$(date +%s)"
  elapsed="$((end - start))"
  stage_log "Train/eval done: method=${method} budget=${budget_tag} eval_backbone=${eval_backbone} seed=${seed}"

  append_manifest \
    method "${method}" dataset "${DATASET}" budget_type "${budget_type}" budget_value "${budget_value}" budget_tag "${budget_tag}" \
    eval_backbone "${eval_backbone}" seed "${seed}" stage "train_eval" seconds "${elapsed}" gpu_count "${GPU_COUNT}" \
    selected_indices_path "${selected_indices_path}" metrics_path "${metrics_path}" log_path "${log_path}" skipped "0"
}

run_budget_group() {
  local method="$1"
  local selection_root="$2"
  local selection_method_tag="$3"
  local train_root="$4"
  local subset_tag="$5"

  for budget in "${BUDGETS[@]}"; do
    local budget_tag
    budget_tag="$(format_budget_tag "${budget}")"
    for eval_backbone in ${EVAL_BACKBONES}; do
      for seed in "${SEEDS[@]}"; do
        run_train_eval_if_needed "${method}" "${selection_root}" "${selection_method_tag}" "${train_root}" "${subset_tag}" \
          "size" "${budget}" "${budget_tag}" "${eval_backbone}" "${seed}"
      done
    done
  done

  for ratio in "${RATIOS[@]}"; do
    local ratio_tag
    ratio_tag="$(ratio_to_tag "${ratio}")"
    for eval_backbone in ${EVAL_BACKBONES}; do
      for seed in "${SEEDS[@]}"; do
        run_train_eval_if_needed "${method}" "${selection_root}" "${selection_method_tag}" "${train_root}" "${subset_tag}" \
          "ratio" "${ratio}" "${ratio_tag}" "${eval_backbone}" "${seed}"
      done
    done
  done
}

cd "${PROJECT_ROOT}"

stage_log "Cross-architecture retrieval supplement start"
stage_log "  dataset=${DATASET}"
stage_log "  source selected-index tag=${SOURCE_MODEL_TAG}"
stage_log "  eval backbones=${EVAL_BACKBONES}"
stage_log "  budgets=${BUDGETS[*]} ratios=${RATIOS[*]} seeds=${SEEDS[*]}"
stage_log "  will not run any sampling stage"

run_budget_group "ours" "${OURS_SELECTION_ROOT}" "${OURS_SELECTION_METHOD_TAG}" "${OURS_TRAIN_ROOT}" "${OURS_SUBSET_TAG}"
run_budget_group "random" "${RANDOM_SELECTION_ROOT}" "${RANDOM_SELECTION_METHOD_TAG}" "${RANDOM_TRAIN_ROOT}" "${RANDOM_SUBSET_TAG}"

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${MANIFEST_PATH}" \
  --output_dir "${REPORT_DIR}"

stage_log "Cross-architecture retrieval supplement done"
stage_log "  detail=${REPORT_DIR}/supplemental_detail.csv"
stage_log "  architecture_bias=${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${REPORT_DIR}/energy_efficiency.csv"
stage_log "  logs=${LOG_DIR}"
