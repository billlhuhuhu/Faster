#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${SCALE_ABLATION_DATASET:-flickr}"
BACKBONE="${SCALE_ABLATION_BACKBONE:-nfnet}"
TEXT_ENCODER="${SCALE_ABLATION_TEXT_ENCODER:-bert}"
MODEL_TAG="$(sanitize_component "${BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"

BUDGETS="${SCALE_ABLATION_BUDGETS:-200}"
RATIOS="${SCALE_ABLATION_RATIOS:-}"
SEEDS="${SCALE_ABLATION_SEEDS:-0}"
RUN_SELECTION="${SCALE_ABLATION_RUN_SELECTION:-1}"
RUN_TRAIN="${SCALE_ABLATION_RUN_TRAIN:-1}"
RUN_GROUPS="${SCALE_ABLATION_RUN_GROUPS:-single_1 single_2 single_4 multi_1_2_4 multi_2_4_8 multi_4_8_16}"

OUTPUT_ROOT="${SCALE_ABLATION_OUTPUT_ROOT:-artifacts/wavelet_scale_ablation_dense_sift_bovw}"
FEATURE_CACHE_ROOT_SHARED="${SCALE_ABLATION_FEATURE_CACHE_ROOT:-artifacts/feature_cache_dense_sift_bovw}"
TOPOLOGY_ROOT_SHARED="${SCALE_ABLATION_TOPOLOGY_ROOT:-artifacts/topology_graph_dense_sift_bovw}"
CROSS_ROOT_BASE="${SCALE_ABLATION_CROSS_ROOT:-${OUTPUT_ROOT}/cross_modal_topology}"
SELECTION_ROOT_BASE="${SCALE_ABLATION_SELECTION_ROOT:-${OUTPUT_ROOT}/subset_selection}"
TRAIN_ROOT="${SCALE_ABLATION_TRAIN_ROOT:-${OUTPUT_ROOT}/subset_train}"
REPORT_BASE="${SCALE_ABLATION_REPORT_ROOT:-${OUTPUT_ROOT}/reports}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
REPORT_DIR="${REPORT_BASE}/scale_ablation_${DATASET}_${RUN_TIMESTAMP}"
mkdir -p "${REPORT_DIR}"

# label|scale_type|scales|variant
CONFIGS=(
  "single_1|single|1|${SCALE_ABLATION_SINGLE_1_VARIANT:-wavelet_scale_single_1_dense_sift_bovw}"
  "single_2|single|2|${SCALE_ABLATION_SINGLE_2_VARIANT:-wavelet_scale_single_2_dense_sift_bovw}"
  "single_4|single|4|${SCALE_ABLATION_SINGLE_4_VARIANT:-wavelet_scale_single_4_dense_sift_bovw}"
  "multi_1_2_4|multi|1,2,4|${SCALE_ABLATION_MULTI_1_2_4_VARIANT:-wavelet_scale_multi_1_2_4_dense_sift_bovw}"
  "multi_2_4_8|multi|2,4,8|${SCALE_ABLATION_MULTI_2_4_8_VARIANT:-wavelet_scale_multi_2_4_8_dense_sift_bovw}"
  "multi_4_8_16|multi|4,8,16|${SCALE_ABLATION_MULTI_4_8_16_VARIANT:-wavelet_scale_multi_4_8_16_dense_sift_bovw}"
)

contains_group() {
  local needle="$1"
  local item
  for item in ${RUN_GROUPS}; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

reuse_root_for_group() {
  local label="$1"
  local env_name
  env_name="SCALE_ABLATION_$(echo "${label}" | tr '[:lower:]' '[:upper:]')_REUSE_TRAIN_ROOT"
  if [[ -n "${!env_name:-}" ]]; then
    echo "${!env_name}"
  elif [[ "${label}" == single_* && -n "${SCALE_ABLATION_SINGLE_REUSE_TRAIN_ROOT:-}" ]]; then
    echo "${SCALE_ABLATION_SINGLE_REUSE_TRAIN_ROOT}"
  else
    echo "${TRAIN_ROOT}"
  fi
}

stage_log "Wavelet scale ablation start: dataset=${DATASET} model=${MODEL_TAG}"
stage_log "  budgets=${BUDGETS} ratios=${RATIOS} seeds=${SEEDS}"
stage_log "  output=${OUTPUT_ROOT}"
stage_log "  run_selection=${RUN_SELECTION} run_train=${RUN_TRAIN}"
stage_log "  run_groups=${RUN_GROUPS}"

CONFIG_ARGS=()
for cfg in "${CONFIGS[@]}"; do
  IFS="|" read -r label scale_type scales variant <<< "${cfg}"
  scale_path="$(echo "${scales}" | tr ',' '_')"
  cfg_train_root="$(reuse_root_for_group "${label}")"
  CONFIG_ARGS+=("${label}|${scale_type}|${scales}|${variant}|${cfg_train_root}")

  if ! contains_group "${label}"; then
    stage_log "Skip scale group by config: ${label} scales=${scales}; aggregate from ${cfg_train_root}"
    continue
  fi

  stage_log "Scale group start: ${label} scales=${scales} variant=${variant}"

  # Keep feature/topology shared, but isolate cross-modal and selection outputs
  # because both stages are scale-dependent while their internal path tags do
  # not encode scale values.
  WAVELET_MAIN_BOVW_DATASET="${DATASET}" \
  WAVELET_MAIN_BOVW_BACKBONE="${BACKBONE}" \
  WAVELET_MAIN_BOVW_TEXT_ENCODER="${TEXT_ENCODER}" \
  WAVELET_MAIN_BOVW_VARIANT="${variant}" \
  WAVELET_MAIN_BOVW_BUDGETS="${BUDGETS}" \
  WAVELET_MAIN_BOVW_RATIOS="${RATIOS}" \
  WAVELET_MAIN_BOVW_FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT_SHARED}" \
  WAVELET_MAIN_BOVW_TOPOLOGY_ROOT="${TOPOLOGY_ROOT_SHARED}" \
  WAVELET_MAIN_BOVW_CROSS_OUTPUT_ROOT="${CROSS_ROOT_BASE}/${label}_${scale_path}" \
  WAVELET_MAIN_BOVW_SELECTION_OUTPUT_ROOT="${SELECTION_ROOT_BASE}/${label}_${scale_path}" \
  WAVELET_MAIN_BOVW_TRAIN_OUTPUT_ROOT="${TRAIN_ROOT}" \
  WAVELET_MAIN_BOVW_REPORT_NAME="scale_ablation_${label}" \
  WAVELET_MAIN_LATEST_SEEDS="${SEEDS}" \
  WAVELET_MAIN_LATEST_RUN_SELECTION="${RUN_SELECTION}" \
  WAVELET_MAIN_LATEST_RUN_TRAIN="${RUN_TRAIN}" \
  WAVELET_MAIN_LATEST_WAVELET_SCALES="${scales}" \
  WAVELET_MAIN_LATEST_MAIN_SCALES="${scales}" \
  WAVELET_FUSION_SCALES="${scales}" \
  bash "${SCRIPT_DIR}/run_wavelet_main_dense_sift_bovw_combo.sh"

  CONFIG_ARGS+=("${cfg}")
  stage_log "Scale group done: ${label}"
done

stage_log "Aggregate scale ablation table start"
python "${PROJECT_ROOT}/tools/aggregate_wavelet_scale_ablation.py" \
  --subset_train_root "${TRAIN_ROOT}" \
  --output_dir "${REPORT_DIR}" \
  --dataset "${DATASET}" \
  --model_tag "${MODEL_TAG}" \
  --budgets ${BUDGETS} \
  --ratios ${RATIOS} \
  --seeds ${SEEDS} \
  --configs "${CONFIG_ARGS[@]}"

stage_log "Wavelet scale ablation done"
stage_log "  report_dir=${REPORT_DIR}"
stage_log "  summary_csv=${REPORT_DIR}/wavelet_scale_ablation_summary.csv"
stage_log "  summary_md=${REPORT_DIR}/wavelet_scale_ablation_summary.md"
