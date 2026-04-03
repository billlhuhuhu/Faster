#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${B1_T_SWEEP_DATASET:-flickr}"
BACKBONE="${B1_T_SWEEP_BACKBONE:-nfnet}"
TEXT_ENCODER="${B1_T_SWEEP_TEXT_ENCODER:-bert}"
SEEDS_STR="${B1_T_SWEEP_SEEDS:-0}"
read -r -a SEEDS <<< "${SEEDS_STR}"
TIMES_STR="${B1_T_SWEEP_TIMES:-1 2 3 4 5}"
read -r -a TIMES <<< "${TIMES_STR}"
FIXED_BUDGET="${B1_T_SWEEP_BUDGET:-500}"

TRAIN_OUTPUT_ROOT_BASE="${B1_T_SWEEP_TRAIN_ROOT:-artifacts/subset_train_b1_diffusion_time}"
CROSS_OUTPUT_ROOT_BASE="${B1_T_SWEEP_CROSS_ROOT:-artifacts/cross_modal_topology_b1_diffusion_time}"
SELECTION_OUTPUT_ROOT_BASE="${B1_T_SWEEP_SELECTION_ROOT:-artifacts/subset_selection_b1_diffusion_time}"
REPORT_NAME="${B1_T_SWEEP_REPORT_NAME:-b1_diffusion_time_sweep}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/${REPORT_NAME}_${DATASET}_${RUN_TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

METHODS=()

stage_log "B1 diffusion-time sweep start: dataset=${DATASET} budget=${FIXED_BUDGET} times=${TIMES[*]} seeds=${SEEDS[*]}"

for diffusion_time in "${TIMES[@]}"; do
  variant="b1_t${diffusion_time}"
  METHODS+=("${variant}")

  stage_log "B1 diffusion-time run start: t=${diffusion_time} variant=${variant}"
  B1_DATASET="${DATASET}" \
  B1_BACKBONE="${BACKBONE}" \
  B1_TEXT_ENCODER="${TEXT_ENCODER}" \
  B1_SEEDS="${SEEDS_STR}" \
  B1_BUDGETS="${FIXED_BUDGET}" \
  B1_VARIANT="${variant}" \
  B1_DIFFUSION_TIME="${diffusion_time}" \
  B1_CROSS_OUTPUT_ROOT="${CROSS_OUTPUT_ROOT_BASE}/t${diffusion_time}" \
  B1_SELECTION_OUTPUT_ROOT="${SELECTION_OUTPUT_ROOT_BASE}/t${diffusion_time}" \
  B1_TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT_BASE}" \
  B1_REPORT_NAME="${variant}_abs" \
  bash "${SCRIPT_DIR}/run_b1_abs.sh" \
    > "${RUN_LOG_DIR}/${variant}.log" 2>&1
  stage_log "B1 diffusion-time run done: t=${diffusion_time} variant=${variant}"
done

python "${PROJECT_ROOT}/tools/aggregate_main_table_metrics.py" \
  --subset_train_root "${TRAIN_OUTPUT_ROOT_BASE}" \
  --output_root "${REPORT_ROOT}" \
  --report_name "${REPORT_NAME}" \
  --datasets "${DATASET}" \
  --backbone "${BACKBONE}" \
  --methods "${METHODS[@]}" \
  --budget_sizes "${FIXED_BUDGET}" \
  --seeds "${SEEDS[@]}"

stage_log "B1 diffusion-time sweep completed. Logs saved to ${RUN_LOG_DIR}"
