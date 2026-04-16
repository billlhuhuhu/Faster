#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

# Only run the 4 newly added baselines.
export BASELINE_METHODS="${BASELINE_METHODS:-presel visa dataprophet dynamic_pruning}"

# Formal default protocol (can be overridden from env).
export ABS_BUDGETS="${ABS_BUDGETS:-100 200 500}"
export RATIOS="${RATIOS:-0.01 0.02 0.03}"
export BASELINE_SEEDS="${BASELINE_SEEDS:-0}"
export BASELINE_EPOCHS="${BASELINE_EPOCHS:-20}"
export BASELINE_DEVICE="${BASELINE_DEVICE:-cuda:0}"
export BASELINE_CONFIG="${BASELINE_CONFIG:-baselines/configs/main_aligned_flickr_nfnet_bert.yaml}"
export BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-artifacts/baselines}"

echo "[new4-formal] methods=${BASELINE_METHODS}"
echo "[new4-formal] abs_budgets=${ABS_BUDGETS}"
echo "[new4-formal] ratios=${RATIOS}"
echo "[new4-formal] seeds=${BASELINE_SEEDS}"
echo "[new4-formal] output_root=${BASELINE_OUTPUT_ROOT}"

# Reuse the resume-capable formal pipeline script.
bash baselines/scripts/run_formal_baselines_final_table.sh

echo ""
echo "[new4-formal] done."
echo "[new4-formal] final table: ${BASELINE_OUTPUT_ROOT}/final_results_table.csv"

