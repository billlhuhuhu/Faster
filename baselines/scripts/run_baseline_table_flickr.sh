#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-artifacts/baselines}"
BASELINE_CONFIG="${BASELINE_CONFIG:-baselines/configs/main_aligned_flickr_nfnet_bert.yaml}"
BASELINE_METHODS="${BASELINE_METHODS:-entropy el2n grand gradmatch glister ccs-rand ccs-herd ccs-kcenter ccs-forget dq dfool nms adap_sne presel visa dataprophet dynamic_pruning}"
BASELINE_BUDGETS="${BASELINE_BUDGETS:-100 200 500}"
BASELINE_SEEDS="${BASELINE_SEEDS:-0}"
BASELINE_DEVICE="${BASELINE_DEVICE:-cpu}"

echo "[baseline-benchmark] config=${BASELINE_CONFIG}"
echo "[baseline-benchmark] methods=${BASELINE_METHODS}"
echo "[baseline-benchmark] budgets=${BASELINE_BUDGETS}"
echo "[baseline-benchmark] seeds=${BASELINE_SEEDS}"
echo "[baseline-benchmark] output_root=${BASELINE_ROOT}"

python -m baselines.runners.benchmark_baselines \
  --config "${BASELINE_CONFIG}" \
  --methods ${BASELINE_METHODS} \
  --budgets ${BASELINE_BUDGETS} \
  --seeds ${BASELINE_SEEDS} \
  --device "${BASELINE_DEVICE}" \
  --output_root "${BASELINE_ROOT}"
