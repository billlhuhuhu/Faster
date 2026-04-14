#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-artifacts/baselines}"
BASELINE_OUT="${BASELINE_OUT:-artifacts/baselines}"
BASELINE_BUDGETS="${BASELINE_BUDGETS:-100 200 500}"

echo "[baseline-export] root=${BASELINE_ROOT}"
echo "[baseline-export] output_dir=${BASELINE_OUT}"
echo "[baseline-export] budgets=${BASELINE_BUDGETS}"

python -m baselines.runners.export_baseline_tables \
  --root "${BASELINE_ROOT}" \
  --output_dir "${BASELINE_OUT}" \
  --budgets ${BASELINE_BUDGETS}

