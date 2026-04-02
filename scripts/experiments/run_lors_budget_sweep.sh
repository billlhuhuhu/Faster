#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

LORS_SWEEP_DATASET="${LORS_SWEEP_DATASET:-flickr}"
LORS_SWEEP_BUDGETS="${LORS_SWEEP_BUDGETS:-100 200 500}"
LORS_SWEEP_DEVICE="${LORS_SWEEP_DEVICE:-0}"
LORS_SWEEP_BUFFER_ROOT="${LORS_SWEEP_BUFFER_ROOT:-buffers_formal}"
LORS_SWEEP_LOG_ROOT="${LORS_SWEEP_LOG_ROOT:-logged_files_formal}"
LORS_SWEEP_FORCE_REBUILD_BUFFER="${LORS_SWEEP_FORCE_REBUILD_BUFFER:-0}"

# Stable defaults for running LoRS on the current codebase without immediately OOM'ing.
# Override any of these from the shell if you want a heavier setting.
LORS_NUM_EXPERTS="${LORS_NUM_EXPERTS:-100}"
LORS_TRAIN_EPOCHS="${LORS_TRAIN_EPOCHS:-50}"
LORS_EVAL_FREQ="${LORS_EVAL_FREQ:-5}"
LORS_ITERATION="${LORS_ITERATION:-3000}"
LORS_EVAL_IT="${LORS_EVAL_IT:-50}"
LORS_NUM_EVAL="${LORS_NUM_EVAL:-1}"
LORS_EPOCH_EVAL_TRAIN="${LORS_EPOCH_EVAL_TRAIN:-20}"
LORS_MINI_BATCH_SIZE="${LORS_MINI_BATCH_SIZE:-16}"
LORS_SYN_STEPS="${LORS_SYN_STEPS:-5}"
LORS_BATCH_TRAIN="${LORS_BATCH_TRAIN:-32}"
LORS_BATCH_TEST="${LORS_BATCH_TEST:-64}"
LORS_EXPERT_EPOCHS="${LORS_EXPERT_EPOCHS:-2}"
LORS_MAX_START_EPOCH="${LORS_MAX_START_EPOCH:-3}"
LORS_DISABLED_WANDB="${LORS_DISABLED_WANDB:-True}"
LORS_NO_AUG="${LORS_NO_AUG:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LORS_FORCE_REDISTILL="${LORS_FORCE_REDISTILL:-0}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SWEEP_REPORT_DIR="${REPORT_ROOT}/lors_budget_sweep_${LORS_SWEEP_DATASET}_${RUN_TIMESTAMP}"
MANIFEST_PATH="${SWEEP_REPORT_DIR}/manifest.json"
CSV_PATH="${SWEEP_REPORT_DIR}/lors_budget_sweep.csv"
mkdir -p "${SWEEP_REPORT_DIR}"

stage_log "LoRS budget sweep start: dataset=${LORS_SWEEP_DATASET} budgets=${LORS_SWEEP_BUDGETS}"
stage_log "LoRS budget sweep roots: buffer=${LORS_SWEEP_BUFFER_ROOT} log=${LORS_SWEEP_LOG_ROOT} force_buffer=${LORS_SWEEP_FORCE_REBUILD_BUFFER}"

MANIFEST_TMP="${SWEEP_REPORT_DIR}/manifest.tmp.jsonl"
: > "${MANIFEST_TMP}"

FORCE_BUFFER_THIS_ROUND="${LORS_SWEEP_FORCE_REBUILD_BUFFER}"
for budget in ${LORS_SWEEP_BUDGETS}; do
  stage_log "LoRS sweep start: budget=${budget}"
  run_tag="budget_${budget}"
  run_name="lors_${LORS_SWEEP_DATASET}_${budget}_${RUN_TIMESTAMP}"

  before_logs="$(mktemp)"
  find "${EXPERIMENT_LOG_ROOT}" -maxdepth 1 -type d -name "lors_baseline_${LORS_SWEEP_DATASET}_*" | sort > "${before_logs}" 2>/dev/null || true

  env \
    CUDA_VISIBLE_DEVICES="${LORS_SWEEP_DEVICE}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    LORS_DATASET="${LORS_SWEEP_DATASET}" \
    LORS_BUFFER_ROOT="${LORS_SWEEP_BUFFER_ROOT}" \
    LORS_LOG_ROOT="${LORS_SWEEP_LOG_ROOT}" \
    LORS_FORCE_REBUILD_BUFFER="${FORCE_BUFFER_THIS_ROUND}" \
    LORS_FORCE_REDISTILL="${LORS_FORCE_REDISTILL}" \
    LORS_RUN_TAG="${run_tag}" \
    LORS_RUN_NAME="${run_name}" \
    LORS_NUM_QUERIES="${budget}" \
    LORS_NUM_EXPERTS="${LORS_NUM_EXPERTS}" \
    LORS_TRAIN_EPOCHS="${LORS_TRAIN_EPOCHS}" \
    LORS_EVAL_FREQ="${LORS_EVAL_FREQ}" \
    LORS_ITERATION="${LORS_ITERATION}" \
    LORS_EVAL_IT="${LORS_EVAL_IT}" \
    LORS_NUM_EVAL="${LORS_NUM_EVAL}" \
    LORS_EPOCH_EVAL_TRAIN="${LORS_EPOCH_EVAL_TRAIN}" \
    LORS_MINI_BATCH_SIZE="${LORS_MINI_BATCH_SIZE}" \
    LORS_SYN_STEPS="${LORS_SYN_STEPS}" \
    LORS_BATCH_TRAIN="${LORS_BATCH_TRAIN}" \
    LORS_BATCH_TEST="${LORS_BATCH_TEST}" \
    LORS_EXPERT_EPOCHS="${LORS_EXPERT_EPOCHS}" \
    LORS_MAX_START_EPOCH="${LORS_MAX_START_EPOCH}" \
    LORS_DISABLED_WANDB="${LORS_DISABLED_WANDB}" \
    LORS_NO_AUG="${LORS_NO_AUG}" \
    bash "${SCRIPT_DIR}/run_lors_baseline.sh"

  latest_log_dir="$(find "${EXPERIMENT_LOG_ROOT}" -maxdepth 1 -type d -name "lors_baseline_${LORS_SWEEP_DATASET}_${run_tag}_*" | sort | tail -n 1)"
  checkpoint_path="$(python - "${latest_log_dir}/distill.log" "${PROJECT_ROOT}" "${LORS_ITERATION}" "${LORS_SWEEP_LOG_ROOT}" "${LORS_SWEEP_DATASET}" "${run_name}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
project_root = Path(sys.argv[2])
iteration = sys.argv[3]
log_root = sys.argv[4]
dataset = sys.argv[5]
run_name = sys.argv[6]
if log_path.exists():
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Saving to (.+)", text)
    if matches:
        raw_dir = matches[-1].strip()
        save_dir = Path(raw_dir)
        if not save_dir.is_absolute():
            save_dir = project_root / save_dir
        candidate = save_dir / f"distilled_{iteration}.pt"
        if candidate.exists():
            print(candidate)
            raise SystemExit(0)
print(project_root / log_root / dataset / run_name / f"distilled_{iteration}.pt")
PY
)"
  evaluate_log_path="${latest_log_dir}/evaluate.log"

  python - "${MANIFEST_TMP}" "${budget}" "${run_name}" "${latest_log_dir}" "${checkpoint_path}" "${evaluate_log_path}" <<'PY'
import json
import sys
from pathlib import Path

manifest_tmp = Path(sys.argv[1])
item = {
    "budget_size": int(sys.argv[2]),
    "run_name": sys.argv[3],
    "run_log_dir": sys.argv[4],
    "checkpoint_path": sys.argv[5],
    "evaluate_log": sys.argv[6],
}
with manifest_tmp.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
PY

  rm -f "${before_logs}"
  FORCE_BUFFER_THIS_ROUND=0
  stage_log "LoRS sweep done: budget=${budget}"
done

python - "${MANIFEST_TMP}" "${MANIFEST_PATH}" "${LORS_SWEEP_DATASET}" <<'PY'
import json
import sys
from pathlib import Path

manifest_tmp = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
dataset = sys.argv[3]
runs = []
for line in manifest_tmp.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    runs.append(json.loads(line))
manifest = {"dataset": dataset, "runs": runs}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved manifest: {manifest_path}")
PY

python "${PROJECT_ROOT}/tools/aggregate_lors_budget_sweep.py" \
  --manifest "${MANIFEST_PATH}" \
  --output_csv "${CSV_PATH}"

stage_log "LoRS budget sweep completed: report=${CSV_PATH}"
