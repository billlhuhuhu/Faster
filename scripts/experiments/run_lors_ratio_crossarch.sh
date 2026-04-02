#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

LORS_RATIO_DATASET="${LORS_RATIO_DATASET:-flickr}"
LORS_RATIO_VALUES="${LORS_RATIO_VALUES:-0.01 0.02 0.05}"
LORS_RATIO_DEVICE="${LORS_RATIO_DEVICE:-0}"
LORS_RATIO_BUFFER_ROOT="${LORS_RATIO_BUFFER_ROOT:-buffers_formal_v2}"
LORS_RATIO_LOG_ROOT="${LORS_RATIO_LOG_ROOT:-logged_files_formal_v2}"
LORS_RATIO_DISTILL_BACKBONE="${LORS_RATIO_DISTILL_BACKBONE:-nfnet}"
LORS_RATIO_TEXT_ENCODER="${LORS_RATIO_TEXT_ENCODER:-bert}"
LORS_RATIO_EVAL_BACKBONES="${LORS_RATIO_EVAL_BACKBONES:-nfnet resnet50 vit_b16}"
LORS_RATIO_FORCE_REDISTILL="${LORS_RATIO_FORCE_REDISTILL:-0}"

IMAGE_ROOT="$(get_image_root "${LORS_RATIO_DATASET}")"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/lors_ratio_crossarch_${LORS_RATIO_DATASET}_${RUN_TIMESTAMP}"
REPORT_DIR="${REPORT_ROOT}/lors_ratio_crossarch_${LORS_RATIO_DATASET}_${RUN_TIMESTAMP}"
MANIFEST_PATH="${REPORT_DIR}/manifest.json"
CSV_PATH="${REPORT_DIR}/lors_ratio_crossarch.csv"
MANIFEST_TMP="${REPORT_DIR}/manifest.tmp.jsonl"
mkdir -p "${RUN_LOG_DIR}" "${REPORT_DIR}"
: > "${MANIFEST_TMP}"

compute_train_size() {
  python - "${LORS_RATIO_DATASET}" "${IMAGE_ROOT}" "${ANN_ROOT}" <<'PY'
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
count = max(1, int(round(total * ratio)))
print(count)
PY
}

ratio_to_tag() {
  local ratio="$1"
  python - "${ratio}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
}

extract_checkpoint_from_distill_log() {
  local distill_log="$1"
  local iteration="$2"
  python - "${distill_log}" "${PROJECT_ROOT}" "${iteration}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
project_root = Path(sys.argv[2])
iteration = sys.argv[3]
if not log_path.exists():
    raise SystemExit(1)
text = log_path.read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"Saving to (.+)", text)
if not matches:
    raise SystemExit(1)
save_dir = Path(matches[-1].strip())
if not save_dir.is_absolute():
    save_dir = project_root / save_dir
print(save_dir / f"distilled_{iteration}.pt")
PY
}

TRAIN_COUNT="$(compute_train_size)"
stage_log "LoRS ratio cross-arch start: dataset=${LORS_RATIO_DATASET} train_count=${TRAIN_COUNT} ratios=${LORS_RATIO_VALUES}"
stage_log "LoRS ratio cross-arch roots: buffer=${LORS_RATIO_BUFFER_ROOT} log=${LORS_RATIO_LOG_ROOT}"

for ratio in ${LORS_RATIO_VALUES}; do
  ratio_tag="$(ratio_to_tag "${ratio}")"
  budget_size="$(ratio_to_count "${TRAIN_COUNT}" "${ratio}")"
  run_name="lors_${LORS_RATIO_DATASET}_${ratio_tag}_${RUN_TIMESTAMP}"

  stage_log "LoRS ratio distill start: ratio=${ratio} budget=${budget_size}"
  env \
    CUDA_VISIBLE_DEVICES="${LORS_RATIO_DEVICE}" \
    LORS_DATASET="${LORS_RATIO_DATASET}" \
    LORS_IMAGE_ENCODER="${LORS_RATIO_DISTILL_BACKBONE}" \
    LORS_TEXT_ENCODER="${LORS_RATIO_TEXT_ENCODER}" \
    LORS_BUFFER_ROOT="${LORS_RATIO_BUFFER_ROOT}" \
    LORS_LOG_ROOT="${LORS_RATIO_LOG_ROOT}" \
    LORS_FORCE_REBUILD_BUFFER="0" \
    LORS_FORCE_REDISTILL="${LORS_RATIO_FORCE_REDISTILL}" \
    LORS_RUN_TAG="${ratio_tag}" \
    LORS_RUN_NAME="${run_name}" \
    LORS_NUM_QUERIES="${budget_size}" \
    bash "${SCRIPT_DIR}/run_lors_baseline.sh"

  baseline_log_dir="$(find "${EXPERIMENT_LOG_ROOT}" -maxdepth 1 -type d -name "lors_baseline_${LORS_RATIO_DATASET}_${ratio_tag}_*" | sort | tail -n 1)"
  distill_log_path="${baseline_log_dir}/distill.log"
  checkpoint_path="$(extract_checkpoint_from_distill_log "${distill_log_path}" "${LORS_ITERATION:-3000}")"

  eval_backbone=""
  for eval_backbone in ${LORS_RATIO_EVAL_BACKBONES}; do
    eval_log_path="${RUN_LOG_DIR}/${ratio_tag}_${eval_backbone}.log"
    if [[ -f "${eval_log_path}" ]]; then
      stage_log "Skip evaluate: ratio=${ratio} backbone=${eval_backbone}"
    else
      stage_log "LoRS evaluate start: ratio=${ratio} budget=${budget_size} backbone=${eval_backbone}"
      eval_extra_args=()
      if [[ "${LORS_NO_AUG:-1}" == "1" ]]; then
        eval_extra_args+=(--no_aug)
      fi

      env CUDA_VISIBLE_DEVICES="${LORS_RATIO_DEVICE}" \
      python "${PROJECT_ROOT}/evaluate_only.py" \
        --dataset "${LORS_RATIO_DATASET}" \
        --image_root "${IMAGE_ROOT}" \
        --ann_root "${ANN_ROOT}" \
        --model_checkpoint_root "${LORS_MODEL_CHECKPOINT_ROOT:-${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}}" \
        --image_encoder "${eval_backbone}" \
        --text_encoder "${LORS_RATIO_TEXT_ENCODER}" \
        --loss_type "${LORS_LOSS_TYPE:-InfoNCE}" \
        --ckpt_path "${checkpoint_path}" \
        --num_eval "${LORS_NUM_EVAL:-1}" \
        --batch_train "${LORS_BATCH_TRAIN:-128}" \
        --batch_size_train "${LORS_BATCH_TRAIN:-128}" \
        --batch_size_test "${LORS_BATCH_TEST:-128}" \
        --disabled_wandb "${LORS_DISABLED_WANDB:-True}" \
        "${eval_extra_args[@]}" \
        > "${eval_log_path}" 2>&1
    fi

    python - "${MANIFEST_TMP}" "${ratio}" "${budget_size}" "${LORS_RATIO_DISTILL_BACKBONE}" "${eval_backbone}" "${run_name}" "${baseline_log_dir}" "${checkpoint_path}" "${eval_log_path}" <<'PY'
import json
import sys
from pathlib import Path

manifest_tmp = Path(sys.argv[1])
item = {
    "ratio": float(sys.argv[2]),
    "budget_size": int(sys.argv[3]),
    "distill_backbone": sys.argv[4],
    "eval_backbone": sys.argv[5],
    "run_name": sys.argv[6],
    "baseline_log_dir": sys.argv[7],
    "checkpoint_path": sys.argv[8],
    "evaluate_log": sys.argv[9],
}
with manifest_tmp.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
PY
  done
done

python - "${MANIFEST_TMP}" "${MANIFEST_PATH}" "${LORS_RATIO_DATASET}" <<'PY'
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

python "${PROJECT_ROOT}/tools/aggregate_lors_ratio_crossarch.py" \
  --manifest "${MANIFEST_PATH}" \
  --output_csv "${CSV_PATH}"

stage_log "LoRS ratio cross-arch completed: report=${CSV_PATH}"
