#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

LORS_RESNET_DATASET="${LORS_RESNET_DATASET:-flickr}"
LORS_RESNET_RATIO="${LORS_RESNET_RATIO:-0.03}"
LORS_RESNET_UPSTREAM_BACKBONE="${LORS_RESNET_UPSTREAM_BACKBONE:-resnet10}"
LORS_RESNET_TEXT_ENCODER="${LORS_RESNET_TEXT_ENCODER:-bert}"
LORS_RESNET_EVAL_BACKBONES="${LORS_RESNET_EVAL_BACKBONES:-nfnet resnet50 vit_b16 resnet10}"
LORS_RESNET_DEVICE="${LORS_RESNET_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
LORS_RESNET_BUFFER_ROOT="${LORS_RESNET_BUFFER_ROOT:-buffers_arch3}"
LORS_RESNET_LOG_ROOT="${LORS_RESNET_LOG_ROOT:-logged_files_arch3}"
LORS_RESNET_OUTPUT_ROOT="${LORS_RESNET_OUTPUT_ROOT:-artifacts/arch_bias_energy_3pct/lors_resnet10_rerun}"
LORS_RESNET_RUN_TAG="${LORS_RESNET_RUN_TAG:-}"
LORS_RESNET_METHOD_NAME="${LORS_RESNET_METHOD_NAME:-lors}"
LORS_RESNET_FORCE_REDISTILL="${LORS_RESNET_FORCE_REDISTILL:-1}"
LORS_RESNET_GPU_COUNT="${LORS_RESNET_GPU_COUNT:-1}"

if [[ -z "${LORS_RESNET_RUN_TAG}" ]]; then
  LORS_RESNET_RUN_TAG="lors_resnet10_3pct_$(date '+%Y%m%d_%H%M%S')"
fi

IMAGE_ROOT="$(get_image_root "${LORS_RESNET_DATASET}")"
MODEL_TAG="$(sanitize_component "${LORS_RESNET_UPSTREAM_BACKBONE}")_$(sanitize_component "${LORS_RESNET_TEXT_ENCODER}")"
LOSS_TAG="${LORS_LOSS_TYPE:-InfoNCE}"
if [[ "${LORS_NO_AUG:-1}" == "1" ]]; then
  LOSS_TAG="${LOSS_TAG}_NoAug"
fi

BUFFER_LEAF_DIR="${LORS_RESNET_BUFFER_ROOT}/${LORS_RESNET_DATASET}/${MODEL_TAG}/${LOSS_TAG}"
RUN_ROOT="${LORS_RESNET_OUTPUT_ROOT}/${LORS_RESNET_RUN_TAG}"
LOG_DIR="${RUN_ROOT}/logs"
MEASURE_DIR="${RUN_ROOT}/measurements"
REPORT_DIR="${RUN_ROOT}/reports"
MANIFEST_JSON="${REPORT_DIR}/lors_resnet10_3pct_manifest.json"
MANIFEST_JSONL="${REPORT_DIR}/supplemental_manifest.jsonl"
RATIO_TAG="$(python - "${LORS_RESNET_RATIO}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
)"
mkdir -p "${LOG_DIR}" "${MEASURE_DIR}" "${REPORT_DIR}" "${LORS_RESNET_LOG_ROOT}"
: > "${MANIFEST_JSONL}"

compute_train_size() {
  python - "${LORS_RESNET_DATASET}" "${IMAGE_ROOT}" "${ANN_ROOT}" <<'PY'
import sys
from types import SimpleNamespace

from src.sklearn_compat import install_sklearn_metrics_stub_if_broken
install_sklearn_metrics_stub_if_broken()

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

write_manifest_row() {
  python - "$@" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
row = {
    "method": sys.argv[2],
    "dataset": sys.argv[3],
    "budget_tag": sys.argv[4],
    "budget_type": "ratio",
    "budget_value": sys.argv[5],
    "stage": sys.argv[6],
    "eval_backbone": sys.argv[7],
    "measurement_path": sys.argv[8],
    "evaluate_log": sys.argv[9],
    "source": sys.argv[10],
    "gpu_count": int(sys.argv[11]),
}
with manifest_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

if [[ ! -d "${BUFFER_LEAF_DIR}" ]]; then
  echo "LoRS replay buffer directory not found: ${BUFFER_LEAF_DIR}" >&2
  echo "Set LORS_RESNET_BUFFER_ROOT so that it contains ${LORS_RESNET_DATASET}/${MODEL_TAG}/${LOSS_TAG}" >&2
  exit 1
fi
if ! compgen -G "${BUFFER_LEAF_DIR}/img_replay_buffer_*.pt" > /dev/null || ! compgen -G "${BUFFER_LEAF_DIR}/txt_replay_buffer_*.pt" > /dev/null; then
  echo "Replay buffer files are incomplete under: ${BUFFER_LEAF_DIR}" >&2
  exit 1
fi

TRAIN_COUNT="$(compute_train_size)"
BUDGET_SIZE="$(ratio_to_count "${TRAIN_COUNT}" "${LORS_RESNET_RATIO}")"
DISTILL_RUN_NAME="${LORS_RESNET_RUN_TAG}_${RATIO_TAG}_${MODEL_TAG}"
CKPT_PATH="${LORS_RESNET_LOG_ROOT}/${LORS_RESNET_DATASET}/${DISTILL_RUN_NAME}/distilled_${LORS_ITERATION:-3000}.pt"
DISTILL_MEASURE="${MEASURE_DIR}/${RATIO_TAG}_${MODEL_TAG}_distill.json"
DISTILL_TEE_LOG="${LOG_DIR}/${RATIO_TAG}_${MODEL_TAG}_distill.log"

stage_log "LoRS resnet10 3% rerun start"
stage_log "  dataset=${LORS_RESNET_DATASET} train_count=${TRAIN_COUNT} ratio=${LORS_RESNET_RATIO} budget=${BUDGET_SIZE}"
stage_log "  upstream=${MODEL_TAG} buffer=${BUFFER_LEAF_DIR}"
stage_log "  eval_backbones=${LORS_RESNET_EVAL_BACKBONES}"
stage_log "  output=${RUN_ROOT}"
stage_log "  distill config: reuse buffer, force_redistill=${LORS_RESNET_FORCE_REDISTILL}, baseline defaults, not full dataset"

stage_log "Measure LoRS distill: ${RATIO_TAG} ${MODEL_TAG}"
python "${PROJECT_ROOT}/tools/measure_command_energy.py" \
  --label "lors_${RATIO_TAG}_${MODEL_TAG}_distill" \
  --output_json "${DISTILL_MEASURE}" \
  --working_dir "${PROJECT_ROOT}" \
  --tee_log "${DISTILL_TEE_LOG}" \
  ${ENERGY_PREFER_ZEUS:+--prefer_zeus} \
  -- \
  env \
    CUDA_VISIBLE_DEVICES="${LORS_RESNET_DEVICE}" \
    LORS_DATASET="${LORS_RESNET_DATASET}" \
    LORS_IMAGE_ENCODER="${LORS_RESNET_UPSTREAM_BACKBONE}" \
    LORS_TEXT_ENCODER="${LORS_RESNET_TEXT_ENCODER}" \
    LORS_BUFFER_ROOT="${LORS_RESNET_BUFFER_ROOT}" \
    LORS_LOG_ROOT="${LORS_RESNET_LOG_ROOT}" \
    LORS_FORCE_REBUILD_BUFFER="0" \
    LORS_FORCE_REDISTILL="${LORS_RESNET_FORCE_REDISTILL}" \
    LORS_RUN_EVALUATE="0" \
    LORS_RUN_TAG="${RATIO_TAG}_${MODEL_TAG}_rerun" \
    LORS_RUN_NAME="${DISTILL_RUN_NAME}" \
    LORS_NUM_QUERIES="${BUDGET_SIZE}" \
    LORS_NO_AUG="${LORS_NO_AUG:-1}" \
    LORS_MINI_BATCH_SIZE="${LORS_MINI_BATCH_SIZE:-100}" \
    LORS_ITERATION="${LORS_ITERATION:-3000}" \
    LORS_EVAL_IT="${LORS_EVAL_IT:-50}" \
    LORS_NUM_EVAL="${LORS_NUM_EVAL:-1}" \
    LORS_EPOCH_EVAL_TRAIN="${LORS_EPOCH_EVAL_TRAIN:-100}" \
    LORS_EXPERT_EPOCHS="${LORS_EXPERT_EPOCHS:-3}" \
    LORS_SYN_STEPS="${LORS_SYN_STEPS:-20}" \
    LORS_MAX_START_EPOCH="${LORS_MAX_START_EPOCH:-25}" \
    LORS_BATCH_TRAIN="${LORS_BATCH_TRAIN:-128}" \
    LORS_BATCH_TEST="${LORS_BATCH_TEST:-128}" \
    bash "${SCRIPT_DIR}/run_lors_baseline.sh"

if [[ ! -f "${CKPT_PATH}" ]]; then
  CKPT_PATH="$(find "${LORS_RESNET_LOG_ROOT}/${LORS_RESNET_DATASET}" -type f -name "distilled_${LORS_ITERATION:-3000}.pt" -path "*${DISTILL_RUN_NAME}*" | sort | tail -n 1)"
fi
if [[ -z "${CKPT_PATH}" || ! -f "${CKPT_PATH}" ]]; then
  echo "No distilled checkpoint found after distill. Expected: ${LORS_RESNET_LOG_ROOT}/${LORS_RESNET_DATASET}/${DISTILL_RUN_NAME}/distilled_${LORS_ITERATION:-3000}.pt" >&2
  exit 1
fi
stage_log "LoRS distill checkpoint: ${CKPT_PATH}"
write_manifest_row "${MANIFEST_JSONL}" "${LORS_RESNET_METHOD_NAME}" "${LORS_RESNET_DATASET}" "${RATIO_TAG}" "${LORS_RESNET_RATIO}" "distill_selection" "" "${DISTILL_MEASURE}" "" "${DISTILL_TEE_LOG}" "${LORS_RESNET_GPU_COUNT}"

python - "${MANIFEST_JSON}" "${LORS_RESNET_DATASET}" "${LORS_RESNET_RATIO}" "${BUDGET_SIZE}" "${MODEL_TAG}" "${CKPT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "dataset": sys.argv[2],
    "ratio": float(sys.argv[3]),
    "budget_size": int(sys.argv[4]),
    "distill_backbone": sys.argv[5].split("_")[0],
    "checkpoint_path": sys.argv[6],
    "runs": [],
}
Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

for eval_backbone in ${LORS_RESNET_EVAL_BACKBONES}; do
  EVAL_LOG="${LOG_DIR}/${RATIO_TAG}_${eval_backbone}_evaluate.log"
  EVAL_MEASURE="${MEASURE_DIR}/${RATIO_TAG}_${eval_backbone}_evaluate.json"
  eval_extra_args=()
  if [[ "${LORS_NO_AUG:-1}" == "1" ]]; then
    eval_extra_args+=(--no_aug)
  fi
  if [[ "${eval_backbone}" == "vit_b16" && "${LORS_VIT_USE_LOW_LR_FINETUNE:-1}" == "1" ]]; then
    eval_extra_args+=(
      --image_trainable true
      --text_trainable false
      --lr_teacher_img "${LORS_VIT_LOWLR_IMG:-0.001}"
      --lr_teacher_txt "${LORS_VIT_LOWLR_TXT:-0.05}"
      --epoch_eval_train "${LORS_VIT_EPOCH_EVAL_TRAIN:-300}"
      --batch_train "${LORS_VIT_BATCH_TRAIN:-32}"
      --batch_size_train "${LORS_VIT_BATCH_TRAIN:-32}"
      --batch_size_test "${LORS_VIT_BATCH_TEST:-64}"
    )
  else
    eval_extra_args+=(
      --epoch_eval_train "${LORS_EPOCH_EVAL_TRAIN:-100}"
      --batch_train "${LORS_BATCH_TRAIN:-128}"
      --batch_size_train "${LORS_BATCH_TRAIN:-128}"
      --batch_size_test "${LORS_BATCH_TEST:-128}"
    )
  fi

  stage_log "Measure LoRS evaluate: ${RATIO_TAG} eval_backbone=${eval_backbone}"
  python "${PROJECT_ROOT}/tools/measure_command_energy.py" \
    --label "lors_${RATIO_TAG}_${eval_backbone}_evaluate" \
    --output_json "${EVAL_MEASURE}" \
    --working_dir "${PROJECT_ROOT}" \
    --tee_log "${EVAL_LOG}" \
    ${ENERGY_PREFER_ZEUS:+--prefer_zeus} \
    -- \
    env CUDA_VISIBLE_DEVICES="${LORS_RESNET_DEVICE}" \
    python "${PROJECT_ROOT}/evaluate_only.py" \
      --dataset "${LORS_RESNET_DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --model_checkpoint_root "${LORS_MODEL_CHECKPOINT_ROOT:-${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}}" \
      --image_encoder "${eval_backbone}" \
      --text_encoder "${LORS_RESNET_TEXT_ENCODER}" \
      --loss_type "${LORS_LOSS_TYPE:-InfoNCE}" \
      --ckpt_path "${CKPT_PATH}" \
      --num_eval "${LORS_NUM_EVAL:-1}" \
      --disabled_wandb "${LORS_DISABLED_WANDB:-True}" \
      "${eval_extra_args[@]}"

  python - "${MANIFEST_JSON}" "${LORS_RESNET_RATIO}" "${BUDGET_SIZE}" "${LORS_RESNET_UPSTREAM_BACKBONE}" "${eval_backbone}" "${DISTILL_RUN_NAME}" "${CKPT_PATH}" "${EVAL_LOG}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload.setdefault("runs", []).append({
    "ratio": float(sys.argv[2]),
    "budget_size": int(sys.argv[3]),
    "distill_backbone": sys.argv[4],
    "eval_backbone": sys.argv[5],
    "run_name": sys.argv[6],
    "checkpoint_path": sys.argv[7],
    "evaluate_log": sys.argv[8],
})
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  write_manifest_row "${MANIFEST_JSONL}" "${LORS_RESNET_METHOD_NAME}" "${LORS_RESNET_DATASET}" "${RATIO_TAG}" "${LORS_RESNET_RATIO}" "training_eval" "${eval_backbone}" "${EVAL_MEASURE}" "${EVAL_LOG}" "${EVAL_LOG}" "${LORS_RESNET_GPU_COUNT}"
done

python "${PROJECT_ROOT}/tools/aggregate_lors_ratio_crossarch.py" \
  --manifest "${MANIFEST_JSON}" \
  --output_csv "${REPORT_DIR}/lors_resnet10_3pct_crossarch.csv"

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${MANIFEST_JSONL}" \
  --output_dir "${REPORT_DIR}"

stage_log "LoRS resnet10 3% rerun completed"
stage_log "  cross-arch table: ${REPORT_DIR}/lors_resnet10_3pct_crossarch.csv"
stage_log "  detail table: ${REPORT_DIR}/supplemental_detail.csv"
stage_log "  arch table: ${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy table: ${REPORT_DIR}/energy_efficiency.csv"
