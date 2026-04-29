#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

LORS3_DATASET="${LORS3_DATASET:-flickr}"
LORS3_RATIO="${LORS3_RATIO:-0.03}"
LORS3_UPSTREAM_BACKBONE="${LORS3_UPSTREAM_BACKBONE:-resnet10}"
LORS3_TEXT_ENCODER="${LORS3_TEXT_ENCODER:-bert}"
LORS3_EVAL_BACKBONES="${LORS3_EVAL_BACKBONES:-nfnet resnet50 vit_b16}"
LORS3_DEVICE="${LORS3_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
LORS3_OUTPUT_ROOT="${LORS3_OUTPUT_ROOT:-artifacts/arch_bias_energy_3pct/lors}"
LORS3_MEASURED_MAX_FILES="${LORS3_MEASURED_MAX_FILES:-1}"
LORS3_MEASURED_MAX_EXPERTS="${LORS3_MEASURED_MAX_EXPERTS:-5}"
LORS3_FULL_MAX_FILES="${LORS3_FULL_MAX_FILES:-1}"
LORS3_FULL_MAX_EXPERTS="${LORS3_FULL_MAX_EXPERTS:-100}"
LORS_EPOCH_EVAL_TRAIN="${LORS_EPOCH_EVAL_TRAIN:-200}"
LORS3_SCALE_FACTOR="${LORS3_SCALE_FACTOR:-$(python - <<PY
measured = max(float("${LORS3_MEASURED_MAX_FILES}") * float("${LORS3_MEASURED_MAX_EXPERTS}"), 1.0)
full = max(float("${LORS3_FULL_MAX_FILES}") * float("${LORS3_FULL_MAX_EXPERTS}"), measured)
print(full / measured)
PY
)}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_TAG="${LORS3_RUN_TAG:-lors_3pct_${RUN_TIMESTAMP}}"
REPORT_DIR="${LORS3_REPORT_DIR:-${LORS3_OUTPUT_ROOT}/reports/${RUN_TAG}}"
LOG_DIR="${LORS3_LOG_DIR:-${LORS3_OUTPUT_ROOT}/logs/${RUN_TAG}}"
MEASURE_DIR="${LORS3_MEASURE_DIR:-${LORS3_OUTPUT_ROOT}/measurements/${RUN_TAG}}"
MANIFEST_PATH="${REPORT_DIR}/manifest.jsonl"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "${MEASURE_DIR}"
: > "${MANIFEST_PATH}"

ENERGY_PREFER_ZEUS="${ENERGY_PREFER_ZEUS:-1}"
ENERGY_GPU_SAMPLER_INTERVAL="${ENERGY_GPU_SAMPLER_INTERVAL:-1.0}"
GPU_COUNT="${LORS3_GPU_COUNT:-$(python - <<PY
devices = "${LORS3_DEVICE}".strip()
print(max(len([x for x in devices.split(",") if x.strip()]), 1))
PY
)}"

IMAGE_ROOT="$(get_image_root "${LORS3_DATASET}")"
MODEL_TAG="$(sanitize_component "${LORS3_UPSTREAM_BACKBONE}")_$(sanitize_component "${LORS3_TEXT_ENCODER}")"
RATIO_TAG="$(python - "${LORS3_RATIO}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
)"

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
  shift 3
  local zeus_args=()
  if [[ "${ENERGY_PREFER_ZEUS}" == "1" ]]; then
    zeus_args+=(--prefer_zeus)
  fi
  python "${PROJECT_ROOT}/tools/measure_command_energy.py" \
    --label "${label}" \
    --output_json "${measurement_path}" \
    --working_dir "${PROJECT_ROOT}" \
    --gpu_sampler_interval "${ENERGY_GPU_SAMPLER_INTERVAL}" \
    --tee_log "${log_path}" \
    "${zeus_args[@]}" \
    -- "$@"
}

compute_train_size() {
  python - "${LORS3_DATASET}" "${IMAGE_ROOT}" "${ANN_ROOT}" <<'PY'
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
print(max(1, int(round(total * ratio))))
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

scale_measurement_json() {
  local measured_json="$1"
  local scaled_json="$2"
  python - "${measured_json}" "${scaled_json}" "${LORS3_SCALE_FACTOR}" <<'PY'
import json
import sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
factor = float(sys.argv[3])
payload = json.loads(src.read_text(encoding="utf-8"))
scaled = dict(payload)
for key in ["wall_seconds", "gpu_energy_Wh", "gpu_energy_Wh_zeus", "gpu_energy_Wh_nvidia_smi", "cpu_energy_Wh", "total_energy_Wh"]:
    value = scaled.get(key)
    if isinstance(value, (int, float)):
        scaled[key] = float(value) * factor
scaled["scaling_factor"] = factor
scaled["scaling_note"] = "Scaled from reduced LoRS expert replay to approximate full distillation energy."
dst.write_text(json.dumps(scaled, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

TRAIN_COUNT="$(compute_train_size)"
BUDGET_SIZE="$(ratio_to_count "${TRAIN_COUNT}" "${LORS3_RATIO}")"

stage_log "LoRS 3% cross-arch + energy start"
stage_log "  dataset=${LORS3_DATASET} ratio=${LORS3_RATIO} budget=${BUDGET_SIZE} tag=${RATIO_TAG}"
stage_log "  upstream=${LORS3_UPSTREAM_BACKBONE}_${LORS3_TEXT_ENCODER} eval_backbones=${LORS3_EVAL_BACKBONES}"
stage_log "  measured experts=${LORS3_MEASURED_MAX_FILES}x${LORS3_MEASURED_MAX_EXPERTS}; full=${LORS3_FULL_MAX_FILES}x${LORS3_FULL_MAX_EXPERTS}; scale=${LORS3_SCALE_FACTOR}"

DISTILL_LOG="${LOG_DIR}/lors_${RATIO_TAG}_sampled_distill.log"
DISTILL_MEASURE="${MEASURE_DIR}/lors_${RATIO_TAG}_sampled_distill.json"
DISTILL_SCALED_MEASURE="${MEASURE_DIR}/lors_${RATIO_TAG}_scaled_full_distill.json"

measure_command "lors_${RATIO_TAG}_sampled_distill" "${DISTILL_MEASURE}" "${DISTILL_LOG}" \
  env CUDA_VISIBLE_DEVICES="${LORS3_DEVICE}" \
    LORS_DATASET="${LORS3_DATASET}" \
    LORS_IMAGE_ENCODER="${LORS3_UPSTREAM_BACKBONE}" \
    LORS_TEXT_ENCODER="${LORS3_TEXT_ENCODER}" \
    LORS_BUFFER_ROOT="${LORS3_BUFFER_ROOT:-buffers_arch3}" \
    LORS_LOG_ROOT="${LORS3_LOG_ROOT:-logged_files_arch3}" \
    LORS_FORCE_REBUILD_BUFFER="0" \
    LORS_FORCE_REDISTILL="1" \
    LORS_RUN_EVALUATE="0" \
    LORS_RUN_TAG="${RATIO_TAG}_arch3_energy" \
    LORS_RUN_NAME="lors_${LORS3_DATASET}_${RATIO_TAG}_${LORS3_UPSTREAM_BACKBONE}_arch3_${RUN_TIMESTAMP}" \
    LORS_NUM_QUERIES="${BUDGET_SIZE}" \
    LORS_ITERATION="${LORS_ITERATION:-3000}" \
    LORS_MAX_FILES="${LORS3_MEASURED_MAX_FILES}" \
    LORS_MAX_EXPERTS="${LORS3_MEASURED_MAX_EXPERTS}" \
    LORS_NUM_EVAL="${LORS_NUM_EVAL:-1}" \
    LORS_BATCH_TRAIN="${LORS_BATCH_TRAIN:-128}" \
    LORS_BATCH_TEST="${LORS_BATCH_TEST:-128}" \
    bash "${SCRIPT_DIR}/run_lors_baseline.sh"

scale_measurement_json "${DISTILL_MEASURE}" "${DISTILL_SCALED_MEASURE}"
CHECKPOINT_PATH="$(extract_checkpoint_from_distill_log "${DISTILL_LOG}" "${LORS_ITERATION:-3000}")"

append_manifest method "lors" dataset "${LORS3_DATASET}" budget_type "ratio" budget_value "${LORS3_RATIO}" budget_tag "${RATIO_TAG}" \
  eval_backbone "" stage "distill_selection" gpu_count "${GPU_COUNT}" budget_size "${BUDGET_SIZE}" checkpoint_path "${CHECKPOINT_PATH}" \
  log_path "${DISTILL_LOG}" measurement_path "${DISTILL_SCALED_MEASURE}" measured_measurement_path "${DISTILL_MEASURE}" \
  scaling_factor "${LORS3_SCALE_FACTOR}" upstream_backbone "${LORS3_UPSTREAM_BACKBONE}" skipped "0"

for eval_backbone in ${LORS3_EVAL_BACKBONES}; do
  EVAL_LOG="${LOG_DIR}/lors_${RATIO_TAG}_${eval_backbone}_evaluate.log"
  EVAL_MEASURE="${MEASURE_DIR}/lors_${RATIO_TAG}_${eval_backbone}_evaluate.json"
  stage_log "Measure LoRS downstream eval: ${RATIO_TAG} eval_backbone=${eval_backbone}"
  measure_command "lors_${RATIO_TAG}_${eval_backbone}_evaluate" "${EVAL_MEASURE}" "${EVAL_LOG}" \
    env CUDA_VISIBLE_DEVICES="${LORS3_DEVICE}" python "${PROJECT_ROOT}/evaluate_only.py" \
      --dataset "${LORS3_DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --model_checkpoint_root "${LORS_MODEL_CHECKPOINT_ROOT:-${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}}" \
      --image_encoder "${eval_backbone}" \
      --text_encoder "${LORS3_TEXT_ENCODER}" \
      --loss_type "${LORS_LOSS_TYPE:-InfoNCE}" \
      --ckpt_path "${CHECKPOINT_PATH}" \
      --num_eval "${LORS_NUM_EVAL:-1}" \
      --batch_train "${LORS_BATCH_TRAIN:-128}" \
      --batch_size_train "${LORS_BATCH_TRAIN:-128}" \
      --batch_size_test "${LORS_BATCH_TEST:-128}" \
      --disabled_wandb "${LORS_DISABLED_WANDB:-True}" \
      --no_aug
  append_manifest method "lors" dataset "${LORS3_DATASET}" budget_type "ratio" budget_value "${LORS3_RATIO}" budget_tag "${RATIO_TAG}" \
    eval_backbone "${eval_backbone}" stage "training_eval" gpu_count "${GPU_COUNT}" budget_size "${BUDGET_SIZE}" checkpoint_path "${CHECKPOINT_PATH}" \
    evaluate_log "${EVAL_LOG}" log_path "${EVAL_LOG}" measurement_path "${EVAL_MEASURE}" upstream_backbone "${LORS3_UPSTREAM_BACKBONE}" skipped "0"
done

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${MANIFEST_PATH}" \
  --output_dir "${REPORT_DIR}"

stage_log "LoRS 3% cross-arch + energy done"
stage_log "  architecture_bias=${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${REPORT_DIR}/energy_efficiency.csv"
stage_log "  detail=${REPORT_DIR}/supplemental_detail.csv"
