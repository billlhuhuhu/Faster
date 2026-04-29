#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

LORS_SCALE_DATASET="${LORS_SCALE_DATASET:-flickr}"
LORS_SCALE_RATIO="${LORS_SCALE_RATIO:-0.05}"
LORS_SCALE_DISTILL_BACKBONE="${LORS_SCALE_DISTILL_BACKBONE:-nfnet}"
LORS_SCALE_TEXT_ENCODER="${LORS_SCALE_TEXT_ENCODER:-bert}"
LORS_SCALE_DEVICE="${LORS_SCALE_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
LORS_SCALE_OUTPUT_ROOT="${LORS_SCALE_OUTPUT_ROOT:-artifacts/lors_scaled_energy_5pct}"
LORS_SCALE_MEASURED_MAX_FILES="${LORS_SCALE_MEASURED_MAX_FILES:-1}"
LORS_SCALE_MEASURED_MAX_EXPERTS="${LORS_SCALE_MEASURED_MAX_EXPERTS:-5}"
LORS_SCALE_FULL_MAX_FILES="${LORS_SCALE_FULL_MAX_FILES:-1}"
LORS_SCALE_FULL_MAX_EXPERTS="${LORS_SCALE_FULL_MAX_EXPERTS:-100}"
LORS_SCALE_FACTOR="${LORS_SCALE_FACTOR:-$(python - <<PY
measured = max(float("${LORS_SCALE_MEASURED_MAX_FILES}") * float("${LORS_SCALE_MEASURED_MAX_EXPERTS}"), 1.0)
full = max(float("${LORS_SCALE_FULL_MAX_FILES}") * float("${LORS_SCALE_FULL_MAX_EXPERTS}"), measured)
print(full / measured)
PY
)}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_TAG="${LORS_SCALE_RUN_TAG:-${LORS_SCALE_DATASET}_ratio05_scaled_${RUN_TIMESTAMP}}"
REPORT_DIR="${LORS_SCALE_REPORT_DIR:-${LORS_SCALE_OUTPUT_ROOT}/reports/${RUN_TAG}}"
LOG_DIR="${LORS_SCALE_LOG_DIR:-${LORS_SCALE_OUTPUT_ROOT}/logs/${RUN_TAG}}"
MEASURE_DIR="${LORS_SCALE_MEASURE_DIR:-${LORS_SCALE_OUTPUT_ROOT}/measurements/${RUN_TAG}}"
ENERGY_MANIFEST="${REPORT_DIR}/manifest.jsonl"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "${MEASURE_DIR}"
: > "${ENERGY_MANIFEST}"

ENERGY_PREFER_ZEUS="${ENERGY_PREFER_ZEUS:-1}"
ENERGY_GPU_SAMPLER_INTERVAL="${ENERGY_GPU_SAMPLER_INTERVAL:-1.0}"
GPU_COUNT="${LORS_SCALE_GPU_COUNT:-$(python - <<PY
devices = "${LORS_SCALE_DEVICE}".strip()
print(max(len([x for x in devices.split(",") if x.strip()]), 1))
PY
)}"

IMAGE_ROOT="$(get_image_root "${LORS_SCALE_DATASET}")"
MODEL_TAG="$(sanitize_component "${LORS_SCALE_DISTILL_BACKBONE}")_$(sanitize_component "${LORS_SCALE_TEXT_ENCODER}")"
RATIO_TAG="$(python - "${LORS_SCALE_RATIO}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
)"

append_manifest() {
  python - "$ENERGY_MANIFEST" "$@" <<'PY'
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
    "${zeus_args[@]}" \
    -- "$@" > "${log_path}" 2>&1
}

compute_train_size() {
  python - "${LORS_SCALE_DATASET}" "${IMAGE_ROOT}" "${ANN_ROOT}" <<'PY'
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
  python - "${measured_json}" "${scaled_json}" "${LORS_SCALE_FACTOR}" <<'PY'
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
scaled["scaling_note"] = "Scaled from a reduced LoRS expert subset to approximate full distillation energy."
dst.write_text(json.dumps(scaled, ensure_ascii=False, indent=2), encoding="utf-8")
print(dst)
PY
}

TRAIN_COUNT="$(compute_train_size)"
BUDGET_SIZE="$(ratio_to_count "${TRAIN_COUNT}" "${LORS_SCALE_RATIO}")"

stage_log "LoRS scaled energy start"
stage_log "  dataset=${LORS_SCALE_DATASET} ratio=${LORS_SCALE_RATIO} budget=${BUDGET_SIZE} tag=${RATIO_TAG}"
stage_log "  measured experts=${LORS_SCALE_MEASURED_MAX_FILES}x${LORS_SCALE_MEASURED_MAX_EXPERTS}; full experts=${LORS_SCALE_FULL_MAX_FILES}x${LORS_SCALE_FULL_MAX_EXPERTS}; scale=${LORS_SCALE_FACTOR}"
stage_log "  device=${LORS_SCALE_DEVICE}"
stage_log "  scope=selection/distillation energy only; downstream evaluate_only.py is not run"

DISTILL_LOG="${LOG_DIR}/lors_${RATIO_TAG}_sampled_distill.log"
DISTILL_MEASURE="${MEASURE_DIR}/lors_${RATIO_TAG}_sampled_distill.json"
DISTILL_SCALED_MEASURE="${MEASURE_DIR}/lors_${RATIO_TAG}_scaled_full_distill.json"

measure_command "lors_${RATIO_TAG}_sampled_distill" "${DISTILL_MEASURE}" "${DISTILL_LOG}" \
  env CUDA_VISIBLE_DEVICES="${LORS_SCALE_DEVICE}" \
    LORS_DATASET="${LORS_SCALE_DATASET}" \
    LORS_IMAGE_ENCODER="${LORS_SCALE_DISTILL_BACKBONE}" \
    LORS_TEXT_ENCODER="${LORS_SCALE_TEXT_ENCODER}" \
    LORS_BUFFER_ROOT="${LORS_SCALE_BUFFER_ROOT:-buffers_formal_v2}" \
    LORS_LOG_ROOT="${LORS_SCALE_LOG_ROOT:-logged_files_formal_v2}" \
    LORS_FORCE_REBUILD_BUFFER="0" \
    LORS_FORCE_REDISTILL="1" \
    LORS_RUN_TAG="${RATIO_TAG}_energy_sample" \
    LORS_RUN_NAME="lors_${LORS_SCALE_DATASET}_${RATIO_TAG}_energy_sample_${RUN_TIMESTAMP}" \
    LORS_NUM_QUERIES="${BUDGET_SIZE}" \
    LORS_ITERATION="${LORS_ITERATION:-3000}" \
    LORS_MAX_FILES="${LORS_SCALE_MEASURED_MAX_FILES}" \
    LORS_MAX_EXPERTS="${LORS_SCALE_MEASURED_MAX_EXPERTS}" \
    LORS_NUM_EVAL="${LORS_NUM_EVAL:-1}" \
    LORS_BATCH_TRAIN="${LORS_BATCH_TRAIN:-128}" \
    LORS_BATCH_TEST="${LORS_BATCH_TEST:-128}" \
    bash "${SCRIPT_DIR}/run_lors_baseline.sh"

scale_measurement_json "${DISTILL_MEASURE}" "${DISTILL_SCALED_MEASURE}" >/dev/null
CHECKPOINT_PATH="$(extract_checkpoint_from_distill_log "${DISTILL_LOG}" "${LORS_ITERATION:-3000}")"

append_manifest method "lors" dataset "${LORS_SCALE_DATASET}" budget_type "ratio" budget_value "${LORS_SCALE_RATIO}" budget_tag "${RATIO_TAG}" \
  eval_backbone "" stage "distill_selection" gpu_count "${GPU_COUNT}" budget_size "${BUDGET_SIZE}" checkpoint_path "${CHECKPOINT_PATH}" \
  log_path "${DISTILL_LOG}" measurement_path "${DISTILL_SCALED_MEASURE}" measured_measurement_path "${DISTILL_MEASURE}" \
  scaling_factor "${LORS_SCALE_FACTOR}" measured_max_files "${LORS_SCALE_MEASURED_MAX_FILES}" measured_max_experts "${LORS_SCALE_MEASURED_MAX_EXPERTS}" \
  full_max_files "${LORS_SCALE_FULL_MAX_FILES}" full_max_experts "${LORS_SCALE_FULL_MAX_EXPERTS}" skipped "0"

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${ENERGY_MANIFEST}" \
  --output_dir "${REPORT_DIR}"

stage_log "LoRS scaled energy done"
stage_log "  architecture_bias=${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${REPORT_DIR}/energy_efficiency.csv"
stage_log "  detail=${REPORT_DIR}/supplemental_detail.csv"
stage_log "  sampled_distill_measure=${DISTILL_MEASURE}"
stage_log "  scaled_distill_measure=${DISTILL_SCALED_MEASURE}"
