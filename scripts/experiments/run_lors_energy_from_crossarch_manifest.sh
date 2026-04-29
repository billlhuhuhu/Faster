#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

LORS_ENERGY_DATASET="${LORS_ENERGY_DATASET:-flickr}"
LORS_ENERGY_MANIFEST="${LORS_ENERGY_MANIFEST:-}"
LORS_ENERGY_RATIOS="${LORS_ENERGY_RATIOS:-0.05}"
LORS_ENERGY_EVAL_BACKBONES="${LORS_ENERGY_EVAL_BACKBONES:-nfnet resnet50 vit_b16}"
LORS_ENERGY_DEVICE="${LORS_ENERGY_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
LORS_ENERGY_OUTPUT_ROOT="${LORS_ENERGY_OUTPUT_ROOT:-artifacts/lors_energy_eval}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_TAG="${LORS_ENERGY_RUN_TAG:-${LORS_ENERGY_DATASET}_${RUN_TIMESTAMP}}"
REPORT_DIR="${LORS_ENERGY_REPORT_DIR:-${LORS_ENERGY_OUTPUT_ROOT}/reports/${RUN_TAG}}"
LOG_DIR="${LORS_ENERGY_LOG_DIR:-${LORS_ENERGY_OUTPUT_ROOT}/logs/${RUN_TAG}}"
MEASURE_DIR="${LORS_ENERGY_MEASURE_DIR:-${LORS_ENERGY_OUTPUT_ROOT}/measurements/${RUN_TAG}}"
ENERGY_MANIFEST="${REPORT_DIR}/manifest.jsonl"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "${MEASURE_DIR}"
: > "${ENERGY_MANIFEST}"

ENERGY_PREFER_ZEUS="${ENERGY_PREFER_ZEUS:-1}"
ENERGY_GPU_SAMPLER_INTERVAL="${ENERGY_GPU_SAMPLER_INTERVAL:-1.0}"
GPU_COUNT="${LORS_ENERGY_GPU_COUNT:-$(python - <<PY
devices = "${LORS_ENERGY_DEVICE}".strip()
print(max(len([x for x in devices.split(",") if x.strip()]), 1))
PY
)}"

IMAGE_ROOT="$(get_image_root "${LORS_ENERGY_DATASET}")"

find_latest_manifest() {
  find "${REPORT_ROOT}" -path "*lors_ratio_crossarch_${LORS_ENERGY_DATASET}_*/manifest.json" -type f 2>/dev/null | sort | tail -n 1
}

if [[ -z "${LORS_ENERGY_MANIFEST}" ]]; then
  LORS_ENERGY_MANIFEST="$(find_latest_manifest)"
fi

if [[ -z "${LORS_ENERGY_MANIFEST}" || ! -f "${LORS_ENERGY_MANIFEST}" ]]; then
  echo "Missing LoRS cross-architecture manifest. Set LORS_ENERGY_MANIFEST=/path/to/manifest.json" >&2
  exit 1
fi

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

stage_log "LoRS-only energy evaluation start"
stage_log "  manifest=${LORS_ENERGY_MANIFEST}"
stage_log "  dataset=${LORS_ENERGY_DATASET} ratios=${LORS_ENERGY_RATIOS} eval_backbones=${LORS_ENERGY_EVAL_BACKBONES}"
stage_log "  device=${LORS_ENERGY_DEVICE} gpu_count=${GPU_COUNT}"

python - "${LORS_ENERGY_MANIFEST}" "${REPORT_DIR}/selected_runs.jsonl" "${LORS_ENERGY_RATIOS}" "${LORS_ENERGY_EVAL_BACKBONES}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
ratios = {float(x) for x in sys.argv[3].split() if x.strip()}
backbones = {x for x in sys.argv[4].split() if x.strip()}
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
rows = []
for item in payload.get("runs", []):
    ratio = float(item.get("ratio"))
    eval_backbone = str(item.get("eval_backbone", ""))
    if ratios and ratio not in ratios:
        continue
    if backbones and eval_backbone not in backbones:
        continue
    item = dict(item)
    item["dataset"] = payload.get("dataset")
    rows.append(item)
out_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
print(f"selected LoRS runs: {len(rows)}")
PY

while IFS= read -r item; do
  [[ -z "${item}" ]] && continue
  ratio="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["ratio"])' "${item}")"
  budget_size="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["budget_size"])' "${item}")"
  eval_backbone="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["eval_backbone"])' "${item}")"
  checkpoint_path="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["checkpoint_path"])' "${item}")"
  distill_backbone="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("distill_backbone","nfnet"))' "${item}")"
  budget_tag="$(python - "${ratio}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
)"
  log_path="${LOG_DIR}/${budget_tag}_${eval_backbone}_evaluate.log"
  measurement_path="${MEASURE_DIR}/${budget_tag}_${eval_backbone}_evaluate.json"

  if [[ ! -f "${checkpoint_path}" ]]; then
    stage_log "Missing checkpoint, skip LoRS ${budget_tag}/${eval_backbone}: ${checkpoint_path}"
    append_manifest method "lors" dataset "${LORS_ENERGY_DATASET}" budget_type "ratio" budget_value "${ratio}" budget_tag "${budget_tag}" \
      eval_backbone "${eval_backbone}" stage "missing_checkpoint" seconds "0" gpu_count "${GPU_COUNT}" checkpoint_path "${checkpoint_path}" skipped "1"
    continue
  fi

  stage_log "Measure LoRS evaluate: ${budget_tag} eval_backbone=${eval_backbone}"
  measure_command "lors_${budget_tag}_${eval_backbone}_evaluate" "${measurement_path}" "${log_path}" \
    env CUDA_VISIBLE_DEVICES="${LORS_ENERGY_DEVICE}" python "${PROJECT_ROOT}/evaluate_only.py" \
      --dataset "${LORS_ENERGY_DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --model_checkpoint_root "${LORS_MODEL_CHECKPOINT_ROOT:-${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}}" \
      --image_encoder "${eval_backbone}" \
      --text_encoder "${LORS_RATIO_TEXT_ENCODER:-bert}" \
      --loss_type "${LORS_LOSS_TYPE:-InfoNCE}" \
      --ckpt_path "${checkpoint_path}" \
      --num_eval "${LORS_NUM_EVAL:-1}" \
      --batch_train "${LORS_BATCH_TRAIN:-128}" \
      --batch_size_train "${LORS_BATCH_TRAIN:-128}" \
      --batch_size_test "${LORS_BATCH_TEST:-128}" \
      --disabled_wandb "${LORS_DISABLED_WANDB:-True}" \
      --no_aug

  append_manifest method "lors" dataset "${LORS_ENERGY_DATASET}" budget_type "ratio" budget_value "${ratio}" budget_tag "${budget_tag}" \
    eval_backbone "${eval_backbone}" stage "training_eval" gpu_count "${GPU_COUNT}" budget_size "${budget_size}" \
    distill_backbone "${distill_backbone}" checkpoint_path "${checkpoint_path}" evaluate_log "${log_path}" \
    log_path "${log_path}" measurement_path "${measurement_path}" skipped "0"
done < "${REPORT_DIR}/selected_runs.jsonl"

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${ENERGY_MANIFEST}" \
  --output_dir "${REPORT_DIR}"

stage_log "LoRS-only energy evaluation done"
stage_log "  detail=${REPORT_DIR}/supplemental_detail.csv"
stage_log "  architecture_bias=${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${REPORT_DIR}/energy_efficiency.csv"
stage_log "  measurements=${MEASURE_DIR}"
