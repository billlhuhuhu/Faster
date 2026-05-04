#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${ARCH_SUPP_DATASET:-flickr}"
RATIO="${ARCH_SUPP_RATIO:-0.03}"
RATIO_TAG="$(python - "${RATIO}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
)"
UPSTREAM_BACKBONE="${ARCH_SUPP_UPSTREAM_BACKBONE:-resnet10}"
TEXT_ENCODER="${ARCH_SUPP_TEXT_ENCODER:-bert}"
EVAL_BACKBONES="${ARCH_SUPP_EVAL_BACKBONES:-${ARCH_SUPP_EVAL_BACKBONE:-nfnet resnet50 vit_b16 resnet10}}"
METHODS="${ARCH_SUPP_METHODS:-repblend lors}"
RUN_TAG="${ARCH_SUPP_RUN_TAG:-resnet10_downstream_supp_$(date '+%Y%m%d_%H%M%S')}"
OUTPUT_ROOT="${ARCH_SUPP_OUTPUT_ROOT:-artifacts/arch_bias_energy_3pct/resnet10_downstream_supplement}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_DIR="${RUN_ROOT}/logs"
MEASURE_DIR="${RUN_ROOT}/measurements"
REPORT_DIR="${RUN_ROOT}/reports"
MANIFEST_PATH="${REPORT_DIR}/manifest.jsonl"
IMAGE_ROOT="$(get_image_root "${DATASET}")"

REPBLEND_ROOT="${REPBLEND_ROOT:-${PROJECT_ROOT}/RepBlend}"
REPBLEND_LOGGED_FILES_ROOT="${REPBLEND_LOGGED_FILES_ROOT:-${REPBLEND_ROOT}/logged_files}"
REPBLEND_ITERATION="${REPBLEND_ITERATION:-3000}"
REPBLEND_LOSS_TYPE="${REPBLEND_LOSS_TYPE:-WBCE}"
REPBLEND_CUDA_VISIBLE_DEVICES="${REPBLEND_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

LORS_LOG_ROOT="${LORS_LOG_ROOT:-${PROJECT_ROOT}/logged_files_arch3}"
LORS_ITERATION="${LORS_ITERATION:-500}"
LORS_LOSS_TYPE="${LORS_LOSS_TYPE:-InfoNCE}"
LORS_CUDA_VISIBLE_DEVICES="${LORS_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

ENERGY_PREFER_ZEUS="${ENERGY_PREFER_ZEUS:-1}"
ENERGY_GPU_SAMPLER_INTERVAL="${ENERGY_GPU_SAMPLER_INTERVAL:-1.0}"
GPU_COUNT="${ARCH_SUPP_GPU_COUNT:-1}"

mkdir -p "${LOG_DIR}" "${MEASURE_DIR}" "${REPORT_DIR}"
: > "${MANIFEST_PATH}"

method_enabled() {
  local name="$1"
  [[ " ${METHODS} " == *" ${name} "* ]]
}

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
  local working_dir="$4"
  shift 4
  local zeus_args=()
  if [[ "${ENERGY_PREFER_ZEUS}" == "1" ]]; then
    zeus_args+=(--prefer_zeus)
  fi
  python "${PROJECT_ROOT}/tools/measure_command_energy.py" \
    --label "${label}" \
    --output_json "${measurement_path}" \
    --working_dir "${working_dir}" \
    --gpu_sampler_interval "${ENERGY_GPU_SAMPLER_INTERVAL}" \
    --tee_log "${log_path}" \
    "${zeus_args[@]}" \
    -- "$@"
}

find_repblend_checkpoint() {
  if [[ -n "${REPBLEND_RESNET10_CKPT_PATH:-}" ]]; then
    echo "${REPBLEND_RESNET10_CKPT_PATH}"
    return 0
  fi
  if [[ -n "${REPBLEND_RESNET10_DISTILL_LOG:-}" ]]; then
    python - "${REPBLEND_RESNET10_DISTILL_LOG}" "${REPBLEND_ROOT}" "${REPBLEND_ITERATION}" <<'PY'
import re
import sys
from pathlib import Path
log_path = Path(sys.argv[1])
root = Path(sys.argv[2])
iteration = sys.argv[3]
text = log_path.read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"Saving to (.+)", text)
if not matches:
    raise SystemExit(1)
save_dir = Path(matches[-1].strip())
if not save_dir.is_absolute():
    save_dir = root / save_dir
print(save_dir / f"distilled_{iteration}.pt")
PY
    return 0
  fi
  python - "${PROJECT_ROOT}" "${REPBLEND_ROOT}" "${UPSTREAM_BACKBONE}" "${REPBLEND_ITERATION}" "${REPBLEND_LOGGED_FILES_ROOT}" <<'PY'
import re
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
repblend_root = Path(sys.argv[2])
upstream = sys.argv[3]
iteration = sys.argv[4]
logged_root = Path(sys.argv[5])

def ckpt_from_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if f"'image_encoder': '{upstream}'" not in text and f'"image_encoder": "{upstream}"' not in text and f"--image_encoder {upstream}" not in text:
        return None
    matches = re.findall(r"Saving to (.+)", text)
    if not matches:
        return None
    save_dir = Path(matches[-1].strip())
    if not save_dir.is_absolute():
        save_dir = repblend_root / save_dir
    ckpt = save_dir / f"distilled_{iteration}.pt"
    return ckpt if ckpt.exists() else None

log_roots = [
    project_root / "artifacts/arch_bias_energy_3pct",
    project_root / "artifacts/arch3pct_nfnet_repblend_lors_clean",
    project_root / "artifacts/arch_bias_energy_5pct_rerun",
]
found = []
for root in log_roots:
    if not root.exists():
        continue
    for log_path in root.rglob("*repblend*distill*.log"):
        ckpt = ckpt_from_log(log_path)
        if ckpt is not None:
            found.append((log_path.stat().st_mtime, ckpt))
if found:
    print(sorted(found)[-1][1])
    raise SystemExit(0)

candidates = []
if logged_root.exists():
    candidates.extend(logged_root.rglob(f"distilled_{iteration}.pt"))
if candidates:
    print(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
    raise SystemExit(0)
raise SystemExit(1)
PY
}

find_lors_checkpoint() {
  if [[ -n "${LORS_RESNET10_CKPT_PATH:-}" ]]; then
    echo "${LORS_RESNET10_CKPT_PATH}"
    return 0
  fi
  python - "${LORS_LOG_ROOT}" "${DATASET}" "${UPSTREAM_BACKBONE}" "${LORS_ITERATION}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]) / sys.argv[2]
upstream = sys.argv[3]
iteration = sys.argv[4]
patterns = [
    f"*{upstream}*/distilled_{iteration}.pt",
    f"*{upstream}*/distilled_3000.pt",
    f"*{upstream}*/distilled_500.pt",
]
found = []
if root.exists():
    for pattern in patterns:
        found.extend(root.glob(pattern))
if found:
    print(sorted(set(found), key=lambda p: p.stat().st_mtime)[-1])
    raise SystemExit(0)
raise SystemExit(1)
PY
}

run_eval() {
  local method="$1"
  local ckpt_path="$2"
  local loss_type="$3"
  local visible_devices="$4"
  local batch_train="$5"
  local batch_test="$6"
  local model_root="$7"
  local log_path="${LOG_DIR}/${method}_${RATIO_TAG}_${EVAL_BACKBONE}_evaluate.log"
  local measurement_path="${MEASURE_DIR}/${method}_${RATIO_TAG}_${EVAL_BACKBONE}_evaluate.json"
  local eval_extra_args=()

  if [[ "${EVAL_BACKBONE}" == "vit_b16" ]]; then
    eval_extra_args+=(
      --image_trainable true
      --text_trainable false
      --lr_teacher_img "${ARCH_SUPP_VIT_LR_IMG:-0.001}"
      --lr_teacher_txt "${ARCH_SUPP_VIT_LR_TXT:-0.05}"
    )
  fi

  if [[ -z "${ckpt_path}" || ! -f "${ckpt_path}" ]]; then
    stage_log "Missing ${method} checkpoint, skip ${EVAL_BACKBONE}: ${ckpt_path}"
    append_manifest method "${method}" dataset "${DATASET}" budget_type "ratio" budget_value "${RATIO}" budget_tag "${RATIO_TAG}" \
      eval_backbone "${EVAL_BACKBONE}" stage "missing_checkpoint" gpu_count "${GPU_COUNT}" checkpoint_path "${ckpt_path}" skipped "1"
    return 0
  fi

  stage_log "Measure ${method} downstream eval: upstream=${UPSTREAM_BACKBONE}+${TEXT_ENCODER} eval=${EVAL_BACKBONE}+${TEXT_ENCODER}"
  measure_command "${method}_${RATIO_TAG}_${EVAL_BACKBONE}_evaluate" "${measurement_path}" "${log_path}" "${PROJECT_ROOT}" \
    env CUDA_VISIBLE_DEVICES="${visible_devices}" \
    python "${PROJECT_ROOT}/evaluate_only.py" \
      --dataset "${DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --model_checkpoint_root "${model_root}" \
      --image_encoder "${EVAL_BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --loss_type "${loss_type}" \
      --ckpt_path "${ckpt_path}" \
      --num_eval "${ARCH_SUPP_NUM_EVAL:-1}" \
      --epoch_eval_train "${ARCH_SUPP_EPOCH_EVAL_TRAIN:-50}" \
      --batch_train "${batch_train}" \
      --batch_size_train "${batch_train}" \
      --batch_size_test "${batch_test}" \
      --disabled_wandb True \
      --no_aug \
      "${eval_extra_args[@]}"

  append_manifest method "${method}" dataset "${DATASET}" budget_type "ratio" budget_value "${RATIO}" budget_tag "${RATIO_TAG}" \
    eval_backbone "${EVAL_BACKBONE}" stage "training_eval" gpu_count "${GPU_COUNT}" checkpoint_path "${ckpt_path}" \
    evaluate_log "${log_path}" log_path "${log_path}" measurement_path "${measurement_path}" skipped "0"
}

stage_log "Supplement downstream resnet10 evaluation start"
stage_log "  methods=${METHODS}"
stage_log "  dataset=${DATASET} ratio=${RATIO_TAG} upstream=${UPSTREAM_BACKBONE}+${TEXT_ENCODER} eval_backbones=${EVAL_BACKBONES}"
stage_log "  output=${RUN_ROOT}"

if method_enabled repblend; then
  REPBLEND_CKPT="$(find_repblend_checkpoint || true)"
  stage_log "RepBlend checkpoint: ${REPBLEND_CKPT:-<not found>}"
  for EVAL_BACKBONE in ${EVAL_BACKBONES}; do
    if [[ "${EVAL_BACKBONE}" == "vit_b16" ]]; then
      ARCH_SUPP_EPOCH_EVAL_TRAIN="${REPBLEND_VIT_EPOCH_EVAL_TRAIN:-300}" \
      run_eval "repblend" "${REPBLEND_CKPT:-}" "${REPBLEND_LOSS_TYPE}" "${REPBLEND_CUDA_VISIBLE_DEVICES}" \
        "${REPBLEND_VIT_BATCH_TRAIN:-32}" "${REPBLEND_VIT_BATCH_TEST:-64}" \
        "${REPBLEND_MODEL_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}"
    else
      run_eval "repblend" "${REPBLEND_CKPT:-}" "${REPBLEND_LOSS_TYPE}" "${REPBLEND_CUDA_VISIBLE_DEVICES}" \
        "${REPBLEND_EVAL_BATCH_TRAIN:-32}" "${REPBLEND_EVAL_BATCH_TEST:-64}" \
        "${REPBLEND_MODEL_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}"
    fi
  done
fi

if method_enabled lors; then
  LORS_CKPT="$(find_lors_checkpoint || true)"
  stage_log "LoRS checkpoint: ${LORS_CKPT:-<not found>}"
  for EVAL_BACKBONE in ${EVAL_BACKBONES}; do
    if [[ "${EVAL_BACKBONE}" == "vit_b16" ]]; then
      ARCH_SUPP_EPOCH_EVAL_TRAIN="${LORS_VIT_EPOCH_EVAL_TRAIN:-300}" \
      run_eval "lors" "${LORS_CKPT:-}" "${LORS_LOSS_TYPE}" "${LORS_CUDA_VISIBLE_DEVICES}" \
        "${LORS_VIT_BATCH_TRAIN:-32}" "${LORS_VIT_BATCH_TEST:-64}" \
        "${LORS_MODEL_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}"
    else
      run_eval "lors" "${LORS_CKPT:-}" "${LORS_LOSS_TYPE}" "${LORS_CUDA_VISIBLE_DEVICES}" \
        "${LORS_EVAL_BATCH_TRAIN:-32}" "${LORS_EVAL_BATCH_TEST:-64}" \
        "${LORS_MODEL_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}"
    fi
  done
fi

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${MANIFEST_PATH}" \
  --output_dir "${REPORT_DIR}"

stage_log "Supplement downstream resnet10 evaluation done"
stage_log "  detail=${REPORT_DIR}/supplemental_detail.csv"
stage_log "  architecture_bias=${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${REPORT_DIR}/energy_efficiency.csv"
