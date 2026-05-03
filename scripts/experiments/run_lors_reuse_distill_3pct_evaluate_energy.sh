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
LORS3_RUN_TAG="${LORS3_RUN_TAG:-lors_3pct_reuse_distill_eval_$(date '+%Y%m%d_%H%M%S')}"

# Energy accounting for selection/setup:
#   scaled buffer setup energy = one measured buffer energy * reused buffer count
#   selection energy = scaled buffer setup energy + measured LoRS distill energy
LORS3_SINGLE_BUFFER_GPU_WH="${LORS3_SINGLE_BUFFER_GPU_WH:-1068.1346074553746}"
LORS3_REUSE_BUFFER_COUNT="${LORS3_REUSE_BUFFER_COUNT:-10}"
LORS3_DISTILL_GPU_WH="${LORS3_DISTILL_GPU_WH:-2185.422607921632}"
LORS3_DISTILL_SECONDS="${LORS3_DISTILL_SECONDS:-41025.04}"
LORS3_DISTILL_MEASURE_PATH="${LORS3_DISTILL_MEASURE_PATH:-artifacts/arch_bias_energy_3pct/lors/measurements/lors_3pct_from10buffers_20260502_122057/lors_ratio_03_distill_from10buffers.json}"
LORS3_SINGLE_BUFFER_SECONDS="${LORS3_SINGLE_BUFFER_SECONDS:-13518.45}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
REPORT_DIR="${LORS3_REPORT_DIR:-${LORS3_OUTPUT_ROOT}/reports/${LORS3_RUN_TAG}}"
LOG_DIR="${LORS3_LOG_DIR:-${LORS3_OUTPUT_ROOT}/logs/${LORS3_RUN_TAG}}"
MEASURE_DIR="${LORS3_MEASURE_DIR:-${LORS3_OUTPUT_ROOT}/measurements/${LORS3_RUN_TAG}}"
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

find_checkpoint() {
  if [[ -n "${LORS3_CHECKPOINT_PATH:-}" ]]; then
    python - "${PROJECT_ROOT}" "${LORS3_CHECKPOINT_PATH}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = Path(sys.argv[2])
if not path.is_absolute():
    path = root / path

if path.is_file():
    print(path)
    raise SystemExit(0)

if path.is_dir():
    candidates = []
    for pattern in ("distilled_*.pt", "**/distilled_*.pt", "*.pt", "**/*.pt"):
        for ckpt in path.glob(pattern):
            if ckpt.is_file():
                candidates.append((ckpt.stat().st_mtime, ckpt))
    if candidates:
        candidates.sort(reverse=True)
        print(candidates[0][1])
        raise SystemExit(0)

raise SystemExit(f"LORS3_CHECKPOINT_PATH does not point to a checkpoint or searchable directory: {path}")
PY
    return 0
  fi
  python - "${PROJECT_ROOT}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
patterns = [
    "artifacts/arch_bias_energy_3pct/lors/logs/lors_3pct_from10buffers_*/lors_ratio_03_distill_from10buffers.log",
    "artifacts/arch_bias_energy_3pct/lors/logs/*/lors_ratio_03_distill_from10buffers.log",
    "experiments/logs/lors_baseline_flickr_ratio_03_from10buffers_energy_*/distill.log",
    "logged_files/flickr/*/distilled_3000.pt",
    "logged_files/flickr/*/distilled_*.pt",
    "logged_files/flickr/**/distilled_*.pt",
]

candidates = []
for pattern in patterns:
    for path in root.glob(pattern):
        if path.suffix == ".pt":
            candidates.append((path.stat().st_mtime, path))
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"Saving to (.+)", text)
        for match in matches[-1:]:
            save_dir = Path(match.strip())
            if not save_dir.is_absolute():
                save_dir = root / save_dir
            ckpt = save_dir / "distilled_3000.pt"
            if ckpt.exists():
                candidates.append((ckpt.stat().st_mtime, ckpt))
            else:
                distilled = sorted(save_dir.glob("distilled_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if distilled:
                    candidates.append((distilled[0].stat().st_mtime, distilled[0]))

if not candidates:
    raise SystemExit("No distilled checkpoint found. Set LORS3_CHECKPOINT_PATH=/path/to/distilled_*.pt or /path/to/logged_files/flickr")
candidates.sort(reverse=True)
print(candidates[0][1])
PY
}

write_combined_selection_measurement() {
  local out_json="$1"
  python - "${out_json}" \
    "${LORS3_SINGLE_BUFFER_GPU_WH}" \
    "${LORS3_REUSE_BUFFER_COUNT}" \
    "${LORS3_DISTILL_GPU_WH}" \
    "${LORS3_SINGLE_BUFFER_SECONDS}" \
    "${LORS3_DISTILL_SECONDS}" \
    "${LORS3_DISTILL_MEASURE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
single_buffer_wh = float(sys.argv[2])
buffer_count = int(float(sys.argv[3]))
distill_wh = float(sys.argv[4])
single_buffer_seconds = float(sys.argv[5])
distill_seconds = float(sys.argv[6])
distill_measure_path = sys.argv[7]

buffer_wh = single_buffer_wh * buffer_count
buffer_seconds = single_buffer_seconds * buffer_count
payload = {
    "label": "lors_ratio_03_selection_combined_buffer_x10_plus_distill",
    "wall_seconds": buffer_seconds + distill_seconds,
    "gpu_energy_Wh": buffer_wh + distill_wh,
    "gpu_energy_method": "manual_scaled_plus_measured_distill",
    "gpu_energy_Wh_buffer_scaled": buffer_wh,
    "gpu_energy_Wh_distill_measured": distill_wh,
    "cpu_energy_Wh": None,
    "cpu_energy_method": "unavailable",
    "total_energy_Wh": buffer_wh + distill_wh,
    "single_buffer_gpu_energy_Wh": single_buffer_wh,
    "reused_buffer_count": buffer_count,
    "single_buffer_wall_seconds": single_buffer_seconds,
    "scaled_buffer_wall_seconds": buffer_seconds,
    "distill_wall_seconds": distill_seconds,
    "distill_measurement_path": distill_measure_path,
    "note": "Selection energy is manually composed as one measured LoRS buffer energy times 10 plus the completed from-10-buffers distillation measurement.",
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

TRAIN_COUNT="$(compute_train_size)"
BUDGET_SIZE="$(ratio_to_count "${TRAIN_COUNT}" "${LORS3_RATIO}")"
CHECKPOINT_PATH="$(find_checkpoint)"
SELECTION_MEASURE="${MEASURE_DIR}/lors_${RATIO_TAG}_selection_combined_buffer_x${LORS3_REUSE_BUFFER_COUNT}_plus_distill.json"
write_combined_selection_measurement "${SELECTION_MEASURE}"

stage_log "LoRS 3% reuse-distill evaluate + energy start"
stage_log "  dataset=${LORS3_DATASET} ratio=${LORS3_RATIO} budget=${BUDGET_SIZE} tag=${RATIO_TAG}"
stage_log "  checkpoint=${CHECKPOINT_PATH}"
stage_log "  combined selection measure=${SELECTION_MEASURE}"
stage_log "  eval_backbones=${LORS3_EVAL_BACKBONES}; vit_b16 uses low_lr_finetune"

append_manifest method "lors" dataset "${LORS3_DATASET}" budget_type "ratio" budget_value "${LORS3_RATIO}" budget_tag "${RATIO_TAG}" \
  eval_backbone "" stage "selection_combined" gpu_count "${GPU_COUNT}" budget_size "${BUDGET_SIZE}" checkpoint_path "${CHECKPOINT_PATH}" \
  measurement_path "${SELECTION_MEASURE}" upstream_backbone "${LORS3_UPSTREAM_BACKBONE}" skipped "0"

for eval_backbone in ${LORS3_EVAL_BACKBONES}; do
  EVAL_LOG="${LOG_DIR}/lors_${RATIO_TAG}_${eval_backbone}_evaluate.log"
  EVAL_MEASURE="${MEASURE_DIR}/lors_${RATIO_TAG}_${eval_backbone}_evaluate.json"
  eval_args=(
    --dataset "${LORS3_DATASET}"
    --image_root "${IMAGE_ROOT}"
    --ann_root "${ANN_ROOT}"
    --model_checkpoint_root "${LORS_MODEL_CHECKPOINT_ROOT:-${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}}"
    --image_encoder "${eval_backbone}"
    --text_encoder "${LORS3_TEXT_ENCODER}"
    --loss_type "${LORS_LOSS_TYPE:-InfoNCE}"
    --ckpt_path "${CHECKPOINT_PATH}"
    --num_eval "${LORS_NUM_EVAL:-1}"
    --batch_train "${LORS_BATCH_TRAIN:-64}"
    --batch_size_train "${LORS_BATCH_TRAIN:-64}"
    --batch_size_test "${LORS_BATCH_TEST:-64}"
    --disabled_wandb "${LORS_DISABLED_WANDB:-True}"
    --no_aug
  )
  if [[ "${eval_backbone}" == "vit_b16" ]]; then
    eval_args+=(
      --image_trainable true
      --text_trainable false
      --lr_teacher_img "${LORS_VIT_LOWLR_IMG:-0.001}"
      --lr_teacher_txt "${LORS_VIT_LOWLR_TXT:-0.05}"
      --epoch_eval_train "${LORS_VIT_EPOCH_EVAL_TRAIN:-300}"
      --batch_train "${LORS_VIT_BATCH_TRAIN:-32}"
      --batch_size_train "${LORS_VIT_BATCH_TRAIN:-32}"
      --batch_size_test "${LORS_VIT_BATCH_TEST:-64}"
    )
  fi

  stage_log "Measure LoRS downstream eval: ${RATIO_TAG} eval_backbone=${eval_backbone}"
  measure_command "lors_${RATIO_TAG}_${eval_backbone}_evaluate" "${EVAL_MEASURE}" "${EVAL_LOG}" \
    env CUDA_VISIBLE_DEVICES="${LORS3_DEVICE}" python "${PROJECT_ROOT}/evaluate_only.py" "${eval_args[@]}"
  append_manifest method "lors" dataset "${LORS3_DATASET}" budget_type "ratio" budget_value "${LORS3_RATIO}" budget_tag "${RATIO_TAG}" \
    eval_backbone "${eval_backbone}" stage "training_eval" gpu_count "${GPU_COUNT}" budget_size "${BUDGET_SIZE}" checkpoint_path "${CHECKPOINT_PATH}" \
    evaluate_log "${EVAL_LOG}" log_path "${EVAL_LOG}" measurement_path "${EVAL_MEASURE}" upstream_backbone "${LORS3_UPSTREAM_BACKBONE}" skipped "0"
done

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${MANIFEST_PATH}" \
  --output_dir "${REPORT_DIR}"

stage_log "LoRS 3% reuse-distill evaluate + energy done"
stage_log "  architecture_bias=${REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${REPORT_DIR}/energy_efficiency.csv"
stage_log "  detail=${REPORT_DIR}/supplemental_detail.csv"
