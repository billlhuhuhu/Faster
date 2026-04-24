#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_NAME_OR_PATH="${VLM_MODEL_NAME_OR_PATH:-Qwen/Qwen2-VL-2B-Instruct}"
DATASET_NAME="${VLM_DATASET_NAME:-llava_instruct_150k}"
ANNOTATION_PATH="${LLAVA_ANNOTATION_PATH:-}"
IMAGE_ROOT="${LLAVA_IMAGE_ROOT:-}"
VAL_ANNOTATION_PATH="${LLAVA_VAL_ANNOTATION_PATH:-}"
OUTPUT_ROOT="${VLM_FINETUNE_OUTPUT_ROOT:-artifacts/vlm_finetune/qwen2vl_llava_subset}"
SEED="${VLM_SEED:-0}"

RATIOS_STR="${VLM_SUBSET_RATIOS:-1 5 10}"
RUN_FULL="${VLM_RUN_FULL:-1}"
RUN_RANDOM="${VLM_RUN_RANDOM:-1}"
RUN_OURS="${VLM_RUN_OURS:-1}"
OURS_SELECTED_INDICES_TEMPLATE="${VLM_OURS_SELECTED_INDICES_TEMPLATE:-}"

FINETUNE_MODE="${VLM_FINETUNE_MODE:-lora}"
NUM_TRAIN_EPOCHS="${VLM_NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${VLM_LEARNING_RATE:-2e-4}"
LORA_R="${VLM_LORA_R:-16}"
LORA_ALPHA="${VLM_LORA_ALPHA:-32}"
LORA_DROPOUT="${VLM_LORA_DROPOUT:-0.05}"
PER_DEVICE_TRAIN_BATCH_SIZE="${VLM_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${VLM_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${VLM_GRADIENT_ACCUMULATION_STEPS:-16}"
MAX_LENGTH="${VLM_MAX_LENGTH:-2048}"
NUM_WORKERS="${VLM_NUM_WORKERS:-2}"
LOGGING_STEPS="${VLM_LOGGING_STEPS:-10}"
EVAL_STEPS="${VLM_EVAL_STEPS:-100}"
SAVE_STEPS="${VLM_SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${VLM_SAVE_TOTAL_LIMIT:-2}"
VAL_RATIO="${VLM_VAL_RATIO:-0.02}"
REPORT_TO="${VLM_REPORT_TO:-none}"
DEVICE_MAP="${VLM_DEVICE_MAP:-}"
EVAL_CONFIG="${VLM_EVAL_CONFIG:-}"
BF16="${VLM_BF16:-1}"
MERGE_LORA_FOR_EVAL="${VLM_MERGE_LORA_FOR_EVAL:-0}"
EVAL_BENCHMARKS="${VLM_EVAL_BENCHMARKS:-GQA,ScienceQA-IMG,MMBench,TextVQA,POPE}"
VLMEVAL_MODEL_KEY="${VLM_VLMEVAL_MODEL_KEY:-qwen2vl_subset}"
VLMEVAL_MODEL_CLASS="${VLM_VLMEVAL_MODEL_CLASS:-Qwen2VLChat}"
FINETUNE_CUDA_VISIBLE_DEVICES="${VLM_FINETUNE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
FINETUNE_USE_TORCHRUN="${VLM_FINETUNE_USE_TORCHRUN:-auto}"
FINETUNE_NPROC_PER_NODE="${VLM_FINETUNE_NPROC_PER_NODE:-}"

count_cuda_devices() {
  local devices="$1"
  if [[ -z "${devices}" ]]; then
    echo "1"
    return
  fi
  python - <<PY
devices = "${devices}".strip()
items = [item for item in devices.split(",") if item.strip()]
print(max(len(items), 1))
PY
}

FINETUNE_VISIBLE_DEVICE_COUNT="$(count_cuda_devices "${FINETUNE_CUDA_VISIBLE_DEVICES}")"
if [[ -z "${FINETUNE_NPROC_PER_NODE}" ]]; then
  FINETUNE_NPROC_PER_NODE="${FINETUNE_VISIBLE_DEVICE_COUNT}"
fi
if [[ "${FINETUNE_USE_TORCHRUN}" == "auto" ]]; then
  if [[ "${FINETUNE_VISIBLE_DEVICE_COUNT}" -gt 1 ]]; then
    FINETUNE_USE_TORCHRUN="1"
  else
    FINETUNE_USE_TORCHRUN="0"
  fi
fi

REPORT_DIR="${OUTPUT_ROOT}/reports"
mkdir -p "${REPORT_DIR}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RAW_CSV_PATH="${REPORT_DIR}/qwen2vl_llava_subset_finetune_${RUN_TIMESTAMP}_raw.csv"
MISSING_TXT_PATH="${REPORT_DIR}/qwen2vl_llava_subset_finetune_${RUN_TIMESTAMP}_missing.txt"

if [[ -z "${ANNOTATION_PATH}" ]]; then
  echo "LLAVA_ANNOTATION_PATH is required. Example: LLAVA_ANNOTATION_PATH=/path/to/llava_instruct_150k.json" >&2
  exit 1
fi
if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "LLAVA_IMAGE_ROOT is required. Example: LLAVA_IMAGE_ROOT=/path/to/coco/train2014" >&2
  exit 1
fi

format_ratio_tag() {
  local ratio="$1"
  python - <<PY
ratio = float("${ratio}")
print(f"ratio_{int(round(ratio)):02d}")
PY
}

resolve_ours_indices_path() {
  local ratio="$1"
  local ratio_tag="$2"
  local path="${OURS_SELECTED_INDICES_TEMPLATE}"
  path="${path//\{ratio\}/${ratio}}"
  path="${path//\{ratio_tag\}/${ratio_tag}}"
  echo "${path}"
}

run_one() {
  local subset_mode="$1"
  local ratio="$2"
  local selected_indices_path="${3:-}"
  local extra_args=()
  local mode_label="${subset_mode}"

  if [[ "${subset_mode}" != "full" ]]; then
    extra_args+=(--subset_ratio "${ratio}")
    mode_label="${subset_mode}_${ratio}"
  fi
  if [[ "${subset_mode}" == "ours" ]]; then
    extra_args+=(--selected_indices_path "${selected_indices_path}")
  fi
  if [[ -n "${VAL_ANNOTATION_PATH}" ]]; then
    extra_args+=(--val_annotation_path "${VAL_ANNOTATION_PATH}")
  fi
  if [[ -n "${DEVICE_MAP}" ]]; then
    extra_args+=(--device_map "${DEVICE_MAP}")
  fi
  if [[ -n "${EVAL_CONFIG}" ]]; then
    extra_args+=(--eval_config "${EVAL_CONFIG}")
  fi
  if [[ "${BF16}" != "1" ]]; then
    extra_args+=(--no_bf16 --fp16)
  fi
  if [[ "${MERGE_LORA_FOR_EVAL}" == "1" ]]; then
    extra_args+=(--merge_lora_for_eval)
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] VLM finetune start: ${mode_label}"
  if [[ -n "${FINETUNE_CUDA_VISIBLE_DEVICES}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] VLM finetune visible GPU(s): ${FINETUNE_CUDA_VISIBLE_DEVICES}"
  fi
  local launcher=(python)
  if [[ "${FINETUNE_USE_TORCHRUN}" == "1" ]]; then
    launcher=(torchrun --standalone --nproc_per_node "${FINETUNE_NPROC_PER_NODE}")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] VLM finetune launcher: torchrun nproc_per_node=${FINETUNE_NPROC_PER_NODE}"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] VLM finetune launcher: python"
  fi
  CUDA_VISIBLE_DEVICES="${FINETUNE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}" \
  "${launcher[@]}" "${PROJECT_ROOT}/run_vlm_finetune.py" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --annotation_path "${ANNOTATION_PATH}" \
    --image_root "${IMAGE_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --subset_mode "${subset_mode}" \
    --seed "${SEED}" \
    --val_ratio "${VAL_RATIO}" \
    --finetune_mode "${FINETUNE_MODE}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max_length "${MAX_LENGTH}" \
    --num_workers "${NUM_WORKERS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --report_to "${REPORT_TO}" \
    --eval_benchmarks "${EVAL_BENCHMARKS}" \
    --eval_backend "vlmevalkit" \
    --vlmeval_model_key "${VLMEVAL_MODEL_KEY}" \
    --vlmeval_model_class "${VLMEVAL_MODEL_CLASS}" \
    "${extra_args[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] VLM finetune done: ${mode_label}"
}

if [[ "${RUN_FULL}" == "1" ]]; then
  run_one "full" "" ""
fi

read -r -a RATIOS <<< "${RATIOS_STR}"
for ratio in "${RATIOS[@]}"; do
  ratio_tag="$(format_ratio_tag "${ratio}")"
  if [[ "${RUN_RANDOM}" == "1" ]]; then
    run_one "random" "${ratio}" ""
  fi
  if [[ "${RUN_OURS}" == "1" ]]; then
    if [[ -z "${OURS_SELECTED_INDICES_TEMPLATE}" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip ours_${ratio}: VLM_OURS_SELECTED_INDICES_TEMPLATE is empty."
      echo "ours_${ratio}: missing VLM_OURS_SELECTED_INDICES_TEMPLATE" >> "${MISSING_TXT_PATH}"
      continue
    fi
    ours_indices_path="$(resolve_ours_indices_path "${ratio}" "${ratio_tag}")"
    if [[ ! -f "${ours_indices_path}" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip ours_${ratio}: selected indices not found at ${ours_indices_path}"
      echo "ours_${ratio}: ${ours_indices_path}" >> "${MISSING_TXT_PATH}"
      continue
    fi
    run_one "ours" "${ratio}" "${ours_indices_path}"
  fi
done

python - "${OUTPUT_ROOT}" "${MODEL_NAME_OR_PATH}" "${DATASET_NAME}" "${RAW_CSV_PATH}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
model_name = sys.argv[2].replace("\\", "-").replace("/", "-").replace(" ", "_")
dataset_name = sys.argv[3].replace("\\", "-").replace("/", "-").replace(" ", "_")
raw_csv_path = Path(sys.argv[4])
base = output_root / model_name / dataset_name

rows = []
for metrics_path in sorted(base.glob("*/seed_*/metrics.json")):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    subset_info_path = metrics_path.parent / "subset_info.json"
    subset_info = {}
    if subset_info_path.exists():
        subset_info = json.loads(subset_info_path.read_text(encoding="utf-8"))
    rows.append({
        "subset_tag": metrics_path.parent.parent.name,
        "seed": metrics_path.parent.name.replace("seed_", ""),
        "subset_mode": payload.get("subset_mode", subset_info.get("subset_mode", "")),
        "subset_ratio": payload.get("subset_ratio", subset_info.get("subset_ratio", "")),
        "num_selected_records": payload.get("num_selected_records", subset_info.get("num_selected_records", "")),
        "train_loss": payload.get("train_loss", ""),
        "final_eval_loss": payload.get("final_eval_loss", ""),
        "train_runtime": payload.get("train_runtime", ""),
        "elapsed_seconds": payload.get("elapsed_seconds", ""),
        "best_model_checkpoint": payload.get("best_model_checkpoint", ""),
        "metrics_path": str(metrics_path),
    })

raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "subset_tag", "seed", "subset_mode", "subset_ratio", "num_selected_records",
    "train_loss", "final_eval_loss", "train_runtime", "elapsed_seconds",
    "best_model_checkpoint", "metrics_path",
]
with raw_csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"saved VLM finetune summary: {raw_csv_path}")
print(f"collected runs: {len(rows)}")
PY

echo "VLM finetune report: ${RAW_CSV_PATH}"
