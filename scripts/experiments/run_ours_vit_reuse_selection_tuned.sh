#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${VIT_TUNE_DATASET:-flickr}"
SOURCE_BACKBONE="${VIT_TUNE_SOURCE_BACKBONE:-nfnet}"
TEXT_ENCODER="${VIT_TUNE_TEXT_ENCODER:-bert}"
RATIO="${VIT_TUNE_RATIO:-0.03}"
SEED="${VIT_TUNE_SEED:-0}"
IMAGE_ROOT="$(get_image_root "${DATASET}")"
SOURCE_MODEL_TAG="$(sanitize_component "${SOURCE_BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
RATIO_TAG="$(python - "${RATIO}" <<'PY'
import sys
print(f"ratio_{int(round(float(sys.argv[1]) * 100)):02d}")
PY
)"

OUTPUT_ROOT="${VIT_TUNE_OUTPUT_ROOT:-artifacts/vit_tuned_ours_reuse_selection}"
REPORT_DIR="${VIT_TUNE_REPORT_DIR:-${OUTPUT_ROOT}/reports}"
LOG_DIR="${VIT_TUNE_LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

SELECTED_INDICES_PATH="${VIT_TUNE_SELECTED_INDICES:-}"
if [[ -z "${SELECTED_INDICES_PATH}" ]]; then
  SELECTED_INDICES_PATH="$(python - "${DATASET}" "${SOURCE_MODEL_TAG}" "${RATIO_TAG}" "${SEED}" <<'PY'
import sys
from pathlib import Path

dataset, model_tag, ratio_tag, seed = sys.argv[1:5]
patterns = [
    f"artifacts/arch_bias_energy_3pct/ours/subset_selection_dense_sift_bovw/*/{dataset}/train/{model_tag}/{ratio_tag}/proxy_opt_lsrc/seed_{seed}/selected_indices.json",
    f"artifacts/subset_selection_dense_sift_bovw/{dataset}/train/{model_tag}/{ratio_tag}/proxy_opt_lsrc/seed_{seed}/selected_indices.json",
    f"artifacts_coco/subset_selection_dense_sift_bovw_coco/{dataset}/train/{model_tag}/{ratio_tag}/proxy_opt_lsrc/seed_{seed}/selected_indices.json",
]
candidates = []
for pattern in patterns:
    candidates.extend(Path(".").glob(pattern))
candidates = [p for p in candidates if p.is_file()]
if not candidates:
    raise SystemExit(
        "No selected_indices.json found. Set VIT_TUNE_SELECTED_INDICES=/path/to/selected_indices.json"
    )
candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(candidates[0])
PY
)"
fi

if [[ ! -f "${SELECTED_INDICES_PATH}" ]]; then
  echo "Selected indices not found: ${SELECTED_INDICES_PATH}" >&2
  exit 1
fi

VIT_TUNE_CONFIGS="${VIT_TUNE_CONFIGS:-projection_only low_lr_finetune}"
VIT_TUNE_EPOCHS="${VIT_TUNE_EPOCHS:-300}"
VIT_TUNE_BATCH_TRAIN="${VIT_TUNE_BATCH_TRAIN:-32}"
VIT_TUNE_BATCH_TEST="${VIT_TUNE_BATCH_TEST:-64}"
VIT_TUNE_TEXT_BATCH_SIZE="${VIT_TUNE_TEXT_BATCH_SIZE:-512}"
VIT_TUNE_NUM_WORKERS="${VIT_TUNE_NUM_WORKERS:-4}"
VIT_TUNE_EVAL_INTERVAL="${VIT_TUNE_EVAL_INTERVAL:-5}"
VIT_TUNE_WEIGHT_DECAY="${VIT_TUNE_WEIGHT_DECAY:-1e-4}"
VIT_TUNE_LR_DECAY_GAMMA="${VIT_TUNE_LR_DECAY_GAMMA:-0.2}"
VIT_TUNE_NO_AUG="${VIT_TUNE_NO_AUG:-0}"
VIT_TUNE_FORCE="${VIT_TUNE_FORCE:-0}"
VIT_TUNE_ENABLE_DP="${VIT_TUNE_ENABLE_DP:-${ENABLE_IMAGE_ENCODER_DATA_PARALLEL:-0}}"
VIT_TUNE_DP_DEVICE_IDS="${VIT_TUNE_DP_DEVICE_IDS:-${IMAGE_ENCODER_DATA_PARALLEL_DEVICE_IDS:-${CUDA_VISIBLE_DEVICES:-}}}"

stage_log "Ours ViT tuned retraining start"
stage_log "  dataset=${DATASET} source=${SOURCE_MODEL_TAG} ratio=${RATIO} seed=${SEED}"
stage_log "  selected=${SELECTED_INDICES_PATH}"
stage_log "  configs=${VIT_TUNE_CONFIGS}"
stage_log "  output=${OUTPUT_ROOT}"

run_one_config() {
  local config="$1"
  local image_trainable="true"
  local lr_img="0.001"
  local lr_txt="0.05"
  local subset_tag="proxy_opt_lsrc_${config}"

  case "${config}" in
    projection_only)
      image_trainable="false"
      lr_img="0.0"
      lr_txt="${VIT_TUNE_PROJ_LR_TXT:-0.05}"
      ;;
    low_lr_finetune)
      image_trainable="true"
      lr_img="${VIT_TUNE_LOWLR_IMG:-0.001}"
      lr_txt="${VIT_TUNE_LOWLR_TXT:-0.05}"
      ;;
    very_low_lr_finetune)
      image_trainable="true"
      lr_img="${VIT_TUNE_VERY_LOWLR_IMG:-0.0003}"
      lr_txt="${VIT_TUNE_VERY_LOWLR_TXT:-0.03}"
      ;;
    *)
      echo "Unknown VIT_TUNE_CONFIG: ${config}" >&2
      exit 1
      ;;
  esac

  local train_root="${OUTPUT_ROOT}/${config}"
  local metrics_path="${train_root}/${DATASET}/vit_b16_${TEXT_ENCODER}/${RATIO_TAG}/${subset_tag}/seed_${SEED}/metrics.json"
  local log_path="${LOG_DIR}/${config}_${RATIO_TAG}_seed${SEED}.log"
  local extra_args=()

  if [[ "${VIT_TUNE_NO_AUG}" == "1" ]]; then
    extra_args+=(--no_aug)
  fi
  if [[ "${VIT_TUNE_ENABLE_DP}" == "1" ]]; then
    extra_args+=(--enable_image_encoder_data_parallel)
    if [[ -n "${VIT_TUNE_DP_DEVICE_IDS}" ]]; then
      extra_args+=(--image_encoder_data_parallel_device_ids "${VIT_TUNE_DP_DEVICE_IDS}")
    fi
  fi

  if [[ "${VIT_TUNE_FORCE}" != "1" && -f "${metrics_path}" ]]; then
    stage_log "Skip ${config}: existing metrics found at ${metrics_path}"
    return 0
  fi

  stage_log "Train ViT config=${config} image_trainable=${image_trainable} lr_img=${lr_img} lr_txt=${lr_txt}"
  python "${PROJECT_ROOT}/run_subset_train.py" \
    --dataset "${DATASET}" \
    --image_root "${IMAGE_ROOT}" \
    --ann_root "${ANN_ROOT}" \
    --selected_indices_path "${SELECTED_INDICES_PATH}" \
    --subset_ratio "${RATIO}" \
    --subset_tag "${subset_tag}" \
    --image_encoder vit_b16 \
    --text_encoder "${TEXT_ENCODER}" \
    --output_root "${train_root}" \
    --batch_size_train "${VIT_TUNE_BATCH_TRAIN}" \
    --batch_size_test "${VIT_TUNE_BATCH_TEST}" \
    --text_batch_size "${VIT_TUNE_TEXT_BATCH_SIZE}" \
    --num_workers "${VIT_TUNE_NUM_WORKERS}" \
    --epochs "${VIT_TUNE_EPOCHS}" \
    --eval_interval "${VIT_TUNE_EVAL_INTERVAL}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --lr_teacher_img "${lr_img}" \
    --lr_teacher_txt "${lr_txt}" \
    --weight_decay "${VIT_TUNE_WEIGHT_DECAY}" \
    --lr_decay_gamma "${VIT_TUNE_LR_DECAY_GAMMA}" \
    --image_trainable "${image_trainable}" \
    --text_trainable false \
    "${extra_args[@]}" \
    2>&1 | tee "${log_path}"
}

for config in ${VIT_TUNE_CONFIGS}; do
  run_one_config "${config}"
done

python - "${OUTPUT_ROOT}" "${REPORT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report_dir = Path(sys.argv[2])
report_dir.mkdir(parents=True, exist_ok=True)
rows = []
for metrics_path in sorted(root.glob("*/**/metrics.json")):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metrics_path.relative_to(root).parts[0]
    row = {
        "config": config,
        "dataset": payload.get("dataset"),
        "backbone": payload.get("backbone"),
        "subset_ratio": payload.get("subset_ratio"),
        "subset_size": payload.get("subset_size"),
        "seed": payload.get("seed"),
        "best_epoch": payload.get("best_epoch"),
        "i2t_r1": payload.get("i2t_r1"),
        "i2t_r5": payload.get("i2t_r5"),
        "i2t_r10": payload.get("i2t_r10"),
        "t2i_r1": payload.get("t2i_r1"),
        "t2i_r5": payload.get("t2i_r5"),
        "t2i_r10": payload.get("t2i_r10"),
        "mean_recall": payload.get("mean_recall"),
        "metrics_path": str(metrics_path),
    }
    rows.append(row)
rows.sort(key=lambda r: float(r.get("mean_recall") or -1), reverse=True)
out = report_dir / "ours_vit_tuned_results.csv"
fields = [
    "config", "dataset", "backbone", "subset_ratio", "subset_size", "seed", "best_epoch",
    "i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall",
    "metrics_path",
]
with out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"saved report: {out}")
if rows:
    best = rows[0]
    print(f"best config={best['config']} mean_recall={best['mean_recall']} metrics={best['metrics_path']}")
PY

stage_log "Ours ViT tuned retraining done"
stage_log "  report=${REPORT_DIR}/ours_vit_tuned_results.csv"
