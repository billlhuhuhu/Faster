#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${RANDOM_BASELINE_DATASET:-flickr}"
BACKBONE="${RANDOM_BASELINE_BACKBONE:-nfnet}"
TEXT_ENCODER="${RANDOM_BASELINE_TEXT_ENCODER:-bert}"
VARIANT="${RANDOM_BASELINE_VARIANT:-random_baseline}"
SEEDS_STR="${RANDOM_BASELINE_SEEDS:-0}"
read -r -a SEEDS <<< "${SEEDS_STR}"
BUDGETS_STR="${RANDOM_BASELINE_BUDGETS:-100 200 500}"
read -r -a BUDGETS <<< "${BUDGETS_STR}"
RATIOS_STR="${RANDOM_BASELINE_RATIOS:-0.01}"
read -r -a RATIOS <<< "${RATIOS_STR}"

SELECTION_OUTPUT_ROOT="${RANDOM_BASELINE_SELECTION_OUTPUT_ROOT:-artifacts/subset_selection_random_baseline}"
TRAIN_OUTPUT_ROOT="${RANDOM_BASELINE_TRAIN_OUTPUT_ROOT:-artifacts/subset_train_random_baseline}"
REPORT_NAME="${RANDOM_BASELINE_REPORT_NAME:-random_baseline_combo}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/${REPORT_NAME}_${DATASET}_${RUN_TIMESTAMP}"
REPORT_DIR="${REPORT_ROOT}/${REPORT_NAME}_${DATASET}_${RUN_TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"
mkdir -p "${REPORT_DIR}"

MODEL_TAG="$(sanitize_component "${BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
IMAGE_ROOT="$(get_image_root "${DATASET}")"
FEATURE_CACHE_DIR="${FEATURE_CACHE_ROOT}/${DATASET}/train/${MODEL_TAG}"

format_ratio_tag_local() {
  local ratio="$1"
  python - <<PY
ratio = float("${ratio}")
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
}

ensure_feature_cache() {
  if [[ -f "${FEATURE_CACHE_DIR}/img_features_selection.pt" && -f "${FEATURE_CACHE_DIR}/txt_features_selection.pt" && -f "${FEATURE_CACHE_DIR}/sample_meta.json" ]]; then
    stage_log "Skip feature cache: existing cache found at ${FEATURE_CACHE_DIR}"
    return 0
  fi

  stage_log "Feature cache start for random baseline: dataset=${DATASET} backbone=${BACKBONE}"
  python "${PROJECT_ROOT}/run_feature_cache.py" \
    --dataset "${DATASET}" \
    --image_encoder "${BACKBONE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --selection_image_repr_method "${SELECTION_IMAGE_REPR_METHOD}" \
    --selection_text_repr_method "${SELECTION_TEXT_REPR_METHOD}" \
    --image_root "${IMAGE_ROOT}" \
    --ann_root "${ANN_ROOT}" \
    --cache_root "${FEATURE_CACHE_ROOT}" \
    --selection_image_size "${SELECTION_IMAGE_SIZE}" \
    --selection_raw_resize_size "${SELECTION_RAW_RESIZE_SIZE}" \
    --selection_raw_pca_dim "${SELECTION_RAW_PCA_DIM}" \
    --selection_image_batch_size "${SELECTION_IMAGE_BATCH_SIZE}" \
    --selection_text_batch_size "${SELECTION_TEXT_BATCH_SIZE}" \
    --hog_orientations "${HOG_ORIENTATIONS}" \
    --hog_pixels_per_cell "${HOG_PIXELS_PER_CELL}" \
    --hog_cells_per_block "${HOG_CELLS_PER_BLOCK}" \
    --color_hist_bins "${COLOR_HIST_BINS}" \
    --color_space "${COLOR_SPACE}" \
    --device "${DEVICE}" \
    > "${RUN_LOG_DIR}/feature_cache.log" 2>&1
  stage_log "Feature cache done"
}

run_random_selection_abs() {
  local budget="$1"
  local seed="$2"
  local budget_tag
  local selected_indices_path
  local selection_log
  local train_log
  local metrics_path
  local train_extra_args=()

  budget_tag="$(format_budget_tag "${budget}")"
  selected_indices_path="${SELECTION_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${budget_tag}/random/seed_${seed}/selected_indices.json"
  metrics_path="${TRAIN_OUTPUT_ROOT}/${DATASET}/${MODEL_TAG}/${budget_tag}/${VARIANT}/seed_${seed}/metrics.json"
  selection_log="${RUN_LOG_DIR}/${budget_tag}_seed${seed}_select.log"
  train_log="${RUN_LOG_DIR}/${budget_tag}_seed${seed}_train.log"

  if [[ ! -f "${selected_indices_path}" ]]; then
    stage_log "Random selection start: budget=${budget} seed=${seed}"
    python "${PROJECT_ROOT}/run_random_subset_selection.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --feature_cache_root "${FEATURE_CACHE_ROOT}" \
      --output_root "${SELECTION_OUTPUT_ROOT}" \
      --budget_size "${budget}" \
      --selection_method random \
      --random_state "${seed}" \
      > "${selection_log}" 2>&1
    stage_log "Random selection done: budget=${budget} seed=${seed}"
  else
    stage_log "Skip random selection: existing selected_indices found at ${selected_indices_path}"
  fi

  if [[ ! -f "${metrics_path}" ]]; then
    if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
      train_extra_args+=(--no_aug)
    fi
    stage_log "Train start: budget=${budget} seed=${seed}"
    python "${PROJECT_ROOT}/run_subset_train.py" \
      --dataset "${DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --selected_indices_path "${selected_indices_path}" \
      --subset_size "${budget}" \
      --subset_tag "${VARIANT}" \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --output_root "${TRAIN_OUTPUT_ROOT}" \
      --batch_size_train "${BATCH_TRAIN}" \
      --batch_size_test "${BATCH_TEST}" \
      --text_batch_size "${TEXT_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      "${train_extra_args[@]}" \
      > "${train_log}" 2>&1
    stage_log "Train done: budget=${budget} seed=${seed}"
  else
    stage_log "Skip train: existing metrics found at ${metrics_path}"
  fi
}

run_random_selection_ratio() {
  local ratio="$1"
  local seed="$2"
  local ratio_tag
  local selected_indices_path
  local selection_log
  local train_log
  local metrics_path
  local train_extra_args=()

  ratio_tag="$(format_ratio_tag_local "${ratio}")"
  selected_indices_path="${SELECTION_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${ratio_tag}/random/seed_${seed}/selected_indices.json"
  metrics_path="${TRAIN_OUTPUT_ROOT}/${DATASET}/${MODEL_TAG}/${ratio_tag}/${VARIANT}/seed_${seed}/metrics.json"
  selection_log="${RUN_LOG_DIR}/${ratio_tag}_seed${seed}_select.log"
  train_log="${RUN_LOG_DIR}/${ratio_tag}_seed${seed}_train.log"

  if [[ ! -f "${selected_indices_path}" ]]; then
    stage_log "Random selection start: ratio=${ratio} seed=${seed}"
    python "${PROJECT_ROOT}/run_random_subset_selection.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --feature_cache_root "${FEATURE_CACHE_ROOT}" \
      --output_root "${SELECTION_OUTPUT_ROOT}" \
      --budget_ratio "${ratio}" \
      --selection_method random \
      --random_state "${seed}" \
      > "${selection_log}" 2>&1
    stage_log "Random selection done: ratio=${ratio} seed=${seed}"
  else
    stage_log "Skip random selection: existing selected_indices found at ${selected_indices_path}"
  fi

  if [[ ! -f "${metrics_path}" ]]; then
    if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
      train_extra_args+=(--no_aug)
    fi
    stage_log "Train start: ratio=${ratio} seed=${seed}"
    python "${PROJECT_ROOT}/run_subset_train.py" \
      --dataset "${DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --selected_indices_path "${selected_indices_path}" \
      --subset_ratio "${ratio}" \
      --subset_tag "${VARIANT}" \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --output_root "${TRAIN_OUTPUT_ROOT}" \
      --batch_size_train "${BATCH_TRAIN}" \
      --batch_size_test "${BATCH_TEST}" \
      --text_batch_size "${TEXT_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      "${train_extra_args[@]}" \
      > "${train_log}" 2>&1
    stage_log "Train done: ratio=${ratio} seed=${seed}"
  else
    stage_log "Skip train: existing metrics found at ${metrics_path}"
  fi
}

cd "${PROJECT_ROOT}"
ensure_feature_cache

stage_log "Random baseline combo start: dataset=${DATASET} budgets=${BUDGETS[*]} ratios=${RATIOS[*]} seeds=${SEEDS[*]}"

for budget in "${BUDGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_random_selection_abs "${budget}" "${seed}"
  done
done

for ratio in "${RATIOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_random_selection_ratio "${ratio}" "${seed}"
  done
done

RAW_CSV_PATH="${REPORT_DIR}/random_baseline_raw.csv"
SUMMARY_CSV_PATH="${REPORT_DIR}/random_baseline_summary.csv"
MISSING_TXT_PATH="${REPORT_DIR}/missing_metrics.txt"

python - "${TRAIN_OUTPUT_ROOT}" "${DATASET}" "${MODEL_TAG}" "${VARIANT}" "${RAW_CSV_PATH}" "${SUMMARY_CSV_PATH}" "${MISSING_TXT_PATH}" "${BUDGETS[*]}" "${RATIOS[*]}" "${SEEDS[*]}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


subset_train_root = Path(sys.argv[1])
dataset = sys.argv[2]
model_tag = sys.argv[3]
variant = sys.argv[4]
raw_csv_path = Path(sys.argv[5])
summary_csv_path = Path(sys.argv[6])
missing_txt_path = Path(sys.argv[7])
budgets = [item for item in sys.argv[8].split() if item.strip()]
ratios = [item for item in sys.argv[9].split() if item.strip()]
seeds = [item for item in sys.argv[10].split() if item.strip()]

raw_rows = []
missing = []

targets = []
for budget in budgets:
    targets.append(("abs", f"size_{int(budget):04d}", str(int(budget))))
for ratio in ratios:
    ratio_value = float(ratio)
    targets.append(("ratio", f"ratio_{int(round(ratio_value * 100)):02d}", f"{ratio_value:.6f}"))

for budget_type, budget_tag, budget_value in targets:
    for seed in seeds:
        metrics_path = subset_train_root / dataset / model_tag / budget_tag / variant / f"seed_{int(seed)}" / "metrics.json"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        raw_rows.append(
            {
                "dataset": dataset,
                "model_tag": model_tag,
                "variant": variant,
                "budget_type": budget_type,
                "budget_tag": budget_tag,
                "budget_value": budget_value,
                "seed": int(seed),
                "i2t_r1": float(payload["i2t_r1"]),
                "i2t_r5": float(payload["i2t_r5"]),
                "i2t_r10": float(payload["i2t_r10"]),
                "t2i_r1": float(payload["t2i_r1"]),
                "t2i_r5": float(payload["t2i_r5"]),
                "t2i_r10": float(payload["t2i_r10"]),
                "mean_recall": float(payload["mean_recall"]),
                "metrics_path": str(metrics_path),
            }
        )

raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
raw_fields = [
    "dataset",
    "model_tag",
    "variant",
    "budget_type",
    "budget_tag",
    "budget_value",
    "seed",
    "i2t_r1",
    "i2t_r5",
    "i2t_r10",
    "t2i_r1",
    "t2i_r5",
    "t2i_r10",
    "mean_recall",
    "metrics_path",
]
with raw_csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=raw_fields)
    writer.writeheader()
    writer.writerows(raw_rows)

grouped = {}
for row in raw_rows:
    key = (row["dataset"], row["model_tag"], row["variant"], row["budget_type"], row["budget_tag"], row["budget_value"])
    grouped.setdefault(key, []).append(row)

summary_rows = []
for key in sorted(grouped.keys()):
    dataset, model_tag, variant, budget_type, budget_tag, budget_value = key
    rows = grouped[key]
    summary = {
        "dataset": dataset,
        "model_tag": model_tag,
        "variant": variant,
        "budget_type": budget_type,
        "budget_tag": budget_tag,
        "budget_value": budget_value,
        "num_runs": len(rows),
    }
    for metric in ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall"]:
        values = [float(item[metric]) for item in rows]
        summary[f"{metric}_mean"] = float(sum(values) / len(values))
        summary[f"{metric}_std"] = safe_std(values)
    summary_rows.append(summary)

summary_fields = [
    "dataset",
    "model_tag",
    "variant",
    "budget_type",
    "budget_tag",
    "budget_value",
    "num_runs",
    "i2t_r1_mean",
    "i2t_r1_std",
    "i2t_r5_mean",
    "i2t_r5_std",
    "i2t_r10_mean",
    "i2t_r10_std",
    "t2i_r1_mean",
    "t2i_r1_std",
    "t2i_r5_mean",
    "t2i_r5_std",
    "t2i_r10_mean",
    "t2i_r10_std",
    "mean_recall_mean",
    "mean_recall_std",
]
with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

with missing_txt_path.open("w", encoding="utf-8") as handle:
    for item in missing:
        handle.write(item + "\n")

print(f"saved raw csv: {raw_csv_path}")
print(f"saved summary csv: {summary_csv_path}")
print(f"saved missing list: {missing_txt_path}")
print(f"collected runs: {len(raw_rows)}")
print(f"grouped entries: {len(summary_rows)}")
PY

stage_log "Random baseline combo completed. Logs saved to ${RUN_LOG_DIR}"
stage_log "Random baseline report dir: ${REPORT_DIR}"
stage_log "Random baseline raw csv: ${RAW_CSV_PATH}"
stage_log "Random baseline summary csv: ${SUMMARY_CSV_PATH}"
