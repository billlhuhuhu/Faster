#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

# Default: the 4 newly added multimodal baselines (override if needed).
BASELINE_METHODS="${BASELINE_METHODS:-presel visa dataprophet dynamic_pruning}"
ABS_BUDGETS="${ABS_BUDGETS:-100 200 500}"
RATIOS="${RATIOS:-0.01 0.02 0.03}"
DATAPROPHET_MASTER_RATIO="${DATAPROPHET_MASTER_RATIO:-0.05}"
BASELINE_SEEDS="${BASELINE_SEEDS:-0}"
BASELINE_DEVICE="${BASELINE_DEVICE:-cuda:0}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-artifacts/baselines_coco}"

BASELINE_DATASET="${BASELINE_DATASET:-coco}"
BASELINE_IMAGE_ENCODER="${BASELINE_IMAGE_ENCODER:-nfnet}"
BASELINE_TEXT_ENCODER="${BASELINE_TEXT_ENCODER:-bert}"
BASELINE_FEATURE_SOURCE="${BASELINE_FEATURE_SOURCE:-artifacts/feature_cache}"
BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-data/coco}"
BASELINE_ANN_ROOT="${BASELINE_ANN_ROOT:-data/COCO}"

# Align with main experiment defaults.
BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
BASELINE_BATCH_TRAIN="${BASELINE_BATCH_TRAIN:-64}"
BASELINE_BATCH_TEST="${BASELINE_BATCH_TEST:-128}"
BASELINE_TEXT_BATCH="${BASELINE_TEXT_BATCH:-1024}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-4}"
BASELINE_CONFIG="${BASELINE_CONFIG:-baselines/configs/main_aligned_flickr_nfnet_bert.yaml}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

echo "[coco-formal] dataset=${BASELINE_DATASET}"
echo "[coco-formal] methods=${BASELINE_METHODS}"
echo "[coco-formal] abs_budgets=${ABS_BUDGETS}"
echo "[coco-formal] ratios=${RATIOS:-<disabled>}"
echo "[coco-formal] dataprophet_master_ratio=${DATAPROPHET_MASTER_RATIO}"
echo "[coco-formal] seeds=${BASELINE_SEEDS}"
echo "[coco-formal] output_root=${BASELINE_OUTPUT_ROOT}"
echo "[coco-formal] device=${BASELINE_DEVICE}"
echo "[coco-formal] image_root=${BASELINE_IMAGE_ROOT}"
echo "[coco-formal] ann_root=${BASELINE_ANN_ROOT}"

max_abs_budget() {
  local max_b=0
  local b
  for b in ${ABS_BUDGETS}; do
    if [[ "${b}" -gt "${max_b}" ]]; then
      max_b="${b}"
    fi
  done
  echo "${max_b}"
}

ratio_to_tag() {
  local ratio="$1"
  python - <<PY
r = float("${ratio}")
print(f"ratio_{int(round(r*100)):02d}")
PY
}

ensure_dataprophet_master_selection() {
  local seed="$1"
  run_ratio_selection_only "dataprophet" "${DATAPROPHET_MASTER_RATIO}" "${seed}"
}

materialize_dataprophet_budget_from_master_ratio() {
  local seed="$1"
  local target_budget="$2"
  local master_ratio="$3"

  local method="dataprophet"
  local master_tag
  master_tag="$(ratio_to_tag "${master_ratio}")"
  local model_tag="${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}"
  local src_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${master_tag}/${method}/seed_${seed}"
  local dst_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}/${method}/budget_$(printf "%04d" "${target_budget}")/seed_${seed}"
  local src_selected="${src_dir}/selected_indices.json"
  local src_summary="${src_dir}/baseline_summary.json"
  local src_scores="${src_dir}/selection_scores.npz"
  local dst_selected="${dst_dir}/selected_indices.json"
  local dst_summary="${dst_dir}/baseline_summary.json"
  local dst_scores="${dst_dir}/selection_scores.npz"

  if [[ -f "${dst_selected}" ]]; then
    echo "[dataprophet-reuse][skip] selection exists budget=${target_budget} seed=${seed}"
    return 0
  fi
  if [[ ! -f "${src_selected}" ]]; then
    echo "[dataprophet-reuse][warn] source selection missing, cannot derive budget=${target_budget} from ratio=${master_ratio}"
    return 1
  fi

  mkdir -p "${dst_dir}"
  python - <<PY
import json
import os
import shutil

src_selected = "${src_selected}"
src_summary = "${src_summary}"
src_scores = "${src_scores}"
dst_selected = "${dst_selected}"
dst_summary = "${dst_summary}"
dst_scores = "${dst_scores}"
target_budget = int("${target_budget}")
master_ratio = float("${master_ratio}")

with open(src_selected, "r", encoding="utf-8") as f:
    payload = json.load(f)
src_idx = [int(x) for x in payload.get("selected_indices", [])]
selected = src_idx[:target_budget]
with open(dst_selected, "w", encoding="utf-8") as f:
    json.dump({"selected_indices": selected}, f, ensure_ascii=False, indent=2)

summary = {}
if os.path.exists(src_summary):
    with open(src_summary, "r", encoding="utf-8") as f:
        summary = json.load(f)
total = int(summary.get("total_train_size", max(len(selected), 1)))
summary["budget"] = int(target_budget)
summary["subset_size"] = int(len(selected))
summary["ratio"] = float(len(selected)) / max(float(total), 1.0)
summary["derived_from_ratio"] = float(master_ratio)
summary["derived_from"] = src_selected
with open(dst_summary, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

if os.path.exists(src_scores):
    shutil.copy2(src_scores, dst_scores)
PY
  echo "[dataprophet-reuse] derived budget=${target_budget} from ratio=${master_ratio} seed=${seed}"
}

materialize_dataprophet_ratio_from_master_ratio() {
  local seed="$1"
  local target_ratio="$2"
  local master_ratio="$3"

  local method="dataprophet"
  local target_tag
  local master_tag
  target_tag="$(ratio_to_tag "${target_ratio}")"
  master_tag="$(ratio_to_tag "${master_ratio}")"

  if [[ "${target_tag}" == "${master_tag}" ]]; then
    return 0
  fi

  local model_tag="${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}"
  local src_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${master_tag}/${method}/seed_${seed}"
  local dst_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${target_tag}/${method}/seed_${seed}"
  local src_selected="${src_dir}/selected_indices.json"
  local src_summary="${src_dir}/baseline_summary.json"
  local src_scores="${src_dir}/selection_scores.npz"
  local dst_selected="${dst_dir}/selected_indices.json"
  local dst_summary="${dst_dir}/baseline_summary.json"
  local dst_scores="${dst_dir}/selection_scores.npz"

  if [[ -f "${dst_selected}" ]]; then
    echo "[dataprophet-reuse][skip] ratio selection exists ratio=${target_ratio} seed=${seed}"
    return 0
  fi
  if [[ ! -f "${src_selected}" ]]; then
    echo "[dataprophet-reuse][warn] source ratio selection missing, cannot derive ratio=${target_ratio} from ratio=${master_ratio}"
    return 1
  fi

  mkdir -p "${dst_dir}"
  python - <<PY
import json
import math
import os
import shutil

src_selected = "${src_selected}"
src_summary = "${src_summary}"
src_scores = "${src_scores}"
dst_selected = "${dst_selected}"
dst_summary = "${dst_summary}"
dst_scores = "${dst_scores}"
target_ratio = float("${target_ratio}")
master_ratio = float("${master_ratio}")

with open(src_selected, "r", encoding="utf-8") as f:
    payload = json.load(f)
src_idx = [int(x) for x in payload.get("selected_indices", [])]

summary = {}
if os.path.exists(src_summary):
    with open(src_summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

total = int(summary.get("total_train_size", max(len(src_idx), 1)))
target_k = int(round(target_ratio * total))
target_k = max(1, min(target_k, len(src_idx)))
selected = src_idx[:target_k]

with open(dst_selected, "w", encoding="utf-8") as f:
    json.dump({"selected_indices": selected}, f, ensure_ascii=False, indent=2)

summary["ratio"] = float(len(selected)) / max(float(total), 1.0)
summary["budget"] = int(len(selected))
summary["subset_size"] = int(len(selected))
summary["derived_from_ratio"] = float(master_ratio)
summary["derived_from"] = src_selected
with open(dst_summary, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

if os.path.exists(src_scores):
    shutil.copy2(src_scores, dst_scores)
PY
  echo "[dataprophet-reuse] derived ratio=${target_ratio} from ratio=${master_ratio} seed=${seed}"
}

run_abs_job() {
  local method="$1"
  local budget="$2"
  local seed="$3"

  local run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}/${method}/budget_$(printf "%04d" "${budget}")/seed_${seed}"
  local selected_path="${run_dir}/selected_indices.json"
  local metrics_path="${run_dir}/downstream_metrics.json"

  if [[ -f "${metrics_path}" ]]; then
    echo "[skip][abs] done method=${method} budget=${budget} seed=${seed}"
    return 0
  fi

  if [[ ! -f "${selected_path}" ]]; then
    echo "[run][abs][selection] method=${method} budget=${budget} seed=${seed}"
    python -m baselines.runners.run_baseline_selection \
      --method "${method}" \
      --budget "${budget}" \
      --dataset_name "${BASELINE_DATASET}" \
      --split train \
      --image_encoder "${BASELINE_IMAGE_ENCODER}" \
      --text_encoder "${BASELINE_TEXT_ENCODER}" \
      --feature_source "${BASELINE_FEATURE_SOURCE}" \
      --output_dir "${BASELINE_OUTPUT_ROOT}" \
      --config "${BASELINE_CONFIG}" \
      --output_layout budget \
      --seed "${seed}" \
      --device "${BASELINE_DEVICE}"
  else
    echo "[skip][abs][selection] exists method=${method} budget=${budget} seed=${seed}"
  fi

  echo "[run][abs][eval] method=${method} budget=${budget} seed=${seed}"
  python -m baselines.runners.evaluate_baseline_subsets \
    --baseline_result_dir "${run_dir}" \
    --dataset_name "${BASELINE_DATASET}" \
    --image_encoder "${BASELINE_IMAGE_ENCODER}" \
    --text_encoder "${BASELINE_TEXT_ENCODER}" \
    --feature_source "${BASELINE_FEATURE_SOURCE}" \
    --image_root "${BASELINE_IMAGE_ROOT}" \
    --ann_root "${BASELINE_ANN_ROOT}" \
    --device "${BASELINE_DEVICE}" \
    --epochs "${BASELINE_EPOCHS}" \
    --batch_size_train "${BASELINE_BATCH_TRAIN}" \
    --batch_size_test "${BASELINE_BATCH_TEST}" \
    --text_batch_size "${BASELINE_TEXT_BATCH}" \
    --num_workers "${BASELINE_NUM_WORKERS}" \
    --eval_interval 1 \
    --no_aug
}

run_ratio_job() {
  local method="$1"
  local ratio="$2"
  local seed="$3"

  local ratio_tag
  ratio_tag="$(python - <<PY
r = float("${ratio}")
print(f"ratio_{int(round(r*100)):02d}")
PY
)"
  local model_tag="${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}"
  local run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${ratio_tag}/${method}/seed_${seed}"
  local selected_path="${run_dir}/selected_indices.json"
  local metrics_path="${run_dir}/downstream_metrics.json"

  if [[ -f "${metrics_path}" ]]; then
    echo "[skip][ratio] done method=${method} ratio=${ratio} seed=${seed}"
    return 0
  fi

  if [[ ! -f "${selected_path}" ]]; then
    echo "[run][ratio][selection] method=${method} ratio=${ratio} seed=${seed}"
    python -m baselines.runners.run_baseline_selection \
      --method "${method}" \
      --ratio "${ratio}" \
      --dataset_name "${BASELINE_DATASET}" \
      --split train \
      --image_encoder "${BASELINE_IMAGE_ENCODER}" \
      --text_encoder "${BASELINE_TEXT_ENCODER}" \
      --feature_source "${BASELINE_FEATURE_SOURCE}" \
      --output_dir "${BASELINE_OUTPUT_ROOT}" \
      --config "${BASELINE_CONFIG}" \
      --output_layout ratio \
      --seed "${seed}" \
      --device "${BASELINE_DEVICE}"
  else
    echo "[skip][ratio][selection] exists method=${method} ratio=${ratio} seed=${seed}"
  fi

  echo "[run][ratio][eval] method=${method} ratio=${ratio} seed=${seed}"
  python -m baselines.runners.evaluate_baseline_subsets \
    --baseline_result_dir "${run_dir}" \
    --dataset_name "${BASELINE_DATASET}" \
    --image_encoder "${BASELINE_IMAGE_ENCODER}" \
    --text_encoder "${BASELINE_TEXT_ENCODER}" \
    --feature_source "${BASELINE_FEATURE_SOURCE}" \
    --image_root "${BASELINE_IMAGE_ROOT}" \
    --ann_root "${BASELINE_ANN_ROOT}" \
    --device "${BASELINE_DEVICE}" \
    --epochs "${BASELINE_EPOCHS}" \
    --batch_size_train "${BASELINE_BATCH_TRAIN}" \
    --batch_size_test "${BASELINE_BATCH_TEST}" \
    --text_batch_size "${BASELINE_TEXT_BATCH}" \
    --num_workers "${BASELINE_NUM_WORKERS}" \
    --eval_interval 1 \
    --no_aug
}

run_ratio_selection_only() {
  local method="$1"
  local ratio="$2"
  local seed="$3"

  local ratio_tag
  ratio_tag="$(ratio_to_tag "${ratio}")"
  local model_tag="${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}"
  local run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${ratio_tag}/${method}/seed_${seed}"
  local selected_path="${run_dir}/selected_indices.json"

  if [[ -f "${selected_path}" ]]; then
    echo "[skip][ratio][selection-only] exists method=${method} ratio=${ratio} seed=${seed}"
    return 0
  fi

  echo "[run][ratio][selection-only] method=${method} ratio=${ratio} seed=${seed}"
  python -m baselines.runners.run_baseline_selection \
    --method "${method}" \
    --ratio "${ratio}" \
    --dataset_name "${BASELINE_DATASET}" \
    --split train \
    --image_encoder "${BASELINE_IMAGE_ENCODER}" \
    --text_encoder "${BASELINE_TEXT_ENCODER}" \
    --feature_source "${BASELINE_FEATURE_SOURCE}" \
    --output_dir "${BASELINE_OUTPUT_ROOT}" \
    --config "${BASELINE_CONFIG}" \
    --output_layout ratio \
    --seed "${seed}" \
    --device "${BASELINE_DEVICE}"
}

# 1) Absolute budgets
for seed in ${BASELINE_SEEDS}; do
  # dataprophet: run one master ratio selection, then derive all budgets.
  if [[ " ${BASELINE_METHODS} " == *" dataprophet "* ]]; then
    ensure_dataprophet_master_selection "${seed}"
    for budget in ${ABS_BUDGETS}; do
      materialize_dataprophet_budget_from_master_ratio "${seed}" "${budget}" "${DATAPROPHET_MASTER_RATIO}" || true
      run_abs_job "dataprophet" "${budget}" "${seed}"
    done
  fi

  for budget in ${ABS_BUDGETS}; do
    for method in ${BASELINE_METHODS}; do
      if [[ "${method}" == "dataprophet" ]]; then
        continue
      fi
      run_abs_job "${method}" "${budget}" "${seed}"
    done
  done
done

# 2) Ratio budgets (optional)
if [[ -n "${RATIOS}" ]]; then
  for seed in ${BASELINE_SEEDS}; do
    # dataprophet: derive all target ratios from the same master ratio run.
    if [[ " ${BASELINE_METHODS} " == *" dataprophet "* ]]; then
      ensure_dataprophet_master_selection "${seed}"
      for ratio in ${RATIOS}; do
        materialize_dataprophet_ratio_from_master_ratio "${seed}" "${ratio}" "${DATAPROPHET_MASTER_RATIO}" || true
        run_ratio_job "dataprophet" "${ratio}" "${seed}"
      done
    fi

    for ratio in ${RATIOS}; do
      for method in ${BASELINE_METHODS}; do
        if [[ "${method}" == "dataprophet" ]]; then
          continue
        fi
        run_ratio_job "${method}" "${ratio}" "${seed}"
      done
    done
  done
fi

# 3) Export merged tables
python -m baselines.runners.export_baseline_tables \
  --root "${BASELINE_OUTPUT_ROOT}" \
  --output_dir "${BASELINE_OUTPUT_ROOT}"

# 4) Keep one compact final table
python - <<PY
import csv
import os

src = os.path.join("${BASELINE_OUTPUT_ROOT}", "main_table_aligned.csv")
dst = os.path.join("${BASELINE_OUTPUT_ROOT}", "final_results_table.csv")
if not os.path.exists(src):
    raise FileNotFoundError(src)

keep_cols = [
    "method", "budget", "ratio", "seed", "dataset", "image_encoder", "text_encoder",
    "sample_unit", "I2T_R1", "I2T_R5", "I2T_R10", "T2I_R1", "T2I_R5", "T2I_R10",
    "MeanRecall", "selection_time", "train_time", "eval_time", "output_dir"
]

with open(src, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    if not r.get("dataset") and r.get("dataset_name"):
        r["dataset"] = r["dataset_name"]

with open(dst, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keep_cols)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k) for k in keep_cols})

print(dst)
PY

echo ""
echo "[coco-formal] done."
echo "[coco-formal] final table: artifacts/baselines_coco/final_results_table.csv"
