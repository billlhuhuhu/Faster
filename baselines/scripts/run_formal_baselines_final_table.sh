#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

BASELINE_CONFIG="${BASELINE_CONFIG:-baselines/configs/main_aligned_flickr_nfnet_bert.yaml}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-artifacts/baselines}"
BASELINE_FEATURE_SOURCE="${BASELINE_FEATURE_SOURCE:-artifacts/feature_cache}"
BASELINE_DEVICE="${BASELINE_DEVICE:-cuda:0}"
BASELINE_DATASET="${BASELINE_DATASET:-flickr}"
BASELINE_IMAGE_ENCODER="${BASELINE_IMAGE_ENCODER:-nfnet}"
BASELINE_TEXT_ENCODER="${BASELINE_TEXT_ENCODER:-bert}"
BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-data/flickr30k}"
BASELINE_ANN_ROOT="${BASELINE_ANN_ROOT:-data/Flickr30k_ann}"
BASELINE_SEEDS="${BASELINE_SEEDS:-0}"
BASELINE_METHODS="${BASELINE_METHODS:-entropy el2n grand gradmatch glister ccs-rand ccs-herd ccs-kcenter ccs-forget dq dfool nms adap_sne}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-20}"

# 绝对预算 + 比例预算
ABS_BUDGETS="${ABS_BUDGETS:-100 200 500}"
RATIOS="${RATIOS:-0.01 0.02 0.03}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-8}"
export LORS_CHECKPOINT_ROOT="${LORS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/distill_utils/checkpoints}"

echo "[formal] project_root=${PROJECT_ROOT}"
echo "[formal] output_root=${BASELINE_OUTPUT_ROOT}"
echo "[formal] device=${BASELINE_DEVICE}"
echo "[formal] methods=${BASELINE_METHODS}"
echo "[formal] abs_budgets=${ABS_BUDGETS}"
echo "[formal] ratios=${RATIOS}"
echo "[formal] seeds=${BASELINE_SEEDS}"
echo "[formal] checkpoints=${LORS_CHECKPOINT_ROOT}"

# 1) 先跑绝对预算（完整闭环）
python -m baselines.runners.run_main_aligned_baselines \
  --config "${BASELINE_CONFIG}" \
  --methods ${BASELINE_METHODS} \
  --budgets ${ABS_BUDGETS} \
  --seeds ${BASELINE_SEEDS} \
  --device "${BASELINE_DEVICE}" \
  --output_root "${BASELINE_OUTPUT_ROOT}" \
  --run_full_pipeline

# 2) 再跑比例预算（完整闭环）
model_tag="${BASELINE_IMAGE_ENCODER}_${BASELINE_TEXT_ENCODER}"
for ratio in ${RATIOS}; do
  ratio_tag=$(python - <<PY
ratio = float("${ratio}")
print(f"ratio_{int(round(ratio*100)):02d}")
PY
)
  for seed in ${BASELINE_SEEDS}; do
    for method in ${BASELINE_METHODS}; do
      echo "[formal][ratio] method=${method} ratio=${ratio} seed=${seed}"
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
        --candidate_pool_mode head \
        --seed "${seed}" \
        --device "${BASELINE_DEVICE}"

      run_dir="${BASELINE_OUTPUT_ROOT}/${BASELINE_DATASET}/train/${model_tag}/${ratio_tag}/${method}/seed_${seed}"
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
        --batch_size_train 64 \
        --batch_size_test 128 \
        --text_batch_size 1024 \
        --num_workers 4 \
        --eval_interval 1 \
        --no_aug
    done
  done
done

# 3) 导出统一表
python -m baselines.runners.export_baseline_tables \
  --root "${BASELINE_OUTPUT_ROOT}" \
  --output_dir "${BASELINE_OUTPUT_ROOT}"

# 4) 只保留你要的“一个总表”
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
    reader = csv.DictReader(f)
    rows = list(reader)

for r in rows:
    # 兼容不同键名
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
echo "[formal] done."
echo "[formal] final table: ${BASELINE_OUTPUT_ROOT}/final_results_table.csv"

