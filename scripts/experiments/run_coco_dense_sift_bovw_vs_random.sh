#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${COCO_BOVW_RANDOM_DATASET:-coco}"
BACKBONE="${COCO_BOVW_RANDOM_BACKBONE:-nfnet}"
TEXT_ENCODER="${COCO_BOVW_RANDOM_TEXT_ENCODER:-bert}"
SEEDS_STR="${COCO_BOVW_RANDOM_SEEDS:-0}"
BUDGETS_STR="${COCO_BOVW_RANDOM_BUDGETS:-100 200 500}"
RATIOS_STR="${COCO_BOVW_RANDOM_RATIOS:-0.01 0.02 0.03}"

DENSE_VARIANT="${COCO_BOVW_RANDOM_DENSE_VARIANT:-wavelet_main_dense_sift_bovw}"
RANDOM_VARIANT="${COCO_BOVW_RANDOM_RANDOM_VARIANT:-random_baseline}"
RUN_DENSE="${COCO_BOVW_RANDOM_RUN_DENSE:-1}"
RUN_RANDOM="${COCO_BOVW_RANDOM_RUN_RANDOM:-1}"

COCO_ARTIFACT_ROOT="${COCO_BOVW_RANDOM_ARTIFACT_ROOT:-artifacts_coco}"

DENSE_FEATURE_ROOT="${COCO_BOVW_RANDOM_DENSE_FEATURE_ROOT:-${COCO_ARTIFACT_ROOT}/feature_cache_dense_sift_bovw_coco}"
DENSE_TOPOLOGY_ROOT="${COCO_BOVW_RANDOM_DENSE_TOPOLOGY_ROOT:-${COCO_ARTIFACT_ROOT}/topology_graph_dense_sift_bovw_coco}"
DENSE_CROSS_ROOT="${COCO_BOVW_RANDOM_DENSE_CROSS_ROOT:-${COCO_ARTIFACT_ROOT}/cross_modal_topology_dense_sift_bovw_coco}"
DENSE_SELECTION_ROOT="${COCO_BOVW_RANDOM_DENSE_SELECTION_ROOT:-${COCO_ARTIFACT_ROOT}/subset_selection_dense_sift_bovw_coco}"
DENSE_TRAIN_ROOT="${COCO_BOVW_RANDOM_DENSE_TRAIN_ROOT:-${COCO_ARTIFACT_ROOT}/subset_train_dense_sift_bovw_coco}"
DENSE_SELECTION_USE_TORCHRUN="${COCO_BOVW_RANDOM_SELECTION_USE_TORCHRUN:-1}"
DENSE_SELECTION_NPROC_PER_NODE="${COCO_BOVW_RANDOM_SELECTION_NPROC_PER_NODE:-}"
DENSE_PROXY_BATCH_SIZE="${COCO_BOVW_RANDOM_PROXY_BATCH_SIZE:-1024}"
DENSE_PROXY_TARGET_BATCH_SIZE="${COCO_BOVW_RANDOM_PROXY_TARGET_BATCH_SIZE:-1024}"
DENSE_LSRC_BATCH_SIZE="${COCO_BOVW_RANDOM_LSRC_BATCH_SIZE:-1024}"

RANDOM_FEATURE_ROOT="${COCO_BOVW_RANDOM_RANDOM_FEATURE_ROOT:-${COCO_ARTIFACT_ROOT}/feature_cache_random_baseline_coco}"
RANDOM_SELECTION_ROOT="${COCO_BOVW_RANDOM_RANDOM_SELECTION_ROOT:-${COCO_ARTIFACT_ROOT}/subset_selection_random_baseline_coco}"
RANDOM_TRAIN_ROOT="${COCO_BOVW_RANDOM_RANDOM_TRAIN_ROOT:-${COCO_ARTIFACT_ROOT}/subset_train_random_baseline_coco}"

REPORT_NAME="${COCO_BOVW_RANDOM_REPORT_NAME:-coco_dense_sift_bovw_vs_random}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
REPORT_DIR="${COCO_ARTIFACT_ROOT}/reports/${REPORT_NAME}_${RUN_TIMESTAMP}"
mkdir -p "${REPORT_DIR}"

stage_log "COCO dense_sift_bovw vs random start: budgets=${BUDGETS_STR} ratios=${RATIOS_STR} seeds=${SEEDS_STR} run_dense=${RUN_DENSE} run_random=${RUN_RANDOM}"
stage_log "Dense selection torchrun=${DENSE_SELECTION_USE_TORCHRUN} nproc=${DENSE_SELECTION_NPROC_PER_NODE:-auto} proxy_batch=${DENSE_PROXY_BATCH_SIZE} target_batch=${DENSE_PROXY_TARGET_BATCH_SIZE} lsrc_batch=${DENSE_LSRC_BATCH_SIZE}"

if [[ "${RUN_DENSE}" == "1" ]]; then
  stage_log "Run COCO dense_sift_bovw main method"
  env \
    WAVELET_MAIN_BOVW_DATASET="${DATASET}" \
    WAVELET_MAIN_BOVW_BACKBONE="${BACKBONE}" \
    WAVELET_MAIN_BOVW_TEXT_ENCODER="${TEXT_ENCODER}" \
    WAVELET_MAIN_BOVW_VARIANT="${DENSE_VARIANT}" \
    WAVELET_MAIN_BOVW_BUDGETS="${BUDGETS_STR}" \
    WAVELET_MAIN_BOVW_RATIOS="${RATIOS_STR}" \
    WAVELET_MAIN_LATEST_SEEDS="${SEEDS_STR}" \
    WAVELET_MAIN_BOVW_FEATURE_CACHE_ROOT="${DENSE_FEATURE_ROOT}" \
    WAVELET_MAIN_BOVW_TOPOLOGY_ROOT="${DENSE_TOPOLOGY_ROOT}" \
    WAVELET_MAIN_BOVW_CROSS_OUTPUT_ROOT="${DENSE_CROSS_ROOT}" \
    WAVELET_MAIN_BOVW_SELECTION_OUTPUT_ROOT="${DENSE_SELECTION_ROOT}" \
    WAVELET_MAIN_BOVW_TRAIN_OUTPUT_ROOT="${DENSE_TRAIN_ROOT}" \
    WAVELET_MAIN_BOVW_REPORT_NAME="${REPORT_NAME}_dense_sift_bovw" \
    WAVELET_MAIN_LATEST_SELECTION_USE_TORCHRUN="${DENSE_SELECTION_USE_TORCHRUN}" \
    WAVELET_MAIN_LATEST_SELECTION_NPROC_PER_NODE="${DENSE_SELECTION_NPROC_PER_NODE}" \
    WAVELET_MAIN_LATEST_PROXY_BATCH_SIZE="${DENSE_PROXY_BATCH_SIZE}" \
    WAVELET_MAIN_LATEST_PROXY_TARGET_BATCH_SIZE="${DENSE_PROXY_TARGET_BATCH_SIZE}" \
    WAVELET_MAIN_LATEST_LSRC_BATCH_SIZE="${DENSE_LSRC_BATCH_SIZE}" \
    SELECTION_IMAGE_REPR_METHOD="dense_sift_bovw" \
    bash "${SCRIPT_DIR}/run_wavelet_main_dense_sift_bovw_combo.sh"
else
  stage_log "Skip COCO dense_sift_bovw main method because COCO_BOVW_RANDOM_RUN_DENSE=${RUN_DENSE}"
fi

if [[ "${RUN_RANDOM}" == "1" ]]; then
  stage_log "Run COCO random baseline"
  env \
    RANDOM_BASELINE_DATASET="${DATASET}" \
    RANDOM_BASELINE_BACKBONE="${BACKBONE}" \
    RANDOM_BASELINE_TEXT_ENCODER="${TEXT_ENCODER}" \
    RANDOM_BASELINE_VARIANT="${RANDOM_VARIANT}" \
    RANDOM_BASELINE_BUDGETS="${BUDGETS_STR}" \
    RANDOM_BASELINE_RATIOS="${RATIOS_STR}" \
    RANDOM_BASELINE_SEEDS="${SEEDS_STR}" \
    FEATURE_CACHE_ROOT="${RANDOM_FEATURE_ROOT}" \
    RANDOM_BASELINE_SELECTION_OUTPUT_ROOT="${RANDOM_SELECTION_ROOT}" \
    RANDOM_BASELINE_TRAIN_OUTPUT_ROOT="${RANDOM_TRAIN_ROOT}" \
    RANDOM_BASELINE_REPORT_NAME="${REPORT_NAME}_random" \
    bash "${SCRIPT_DIR}/run_random_baseline_combo.sh"
else
  stage_log "Skip COCO random baseline because COCO_BOVW_RANDOM_RUN_RANDOM=${RUN_RANDOM}"
fi

RAW_CSV_PATH="${REPORT_DIR}/coco_dense_sift_bovw_vs_random_raw.csv"
SUMMARY_CSV_PATH="${REPORT_DIR}/coco_dense_sift_bovw_vs_random_summary.csv"
MISSING_TXT_PATH="${REPORT_DIR}/missing_metrics.txt"

python - "${RAW_CSV_PATH}" "${SUMMARY_CSV_PATH}" "${MISSING_TXT_PATH}" "${DATASET}" "${BACKBONE}" "${TEXT_ENCODER}" "${BUDGETS_STR}" "${RATIOS_STR}" "${SEEDS_STR}" "${DENSE_VARIANT}" "${DENSE_TRAIN_ROOT}" "${RANDOM_VARIANT}" "${RANDOM_TRAIN_ROOT}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


raw_csv_path = Path(sys.argv[1])
summary_csv_path = Path(sys.argv[2])
missing_txt_path = Path(sys.argv[3])
dataset = sys.argv[4]
backbone = sys.argv[5]
text_encoder = sys.argv[6]
budgets = [item for item in sys.argv[7].split() if item.strip()]
ratios = [item for item in sys.argv[8].split() if item.strip()]
seeds = [item for item in sys.argv[9].split() if item.strip()]
dense_variant = sys.argv[10]
dense_train_root = Path(sys.argv[11])
random_variant = sys.argv[12]
random_train_root = Path(sys.argv[13])
model_tag = f"{backbone}_{text_encoder}"

methods = [
    {
        "method": "dense_sift_bovw",
        "variant": dense_variant,
        "train_root": dense_train_root,
    },
    {
        "method": "random",
        "variant": random_variant,
        "train_root": random_train_root,
    },
]

targets = []
for budget in budgets:
    targets.append(("abs", f"size_{int(budget):04d}", str(int(budget))))
for ratio in ratios:
    ratio_value = float(ratio)
    targets.append(("ratio", f"ratio_{int(round(ratio_value * 100)):02d}", f"{ratio_value:.6f}"))

raw_rows = []
missing = []
for method_cfg in methods:
    for budget_type, budget_tag, budget_value in targets:
        for seed in seeds:
            metrics_path = (
                method_cfg["train_root"]
                / dataset
                / model_tag
                / budget_tag
                / method_cfg["variant"]
                / f"seed_{int(seed)}"
                / "metrics.json"
            )
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            raw_rows.append(
                {
                    "dataset": dataset,
                    "model_tag": model_tag,
                    "method": method_cfg["method"],
                    "variant": method_cfg["variant"],
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

raw_fields = [
    "dataset",
    "model_tag",
    "method",
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
    key = (
        row["dataset"],
        row["model_tag"],
        row["method"],
        row["variant"],
        row["budget_type"],
        row["budget_tag"],
        row["budget_value"],
    )
    grouped.setdefault(key, []).append(row)

summary_rows = []
for key in sorted(grouped.keys()):
    dataset, model_tag, method, variant, budget_type, budget_tag, budget_value = key
    rows = grouped[key]
    summary = {
        "dataset": dataset,
        "model_tag": model_tag,
        "method": method,
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
    "method",
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

stage_log "COCO dense_sift_bovw vs random completed."
stage_log "Report dir: ${REPORT_DIR}"
stage_log "Raw csv: ${RAW_CSV_PATH}"
stage_log "Summary csv: ${SUMMARY_CSV_PATH}"
