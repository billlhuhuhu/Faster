#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${HOG_WHITEN_COMPARE_DATASET:-flickr}"
BACKBONE="${HOG_WHITEN_COMPARE_BACKBONE:-nfnet}"
TEXT_ENCODER="${HOG_WHITEN_COMPARE_TEXT_ENCODER:-bert}"
SEEDS_STR="${HOG_WHITEN_COMPARE_SEEDS:-0}"
BUDGETS_STR="${HOG_WHITEN_COMPARE_BUDGETS:-100 200 500}"
RATIOS_STR="${HOG_WHITEN_COMPARE_RATIOS-0.01 0.02 0.03}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
REPORT_NAME="${HOG_WHITEN_COMPARE_REPORT_NAME:-hog_hellinger_chi2_whitening}"
REPORT_DIR="${REPORT_ROOT}/${REPORT_NAME}_${DATASET}_${RUN_TIMESTAMP}"
mkdir -p "${REPORT_DIR}"

run_one_variant() {
  local image_mode="$1"
  local variant_name="$2"
  local feature_root="$3"
  local topology_root="$4"
  local cross_root="$5"
  local selection_root="$6"
  local train_root="$7"
  local report_name="$8"

  stage_log "Run hog-whitening experiment: mode=${image_mode} variant=${variant_name}"
  env \
    WAVELET_MAIN_LATEST_DATASET="${DATASET}" \
    WAVELET_MAIN_LATEST_BACKBONE="${BACKBONE}" \
    WAVELET_MAIN_LATEST_TEXT_ENCODER="${TEXT_ENCODER}" \
    WAVELET_MAIN_LATEST_VARIANT="${variant_name}" \
    WAVELET_MAIN_LATEST_BUDGETS="${BUDGETS_STR}" \
    WAVELET_MAIN_LATEST_RATIOS="${RATIOS_STR}" \
    WAVELET_MAIN_LATEST_SEEDS="${SEEDS_STR}" \
    FEATURE_CACHE_ROOT="${feature_root}" \
    TOPOLOGY_ROOT="${topology_root}" \
    WAVELET_MAIN_LATEST_CROSS_OUTPUT_ROOT="${cross_root}" \
    WAVELET_MAIN_LATEST_SELECTION_OUTPUT_ROOT="${selection_root}" \
    WAVELET_MAIN_LATEST_TRAIN_OUTPUT_ROOT="${train_root}" \
    WAVELET_MAIN_LATEST_REPORT_NAME="${report_name}" \
    SELECTION_IMAGE_REPR_METHOD="${image_mode}" \
    bash "${SCRIPT_DIR}/run_wavelet_main_latest_combo.sh"
}

run_one_variant \
  "hog_color_hellinger_pca_whitening" \
  "wavelet_main_hog_hellinger_pca_whitening" \
  "artifacts/feature_cache_hog_hellinger_pca_whitening" \
  "artifacts/topology_graph_hog_hellinger_pca_whitening" \
  "artifacts/cross_modal_topology_hog_hellinger_pca_whitening" \
  "artifacts/subset_selection_hog_hellinger_pca_whitening" \
  "artifacts/subset_train_hog_hellinger_pca_whitening" \
  "hog_hellinger_pca_whitening"

run_one_variant \
  "hog_color_chi2_pca_whitening" \
  "wavelet_main_hog_chi2_pca_whitening" \
  "artifacts/feature_cache_hog_chi2_pca_whitening" \
  "artifacts/topology_graph_hog_chi2_pca_whitening" \
  "artifacts/cross_modal_topology_hog_chi2_pca_whitening" \
  "artifacts/subset_selection_hog_chi2_pca_whitening" \
  "artifacts/subset_train_hog_chi2_pca_whitening" \
  "hog_chi2_pca_whitening"

RAW_CSV_PATH="${REPORT_DIR}/hog_whitening_compare_raw.csv"
SUMMARY_CSV_PATH="${REPORT_DIR}/hog_whitening_compare_summary.csv"
MISSING_TXT_PATH="${REPORT_DIR}/missing_metrics.txt"

python - "${RAW_CSV_PATH}" "${SUMMARY_CSV_PATH}" "${MISSING_TXT_PATH}" "${DATASET}" "${BACKBONE}" "${TEXT_ENCODER}" "${BUDGETS_STR}" "${RATIOS_STR}" "${SEEDS_STR}" "${K_NEIGHBORS}" "${TOPOLOGY_METRIC_IMAGE}" <<'PY'
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
k_neighbors = str(sys.argv[10])
topology_metric_image = str(sys.argv[11])
model_tag = f"{backbone}_{text_encoder}"

variants = [
    {
        "image_feature_mode": "hog_color_hellinger_pca_whitening",
        "variant": "wavelet_main_hog_hellinger_pca_whitening",
        "train_root": Path("artifacts/subset_train_hog_hellinger_pca_whitening"),
        "topology_root": Path("artifacts/topology_graph_hog_hellinger_pca_whitening"),
    },
    {
        "image_feature_mode": "hog_color_chi2_pca_whitening",
        "variant": "wavelet_main_hog_chi2_pca_whitening",
        "train_root": Path("artifacts/subset_train_hog_chi2_pca_whitening"),
        "topology_root": Path("artifacts/topology_graph_hog_chi2_pca_whitening"),
    },
]

topology_graph_tag = f"k{k_neighbors}_{topology_metric_image}"
topology_suffix = Path(dataset) / "train" / model_tag / "image" / topology_graph_tag / "summary.json"
for variant_cfg in variants:
    topology_summary_path = variant_cfg["topology_root"] / topology_suffix
    topology_summary = None
    if topology_summary_path.exists():
        topology_summary = json.loads(topology_summary_path.read_text(encoding="utf-8"))
    variant_cfg["topology_summary"] = topology_summary
    variant_cfg["topology_summary_path"] = topology_summary_path

targets = []
for budget in budgets:
    targets.append(("abs", f"size_{int(budget):04d}", str(int(budget))))
for ratio in ratios:
    ratio_value = float(ratio)
    targets.append(("ratio", f"ratio_{int(round(ratio_value * 100)):02d}", f"{ratio_value:.6f}"))

raw_rows = []
missing = []
for variant_cfg in variants:
    for budget_type, budget_tag, budget_value in targets:
        for seed in seeds:
            metrics_path = (
                variant_cfg["train_root"]
                / dataset
                / model_tag
                / budget_tag
                / variant_cfg["variant"]
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
                    "image_feature_mode": variant_cfg["image_feature_mode"],
                    "variant": variant_cfg["variant"],
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
                    "image_spectral_entropy": (
                        float(variant_cfg["topology_summary"].get("spectral_entropy"))
                        if variant_cfg["topology_summary"] is not None
                        else None
                    ),
                    "image_collapse_score": (
                        float(variant_cfg["topology_summary"].get("collapse_score"))
                        if variant_cfg["topology_summary"] is not None
                        else None
                    ),
                    "image_avg_degree": (
                        float(variant_cfg["topology_summary"].get("avg_degree"))
                        if variant_cfg["topology_summary"] is not None
                        else None
                    ),
                    "topology_summary_path": str(variant_cfg["topology_summary_path"]),
                    "metrics_path": str(metrics_path),
                }
            )

raw_fields = [
    "dataset",
    "model_tag",
    "image_feature_mode",
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
    "image_spectral_entropy",
    "image_collapse_score",
    "image_avg_degree",
    "topology_summary_path",
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
        row["image_feature_mode"],
        row["variant"],
        row["budget_type"],
        row["budget_tag"],
        row["budget_value"],
    )
    grouped.setdefault(key, []).append(row)

summary_rows = []
for key in sorted(grouped.keys()):
    dataset, model_tag, image_feature_mode, variant, budget_type, budget_tag, budget_value = key
    rows = grouped[key]
    summary = {
        "dataset": dataset,
        "model_tag": model_tag,
        "image_feature_mode": image_feature_mode,
        "variant": variant,
        "budget_type": budget_type,
        "budget_tag": budget_tag,
        "budget_value": budget_value,
        "num_runs": len(rows),
        "image_spectral_entropy": rows[0]["image_spectral_entropy"],
        "image_collapse_score": rows[0]["image_collapse_score"],
        "image_avg_degree": rows[0]["image_avg_degree"],
    }
    for metric in ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall"]:
        values = [float(item[metric]) for item in rows]
        summary[f"{metric}_mean"] = float(sum(values) / len(values))
        summary[f"{metric}_std"] = safe_std(values)
    summary_rows.append(summary)

summary_fields = [
    "dataset",
    "model_tag",
    "image_feature_mode",
    "variant",
    "budget_type",
    "budget_tag",
    "budget_value",
    "num_runs",
    "image_spectral_entropy",
    "image_collapse_score",
    "image_avg_degree",
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

stage_log "HOG Hellinger/Chi2 whitening compare completed."
stage_log "Report dir: ${REPORT_DIR}"
stage_log "Raw csv: ${RAW_CSV_PATH}"
stage_log "Summary csv: ${SUMMARY_CSV_PATH}"
