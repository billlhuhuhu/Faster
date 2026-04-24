#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET="${RETRIEVAL_ABLATION_DATASET:-flickr}"
BACKBONE="${RETRIEVAL_ABLATION_BACKBONE:-nfnet}"
TEXT_ENCODER="${RETRIEVAL_ABLATION_TEXT_ENCODER:-bert}"
BUDGETS_STR="${RETRIEVAL_ABLATION_BUDGETS:-100 200 500}"
RATIOS_STR="${RETRIEVAL_ABLATION_RATIOS:-0.01 0.02 0.03}"
SEEDS_STR="${RETRIEVAL_ABLATION_SEEDS:-0}"

ABLATION_ROOT="${RETRIEVAL_ABLATION_ROOT:-artifacts/retrieval_dense_sift_bovw_module_ablation}"
REPORT_ROOT_ABL="${RETRIEVAL_ABLATION_REPORT_ROOT:-${ABLATION_ROOT}/reports}"
LOG_ROOT_ABL="${RETRIEVAL_ABLATION_LOG_ROOT:-${ABLATION_ROOT}/logs}"

FEATURE_CACHE_ROOT_SHARED="${RETRIEVAL_ABLATION_FEATURE_CACHE_ROOT:-artifacts/feature_cache_dense_sift_bovw}"
TOPOLOGY_ROOT_SHARED="${RETRIEVAL_ABLATION_TOPOLOGY_ROOT:-artifacts/topology_graph_dense_sift_bovw}"

mkdir -p "${REPORT_ROOT_ABL}" "${LOG_ROOT_ABL}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_variant() {
  local variant="$1"
  local note="$2"
  shift 2

  echo "[$(timestamp)] Retrieval ablation start: ${variant}"
  echo "[$(timestamp)] Note: ${note}"

  env \
    WAVELET_MAIN_LATEST_DATASET="${DATASET}" \
    WAVELET_MAIN_LATEST_BACKBONE="${BACKBONE}" \
    WAVELET_MAIN_LATEST_TEXT_ENCODER="${TEXT_ENCODER}" \
    WAVELET_MAIN_LATEST_VARIANT="${variant}" \
    WAVELET_MAIN_LATEST_BUDGETS="${BUDGETS_STR}" \
    WAVELET_MAIN_LATEST_RATIOS="${RATIOS_STR}" \
    WAVELET_MAIN_LATEST_SEEDS="${SEEDS_STR}" \
    FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT_SHARED}" \
    TOPOLOGY_ROOT="${TOPOLOGY_ROOT_SHARED}" \
    WAVELET_MAIN_LATEST_CROSS_OUTPUT_ROOT="${ABLATION_ROOT}/cross_modal/${variant}" \
    WAVELET_MAIN_LATEST_SELECTION_OUTPUT_ROOT="${ABLATION_ROOT}/subset_selection/${variant}" \
    WAVELET_MAIN_LATEST_TRAIN_OUTPUT_ROOT="${ABLATION_ROOT}/subset_train/${variant}" \
    REPORT_ROOT="${REPORT_ROOT_ABL}" \
    EXPERIMENT_LOG_ROOT="${LOG_ROOT_ABL}" \
    WAVELET_MAIN_LATEST_REPORT_NAME="${variant}" \
    SELECTION_IMAGE_REPR_METHOD="dense_sift_bovw" \
    WAVELET_MAIN_LATEST_WAVELET_FUSION_WEIGHT_MODE="collapse_aware" \
    "$@" \
    bash "${SCRIPT_DIR}/run_wavelet_main_latest_combo.sh"

  echo "[$(timestamp)] Retrieval ablation done: ${variant}"
}

# Exp A: remove stage-2 correction and stage-3 adaptive/wavelet fusion.
# The graph degrades to the plain 0.5A + 0.5B fusion path while keeping the
# downstream sampler unchanged.
run_variant \
  "ablate_plain_direct_fusion" \
  "Stage2 correction OFF, Stage3 fusion OFF; use plain dual-modality average graph." \
  WAVELET_MAIN_LATEST_STAGE2_SWITCH="0" \
  WAVELET_MAIN_LATEST_STAGE3_SWITCH="0" \
  WAVELET_MAIN_LATEST_STAGE4_SWITCH="1"

# Exp B: remove the LSRC/LORS local relation module from proxy optimization and
# matching, keeping correction/fusion and wavelet main alignment enabled.
run_variant \
  "ablate_no_lsrc" \
  "Stage4 LSRC/LORS OFF; correction, latent fusion, and wavelet main alignment remain ON." \
  WAVELET_MAIN_LATEST_STAGE2_SWITCH="1" \
  WAVELET_MAIN_LATEST_STAGE3_SWITCH="1" \
  WAVELET_MAIN_LATEST_STAGE4_SWITCH="0" \
  WAVELET_MAIN_LATEST_ENABLE_LSRC="0" \
  WAVELET_MAIN_LATEST_KEEP_LSRC="0" \
  WAVELET_MAIN_LATEST_LAMBDA_LSRC="0.0" \
  WAVELET_MAIN_LATEST_COST_ETA_LSRC="0.0"

# Exp C: remove the wavelet-domain main alignment. This falls back to the
# generic legacy PDCFD/global distribution proxy objective and disables wavelet
# matching cost, while retaining correction/fusion and LSRC.
run_variant \
  "ablate_no_wavelet_alignment" \
  "Replace wavelet_main with generic legacy pdcfd global distribution matching and remove wavelet matching cost." \
  WAVELET_MAIN_LATEST_STAGE2_SWITCH="1" \
  WAVELET_MAIN_LATEST_STAGE3_SWITCH="1" \
  WAVELET_MAIN_LATEST_STAGE4_SWITCH="1" \
  WAVELET_MAIN_LATEST_PROXY_LOSS_TYPE="pdcfd" \
  WAVELET_MAIN_LATEST_COST_BETA_WAVELET="0.0" \
  WAVELET_MAIN_LATEST_MATCHING_WAVELET_WEIGHT="0.0"

FINAL_RAW_CSV="${REPORT_ROOT_ABL}/retrieval_dense_sift_bovw_module_ablation_raw.csv"
FINAL_SUMMARY_CSV="${REPORT_ROOT_ABL}/retrieval_dense_sift_bovw_module_ablation_summary.csv"
FINAL_MISSING_TXT="${REPORT_ROOT_ABL}/retrieval_dense_sift_bovw_module_ablation_missing.txt"

python - "${ABLATION_ROOT}" "${DATASET}" "${BACKBONE}_${TEXT_ENCODER}" "${BUDGETS_STR}" "${RATIOS_STR}" "${SEEDS_STR}" "${FINAL_RAW_CSV}" "${FINAL_SUMMARY_CSV}" "${FINAL_MISSING_TXT}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset = sys.argv[2]
model_tag = sys.argv[3]
budgets = [item for item in sys.argv[4].split() if item.strip()]
ratios = [item for item in sys.argv[5].split() if item.strip()]
seeds = [item for item in sys.argv[6].split() if item.strip()]
raw_csv = Path(sys.argv[7])
summary_csv = Path(sys.argv[8])
missing_txt = Path(sys.argv[9])

variants = [
    ("ablate_plain_direct_fusion", "no_correction_no_adaptive_fusion"),
    ("ablate_no_lsrc", "no_lsrc_lors"),
    ("ablate_no_wavelet_alignment", "no_wavelet_alignment"),
]

targets = []
for budget in budgets:
    targets.append(("abs", f"size_{int(float(budget)):04d}", str(int(float(budget)))))
for ratio in ratios:
    ratio_value = float(ratio)
    targets.append(("ratio", f"ratio_{int(round(ratio_value * 100)):02d}", f"{ratio_value:.6f}"))

rows = []
missing = []
for variant, ablation in variants:
    train_root = root / "subset_train" / variant
    for budget_type, budget_tag, budget_value in targets:
        for seed in seeds:
            metrics_path = train_root / dataset / model_tag / budget_tag / variant / f"seed_{int(seed)}" / "metrics.json"
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {
                "ablation": ablation,
                "variant": variant,
                "dataset": dataset,
                "model_tag": model_tag,
                "budget_type": budget_type,
                "budget_tag": budget_tag,
                "budget_value": budget_value,
                "seed": int(seed),
                "metrics_path": str(metrics_path),
            }
            for metric in ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall"]:
                row[metric] = float(payload[metric])
            rows.append(row)

raw_csv.parent.mkdir(parents=True, exist_ok=True)
raw_fields = [
    "ablation", "variant", "dataset", "model_tag", "budget_type", "budget_tag",
    "budget_value", "seed", "i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1",
    "t2i_r5", "t2i_r10", "mean_recall", "metrics_path",
]
with raw_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=raw_fields)
    writer.writeheader()
    writer.writerows(rows)

def std(values):
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0

groups = {}
for row in rows:
    key = (row["ablation"], row["variant"], row["dataset"], row["model_tag"], row["budget_type"], row["budget_tag"], row["budget_value"])
    groups.setdefault(key, []).append(row)

summary_rows = []
for key in sorted(groups):
    ablation, variant, dataset, model_tag, budget_type, budget_tag, budget_value = key
    group_rows = groups[key]
    out = {
        "ablation": ablation,
        "variant": variant,
        "dataset": dataset,
        "model_tag": model_tag,
        "budget_type": budget_type,
        "budget_tag": budget_tag,
        "budget_value": budget_value,
        "num_runs": len(group_rows),
    }
    for metric in ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall"]:
        values = [float(item[metric]) for item in group_rows]
        out[f"{metric}_mean"] = float(sum(values) / len(values))
        out[f"{metric}_std"] = std(values)
    summary_rows.append(out)

summary_fields = [
    "ablation", "variant", "dataset", "model_tag", "budget_type", "budget_tag",
    "budget_value", "num_runs",
    "i2t_r1_mean", "i2t_r1_std", "i2t_r5_mean", "i2t_r5_std",
    "i2t_r10_mean", "i2t_r10_std", "t2i_r1_mean", "t2i_r1_std",
    "t2i_r5_mean", "t2i_r5_std", "t2i_r10_mean", "t2i_r10_std",
    "mean_recall_mean", "mean_recall_std",
]
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

missing_txt.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
print(f"saved final raw table: {raw_csv}")
print(f"saved final summary table: {summary_csv}")
print(f"saved missing list: {missing_txt}")
print(f"collected rows: {len(rows)}")
print(f"summary rows: {len(summary_rows)}")
PY

echo "[$(timestamp)] Retrieval dense_sift_bovw module ablation completed"
echo "  raw table: ${FINAL_RAW_CSV}"
echo "  summary table: ${FINAL_SUMMARY_CSV}"
echo "  missing list: ${FINAL_MISSING_TXT}"
