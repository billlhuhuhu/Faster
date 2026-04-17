#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${MODULE_DIAG_DATASET:-flickr}"
BACKBONE="${MODULE_DIAG_BACKBONE:-nfnet}"
TEXT_ENCODER="${MODULE_DIAG_TEXT_ENCODER:-bert}"
SEEDS_STR="${MODULE_DIAG_SEEDS:-0}"
read -r -a SEEDS <<< "${SEEDS_STR}"
BUDGETS_STR="${MODULE_DIAG_BUDGETS:-100 200 500}"
read -r -a BUDGETS <<< "${BUDGETS_STR}"
RATIOS_STR="${MODULE_DIAG_RATIOS:-0.01 0.02 0.03}"
read -r -a RATIOS <<< "${RATIOS_STR}"

BASE_CROSS_ROOT="${MODULE_DIAG_CROSS_ROOT:-artifacts/cross_modal_topology_module_diagnostic}"
BASE_SELECTION_ROOT="${MODULE_DIAG_SELECTION_ROOT:-artifacts/subset_selection_module_diagnostic}"
BASE_TRAIN_ROOT="${MODULE_DIAG_TRAIN_ROOT:-artifacts/subset_train_module_diagnostic}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
REPORT_DIR="${REPORT_ROOT}/module_diagnostic_ablation_${DATASET}_${RUN_TIMESTAMP}"
mkdir -p "${REPORT_DIR}"

MODEL_TAG="$(sanitize_component "${BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
RAW_CSV_PATH="${REPORT_DIR}/module_diagnostic_ablation_raw.csv"
SUMMARY_CSV_PATH="${REPORT_DIR}/module_diagnostic_ablation_summary.csv"
MISSING_TXT_PATH="${REPORT_DIR}/missing_metrics.txt"

exp_stage2_flag() {
  case "$1" in
    0|2|3) echo "0" ;;
    1|4) echo "1" ;;
    *) echo "1" ;;
  esac
}

exp_stage3_flag() {
  case "$1" in
    0|1|3) echo "0" ;;
    2|4) echo "1" ;;
    *) echo "1" ;;
  esac
}

exp_stage4_flag() {
  case "$1" in
    0|1|2) echo "0" ;;
    3|4) echo "1" ;;
    *) echo "1" ;;
  esac
}

exp_note() {
  # Diagnostic logic:
  # Exp 0 checks whether the plain averaged graph plus pure geometric selection
  # already beats random. If not, stage-one graph construction is suspect.
  # Exp 1 isolates stage two. A drop vs Exp 0 suggests asymmetric correction is too aggressive.
  # Exp 2 isolates stage three. A drop vs Exp 0 suggests multiscale fusion weights are harming useful modality cues.
  # Exp 3 isolates stage four. A drop vs Exp 0 suggests LSRC repulsion/coverage is over-regularizing proxy placement.
  # Exp 4 is the full method and serves as the control for additive effects.
  case "$1" in
    0) echo "Plain dual-modality average graph plus pure geometric subset selection baseline." ;;
    1) echo "Only stage-two asymmetric correction enabled to test whether correction alone hurts retrieval." ;;
    2) echo "Only stage-three multiscale fusion enabled to test whether fusion weights harm retrieval." ;;
    3) echo "Only stage-four LSRC repulsion enabled on the plain average graph to test over-regularization." ;;
    4) echo "Full system with stage-two correction, stage-three fusion, and stage-four LSRC enabled." ;;
    *) echo "Custom module-diagnostic experiment." ;;
  esac
}

stage_log "Module-diagnostic ablation start: dataset=${DATASET} budgets=${BUDGETS[*]} ratios=${RATIOS[*]} seeds=${SEEDS[*]}"

for exp_id in 0 1 2 3 4; do
  exp_variant="module_diag_exp${exp_id}"
  exp_cross_root="${BASE_CROSS_ROOT}/exp_${exp_id}"
  exp_selection_root="${BASE_SELECTION_ROOT}/exp_${exp_id}"
  exp_train_root="${BASE_TRAIN_ROOT}/exp_${exp_id}"
  exp_report_name="module_diag_exp${exp_id}"

  stage_log "Run Exp ${exp_id}: stage2=$(exp_stage2_flag "${exp_id}") stage3=$(exp_stage3_flag "${exp_id}") stage4=$(exp_stage4_flag "${exp_id}")"
  stage_log "Exp ${exp_id} note: $(exp_note "${exp_id}")"

  WAVELET_MAIN_SCALE_ENTROPY_VARIANT="${exp_variant}" \
  WAVELET_MAIN_SCALE_ENTROPY_BUDGETS="${BUDGETS_STR}" \
  WAVELET_MAIN_SCALE_ENTROPY_RATIOS="${RATIOS_STR}" \
  WAVELET_MAIN_SCALE_ENTROPY_CROSS_OUTPUT_ROOT="${exp_cross_root}" \
  WAVELET_MAIN_SCALE_ENTROPY_SELECTION_OUTPUT_ROOT="${exp_selection_root}" \
  WAVELET_MAIN_SCALE_ENTROPY_TRAIN_OUTPUT_ROOT="${exp_train_root}" \
  WAVELET_MAIN_SCALE_ENTROPY_REPORT_NAME="${exp_report_name}" \
  WAVELET_MAIN_LATEST_DATASET="${DATASET}" \
  WAVELET_MAIN_LATEST_BACKBONE="${BACKBONE}" \
  WAVELET_MAIN_LATEST_TEXT_ENCODER="${TEXT_ENCODER}" \
  WAVELET_MAIN_LATEST_SEEDS="${SEEDS_STR}" \
  WAVELET_MAIN_LATEST_DIAGNOSTIC_EXPERIMENT_ID="${exp_id}" \
  bash "${SCRIPT_DIR}/run_wavelet_main_scale_entropy_combo.sh"
done

python - "${BASE_TRAIN_ROOT}" "${DATASET}" "${MODEL_TAG}" "${RAW_CSV_PATH}" "${SUMMARY_CSV_PATH}" "${MISSING_TXT_PATH}" "${BUDGETS_STR}" "${RATIOS_STR}" "${SEEDS_STR}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def exp_note(exp_id):
    notes = {
        0: "Exp 0: stage2 OFF, stage3 OFF, stage4 OFF. Plain average graph plus pure geometric selection baseline.",
        1: "Exp 1: stage2 ON, stage3 OFF, stage4 OFF. Isolates whether asymmetric correction alone hurts retrieval.",
        2: "Exp 2: stage2 OFF, stage3 ON, stage4 OFF. Isolates whether multiscale fusion weights hurt retrieval.",
        3: "Exp 3: stage2 OFF, stage3 OFF, stage4 ON. Isolates whether LSRC repulsion over-regularizes proxy placement.",
        4: "Exp 4: stage2 ON, stage3 ON, stage4 ON. Full method control group.",
    }
    return notes.get(exp_id, "Custom module-diagnostic experiment.")


base_train_root = Path(sys.argv[1])
dataset = sys.argv[2]
model_tag = sys.argv[3]
raw_csv_path = Path(sys.argv[4])
summary_csv_path = Path(sys.argv[5])
missing_txt_path = Path(sys.argv[6])
budgets = [item for item in sys.argv[7].split() if item.strip()]
ratios = [item for item in sys.argv[8].split() if item.strip()]
seeds = [item for item in sys.argv[9].split() if item.strip()]

raw_rows = []
missing = []

targets = []
for budget in budgets:
    targets.append(("abs", f"size_{int(budget):04d}", str(int(budget))))
for ratio in ratios:
    ratio_value = float(ratio)
    targets.append(("ratio", f"ratio_{int(round(ratio_value * 100)):02d}", f"{ratio_value:.6f}"))

for exp_id in range(5):
    variant = f"module_diag_exp{exp_id}"
    for budget_type, budget_tag, budget_value in targets:
        for seed in seeds:
            metrics_path = base_train_root / f"exp_{exp_id}" / dataset / model_tag / budget_tag / variant / f"seed_{int(seed)}" / "metrics.json"
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            raw_rows.append(
                {
                    "dataset": dataset,
                    "model_tag": model_tag,
                    "variant": variant,
                    "exp_id": int(payload.get("diagnostic_experiment_id", exp_id)),
                    "stage2_on": bool(payload.get("enable_stage2_correction", True)),
                    "stage3_on": bool(payload.get("enable_stage3_fusion", True)),
                    "stage4_on": bool(payload.get("enable_stage4_lsrc", True)),
                    "experiment_note": exp_note(int(payload.get("diagnostic_experiment_id", exp_id))),
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
    "exp_id",
    "stage2_on",
    "stage3_on",
    "stage4_on",
    "experiment_note",
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
        row["variant"],
        row["exp_id"],
        row["stage2_on"],
        row["stage3_on"],
        row["stage4_on"],
        row["experiment_note"],
        row["budget_type"],
        row["budget_tag"],
        row["budget_value"],
    )
    grouped.setdefault(key, []).append(row)

summary_rows = []
for key in sorted(grouped.keys(), key=lambda item: (item[3], item[8], item[9])):
    dataset, model_tag, variant, exp_id, stage2_on, stage3_on, stage4_on, experiment_note, budget_type, budget_tag, budget_value = key
    rows = grouped[key]
    summary = {
        "dataset": dataset,
        "model_tag": model_tag,
        "variant": variant,
        "exp_id": exp_id,
        "stage2_on": stage2_on,
        "stage3_on": stage3_on,
        "stage4_on": stage4_on,
        "experiment_note": experiment_note,
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
    "exp_id",
    "stage2_on",
    "stage3_on",
    "stage4_on",
    "experiment_note",
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

stage_log "Module-diagnostic ablation completed."
stage_log "Report dir: ${REPORT_DIR}"
stage_log "Raw table: ${RAW_CSV_PATH}"
stage_log "Summary table: ${SUMMARY_CSV_PATH}"
