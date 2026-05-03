#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${ARCH3_CLEAN_DATASET:-flickr}"
RATIO="${ARCH3_CLEAN_RATIO:-0.03}"
TEXT_ENCODER="${ARCH3_CLEAN_TEXT_ENCODER:-bert}"
UPSTREAM_BACKBONE="${ARCH3_CLEAN_UPSTREAM_BACKBONE:-nfnet}"
EVAL_BACKBONES="${ARCH3_CLEAN_EVAL_BACKBONES:-nfnet resnet50 vit_b16}"
RUN_REPBLEND="${ARCH3_CLEAN_RUN_REPBLEND:-1}"
RUN_LORS="${ARCH3_CLEAN_RUN_LORS:-1}"

OUTPUT_ROOT="${ARCH3_CLEAN_OUTPUT_ROOT:-artifacts/arch3pct_nfnet_repblend_lors_clean}"
RUN_TAG="${ARCH3_CLEAN_RUN_TAG:-${DATASET}_ratio03_nfnet_repblend_lors_$(date '+%Y%m%d_%H%M%S')}"
REPORT_DIR="${OUTPUT_ROOT}/reports/${RUN_TAG}"
LOG_DIR="${OUTPUT_ROOT}/logs/${RUN_TAG}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

REPBLEND_REPORT_DIR="${OUTPUT_ROOT}/repblend/reports/${RUN_TAG}"
LORS_REPORT_ROOT="${REPORT_DIR}/lors_raw_reports"
LORS_LOG_ROOT_CLEAN="${LOG_DIR}/lors_logs"

stage_log "Clean 3% architecture + energy experiment start"
stage_log "  dataset=${DATASET} ratio=${RATIO} upstream=${UPSTREAM_BACKBONE}_${TEXT_ENCODER}"
stage_log "  eval_backbones=${EVAL_BACKBONES}"
stage_log "  output=${OUTPUT_ROOT}"
stage_log "  run_repblend=${RUN_REPBLEND} run_lors=${RUN_LORS}"

if [[ "${RUN_REPBLEND}" == "1" ]]; then
  stage_log "Run RepBlend 3%: upstream=${UPSTREAM_BACKBONE}, vit_b16=low_lr_finetune"
  ARCH3_DATASET="${DATASET}" \
  ARCH3_UPSTREAM_BACKBONE="${UPSTREAM_BACKBONE}" \
  ARCH3_TEXT_ENCODER="${TEXT_ENCODER}" \
  ARCH3_EVAL_BACKBONES="${EVAL_BACKBONES}" \
  ARCH3_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  ARCH3_RUN_TAG="${RUN_TAG}" \
  REPBLEND_ROOT="${REPBLEND_ROOT:-${PROJECT_ROOT}/RepBlend}" \
  REPBLEND_FORCE_REDISTILL="${REPBLEND_FORCE_REDISTILL:-1}" \
  REPBLEND_VIT_USE_LOW_LR_FINETUNE="${REPBLEND_VIT_USE_LOW_LR_FINETUNE:-1}" \
  REPBLEND_VIT_LOWLR_IMG="${REPBLEND_VIT_LOWLR_IMG:-0.001}" \
  REPBLEND_VIT_LOWLR_TXT="${REPBLEND_VIT_LOWLR_TXT:-0.05}" \
  REPBLEND_VIT_EPOCH_EVAL_TRAIN="${REPBLEND_VIT_EPOCH_EVAL_TRAIN:-300}" \
  REPBLEND_VIT_BATCH_TRAIN="${REPBLEND_VIT_BATCH_TRAIN:-32}" \
  REPBLEND_VIT_BATCH_TEST="${REPBLEND_VIT_BATCH_TEST:-64}" \
  ENERGY_PREFER_ZEUS="${ENERGY_PREFER_ZEUS:-1}" \
  bash "${SCRIPT_DIR}/run_arch3pct_repblend_energy.sh" 2>&1 | tee "${LOG_DIR}/repblend_pipeline.log"
fi

if [[ "${RUN_LORS}" == "1" ]]; then
  stage_log "Run LoRS 3% using run_lors_ratio_crossarch.sh roots/config"
  REPORT_ROOT="${LORS_REPORT_ROOT}" \
  EXPERIMENT_LOG_ROOT="${LORS_LOG_ROOT_CLEAN}" \
  LORS_RATIO_DATASET="${DATASET}" \
  LORS_RATIO_VALUES="${RATIO}" \
  LORS_RATIO_DEVICE="${LORS_RATIO_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}" \
  LORS_RATIO_BUFFER_ROOT="${LORS_RATIO_BUFFER_ROOT:-buffers_formal_v2}" \
  LORS_RATIO_LOG_ROOT="${LORS_RATIO_LOG_ROOT:-logged_files_formal_v2}" \
  LORS_RATIO_DISTILL_BACKBONE="${UPSTREAM_BACKBONE}" \
  LORS_RATIO_TEXT_ENCODER="${TEXT_ENCODER}" \
  LORS_RATIO_EVAL_BACKBONES="${EVAL_BACKBONES}" \
  LORS_RATIO_FORCE_REDISTILL="${LORS_RATIO_FORCE_REDISTILL:-0}" \
  LORS_BATCH_TRAIN="${LORS_BATCH_TRAIN:-128}" \
  LORS_BATCH_TEST="${LORS_BATCH_TEST:-128}" \
  LORS_VIT_USE_LOW_LR_FINETUNE="${LORS_VIT_USE_LOW_LR_FINETUNE:-1}" \
  LORS_VIT_LOWLR_IMG="${LORS_VIT_LOWLR_IMG:-0.001}" \
  LORS_VIT_LOWLR_TXT="${LORS_VIT_LOWLR_TXT:-0.05}" \
  LORS_VIT_EPOCH_EVAL_TRAIN="${LORS_VIT_EPOCH_EVAL_TRAIN:-300}" \
  LORS_VIT_BATCH_TRAIN="${LORS_VIT_BATCH_TRAIN:-32}" \
  LORS_VIT_BATCH_TEST="${LORS_VIT_BATCH_TEST:-64}" \
  LORS_DISABLED_WANDB="${LORS_DISABLED_WANDB:-True}" \
  bash "${SCRIPT_DIR}/run_lors_ratio_crossarch.sh" 2>&1 | tee "${LOG_DIR}/lors_ratio_crossarch.log"
fi

stage_log "Merge RepBlend + LoRS tables"
python - "${REPORT_DIR}" "${REPBLEND_REPORT_DIR}" "${LORS_REPORT_ROOT}" "${DATASET}" "${RATIO}" "${UPSTREAM_BACKBONE}" "${TEXT_ENCODER}" <<'PY'
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path


def to_float(value):
    try:
        if value in {"", None, "-"}:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ratio_tag(ratio):
    return f"ratio_{int(round(float(ratio) * 100)):02d}"


def latest_lors_csv(root):
    candidates = sorted(Path(root).glob("lors_ratio_crossarch_*/lors_ratio_crossarch.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def repblend_rows(repblend_report_dir):
    detail_path = Path(repblend_report_dir) / "supplemental_detail.csv"
    rows = []
    for row in read_csv(detail_path):
        row = dict(row)
        row["method"] = "repblend"
        row["upstream_backbone"] = upstream
        row["text_encoder"] = text_encoder
        row["source_table"] = str(detail_path)
        rows.append(row)
    return rows


def lors_rows(lors_csv_path):
    if not lors_csv_path:
        return []
    rows = []
    for row in read_csv(lors_csv_path):
        mean_recall = to_float(row.get("r_mean"))
        eval_backbone = row.get("eval_backbone", "")
        item = {
            "method": "lors",
            "dataset": row.get("dataset", dataset),
            "budget_tag": ratio_tag(ratio),
            "budget_type": "ratio",
            "budget_value": f"{float(ratio):.6f}",
            "eval_backbone": eval_backbone,
            "mean_recall": "" if mean_recall is None else mean_recall,
            "test_accuracy": "" if mean_recall is None else mean_recall,
            "i2t_r1": row.get("txt_r1", ""),
            "i2t_r5": row.get("txt_r5", ""),
            "i2t_r10": row.get("txt_r10", ""),
            "t2i_r1": row.get("img_r1", ""),
            "t2i_r5": row.get("img_r5", ""),
            "t2i_r10": row.get("img_r10", ""),
            "stage": "training_eval_reused_distill",
            "seconds": "",
            "gpu_count": os.environ.get("LORS_REUSED_GPU_COUNT", ""),
            "gpu_hours": "",
            "gpu_energy_Wh": "",
            "cpu_energy_Wh": "",
            "energy_Wh": "",
            "selection_time_seconds": "",
            "training_time_seconds": "",
            "total_time_seconds": "",
            "selection_energy_Wh": "",
            "training_energy_Wh": "",
            "total_energy_Wh": "",
            "checkpoint_path": row.get("checkpoint_path", ""),
            "evaluate_log": row.get("evaluate_log", ""),
            "source_table": str(lors_csv_path),
            "upstream_backbone": row.get("distill_backbone", upstream),
            "text_encoder": text_encoder,
            "note": "LoRS reuses run_lors_ratio_crossarch.sh buffers/distill logic; energy is blank unless manual reused energy env vars are provided.",
        }
        rows.append(item)
    selection_energy = to_float(os.environ.get("LORS_REUSED_SELECTION_ENERGY_WH"))
    selection_seconds = to_float(os.environ.get("LORS_REUSED_SELECTION_TIME_SECONDS"))
    if selection_energy is not None or selection_seconds is not None:
        rows.append(
            {
                "method": "lors",
                "dataset": dataset,
                "budget_tag": ratio_tag(ratio),
                "budget_type": "ratio",
                "budget_value": f"{float(ratio):.6f}",
                "eval_backbone": "",
                "stage": "selection_reused_manual",
                "seconds": "" if selection_seconds is None else selection_seconds,
                "gpu_count": os.environ.get("LORS_REUSED_GPU_COUNT", "1"),
                "gpu_energy_Wh": "" if selection_energy is None else selection_energy,
                "energy_Wh": "" if selection_energy is None else selection_energy,
                "upstream_backbone": upstream,
                "text_encoder": text_encoder,
                "source_table": str(lors_csv_path),
                "note": "Manual reused LoRS selection/buffer/distill energy.",
            }
        )
    return rows


def build_architecture(rows):
    groups = {}
    for row in rows:
        mr = to_float(row.get("mean_recall"))
        eval_backbone = row.get("eval_backbone")
        if mr is None or not eval_backbone:
            continue
        key = (row.get("method"), row.get("dataset"), row.get("budget_tag"), row.get("budget_type"), str(row.get("budget_value")), row.get("upstream_backbone"), row.get("text_encoder"))
        groups.setdefault(key, []).append((eval_backbone, mr))
    out = []
    for key, values in sorted(groups.items()):
        mrs = [v for _, v in values]
        item = {
            "method": key[0],
            "dataset": key[1],
            "budget_tag": key[2],
            "budget_type": key[3],
            "budget_value": key[4],
            "upstream_backbone": key[5],
            "text_encoder": key[6],
            "num_architectures": len(set(name for name, _ in values)),
            "eval_backbones": " ".join(sorted(set(name for name, _ in values))),
            "mean_recall_mean": statistics.mean(mrs),
            "test_accuracy_mean": statistics.mean(mrs),
            "arch_std": statistics.pstdev(mrs) if len(mrs) > 1 else 0.0,
            "arch_max_drop": max(mrs) - min(mrs),
        }
        for name, mr in values:
            item[f"mr_{name}"] = mr
        out.append(item)
    return out


def build_energy(repblend_report_dir, detail_rows):
    rows = []
    rep_energy = read_csv(Path(repblend_report_dir) / "energy_efficiency.csv")
    for row in rep_energy:
        row = dict(row)
        row["method"] = "repblend"
        row["upstream_backbone"] = upstream
        row["text_encoder"] = text_encoder
        rows.append(row)

    selection_items = [r for r in detail_rows if r.get("method") == "lors" and str(r.get("stage", "")).startswith("selection")]
    eval_items = [r for r in detail_rows if r.get("method") == "lors" and r.get("eval_backbone")]
    selection_energy = sum(to_float(r.get("energy_Wh")) or to_float(r.get("total_energy_Wh")) or 0.0 for r in selection_items)
    selection_seconds = sum(to_float(r.get("seconds")) or 0.0 for r in selection_items)
    for item in eval_items:
        mr = to_float(item.get("mean_recall"))
        rows.append(
            {
                "method": "lors",
                "dataset": item.get("dataset", dataset),
                "budget_tag": ratio_tag(ratio),
                "budget_type": "ratio",
                "budget_value": f"{float(ratio):.6f}",
                "eval_backbone": item.get("eval_backbone", ""),
                "selection_time_seconds": selection_seconds if selection_seconds else "",
                "training_time_seconds": "",
                "total_time_seconds": selection_seconds if selection_seconds else "",
                "selection_energy_Wh": selection_energy if selection_energy else "",
                "training_energy_Wh": "",
                "total_energy_Wh": selection_energy if selection_energy else "",
                "mean_recall": "" if mr is None else mr,
                "test_accuracy": "" if mr is None else mr,
                "test_accuracy_per_Wh": "" if not selection_energy or mr is None else mr / selection_energy,
                "upstream_backbone": upstream,
                "text_encoder": text_encoder,
                "note": "LoRS energy is reused/manual; eval energy is not re-measured by run_lors_ratio_crossarch.sh.",
            }
        )
    return rows


report_dir = Path(sys.argv[1])
repblend_report = Path(sys.argv[2])
lors_root = Path(sys.argv[3])
dataset = sys.argv[4]
ratio = sys.argv[5]
upstream = sys.argv[6]
text_encoder = sys.argv[7]

lors_csv = latest_lors_csv(lors_root)
detail = repblend_rows(repblend_report) + lors_rows(lors_csv)
arch = build_architecture(detail)
energy = build_energy(repblend_report, detail)

fields_detail = [
    "method", "dataset", "budget_tag", "budget_type", "budget_value", "upstream_backbone", "text_encoder", "eval_backbone",
    "mean_recall", "i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10",
    "test_accuracy", "stage", "seconds", "gpu_count", "gpu_hours", "gpu_energy_Wh", "cpu_energy_Wh", "energy_Wh",
    "selection_time_seconds", "training_time_seconds", "total_time_seconds",
    "selection_energy_Wh", "training_energy_Wh", "total_energy_Wh",
    "checkpoint_path", "measurement_path", "metrics_path", "evaluate_log", "source_table", "note",
]
write_csv(report_dir / "combined_detail.csv", detail, fields_detail)
write_csv(report_dir / "combined_architecture_bias.csv", arch)
write_csv(report_dir / "combined_energy_efficiency.csv", energy)

with (report_dir / "inputs.json").open("w", encoding="utf-8") as handle:
    json.dump(
        {
            "repblend_report_dir": str(repblend_report),
            "lors_report_root": str(lors_root),
            "lors_csv": "" if lors_csv is None else str(lors_csv),
            "dataset": dataset,
            "ratio": ratio,
            "upstream_backbone": upstream,
            "text_encoder": text_encoder,
        },
        handle,
        indent=2,
        ensure_ascii=False,
    )

print(f"saved combined detail: {report_dir / 'combined_detail.csv'}")
print(f"saved combined architecture: {report_dir / 'combined_architecture_bias.csv'}")
print(f"saved combined energy: {report_dir / 'combined_energy_efficiency.csv'}")
PY

stage_log "Clean 3% architecture + energy experiment done"
stage_log "  report_dir=${REPORT_DIR}"
stage_log "  combined_architecture=${REPORT_DIR}/combined_architecture_bias.csv"
stage_log "  combined_energy=${REPORT_DIR}/combined_energy_efficiency.csv"
stage_log "  combined_detail=${REPORT_DIR}/combined_detail.csv"
