#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASETS="${FULL_RETRIEVAL_DATASETS:-flickr coco}"
BACKBONE="${FULL_RETRIEVAL_BACKBONE:-nfnet}"
TEXT_ENCODER="${FULL_RETRIEVAL_TEXT_ENCODER:-bert}"
SEEDS="${FULL_RETRIEVAL_SEEDS:-0}"
FULL_RETRIEVAL_INDEX_MODE="${FULL_RETRIEVAL_INDEX_MODE:-image_unique}"
OUTPUT_ROOT="${FULL_RETRIEVAL_OUTPUT_ROOT:-artifacts/full_dataset_retrieval_nfnet_bert_upper_bound}"
INDICES_ROOT="${FULL_RETRIEVAL_INDICES_ROOT:-${OUTPUT_ROOT}/selected_indices_full}"
TRAIN_ROOT="${FULL_RETRIEVAL_TRAIN_ROOT:-${OUTPUT_ROOT}/subset_train_full}"
REPORT_ROOT_FULL="${FULL_RETRIEVAL_REPORT_ROOT:-${OUTPUT_ROOT}/reports}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
REPORT_DIR="${REPORT_ROOT_FULL}/full_dataset_nfnet_bert_${RUN_TIMESTAMP}"
mkdir -p "${REPORT_DIR}" "${INDICES_ROOT}" "${TRAIN_ROOT}"

BATCH_TRAIN="${FULL_RETRIEVAL_BATCH_TRAIN:-${BATCH_SIZE_TRAIN:-32}}"
BATCH_TEST="${FULL_RETRIEVAL_BATCH_TEST:-${BATCH_SIZE_TEST:-128}}"
TEXT_BATCH_SIZE="${FULL_RETRIEVAL_TEXT_BATCH_SIZE:-${TEXT_BATCH_SIZE}}"
FULL_EPOCHS="${FULL_RETRIEVAL_EPOCHS:-50}"
FULL_EVAL_INTERVAL="${FULL_RETRIEVAL_EVAL_INTERVAL:-1}"
FULL_LR_IMG="${FULL_RETRIEVAL_LR_IMG:-0.001}"
FULL_LR_TXT="${FULL_RETRIEVAL_LR_TXT:-0.05}"
FULL_WEIGHT_DECAY="${FULL_RETRIEVAL_WEIGHT_DECAY:-5e-4}"
FULL_LR_DECAY_GAMMA="${FULL_RETRIEVAL_LR_DECAY_GAMMA:-0.1}"
FULL_IMAGE_TRAINABLE="${FULL_RETRIEVAL_IMAGE_TRAINABLE:-true}"
FULL_TEXT_TRAINABLE="${FULL_RETRIEVAL_TEXT_TRAINABLE:-false}"

MODEL_TAG="$(sanitize_component "${BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
SUBSET_TAG="${FULL_RETRIEVAL_SUBSET_TAG:-full_${FULL_RETRIEVAL_INDEX_MODE}_${MODEL_TAG}_upper_bound}"

if [[ "${FULL_RETRIEVAL_INDEX_MODE}" != "image_unique" && "${FULL_RETRIEVAL_INDEX_MODE}" != "pair_all" ]]; then
  echo "Unsupported FULL_RETRIEVAL_INDEX_MODE=${FULL_RETRIEVAL_INDEX_MODE}; expected image_unique or pair_all" >&2
  exit 1
fi

generate_full_indices() {
  local dataset="$1"
  local image_root="$2"
  local output_path="$3"
  local index_mode="$4"
  python - "${dataset}" "${image_root}" "${ANN_ROOT}" "${output_path}" "${index_mode}" <<'PY'
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from src.sklearn_compat import install_sklearn_metrics_stub_if_broken

install_sklearn_metrics_stub_if_broken()

from data import create_dataset

dataset, image_root, ann_root, output_path, index_mode = sys.argv[1:6]
args = SimpleNamespace(
    dataset=dataset,
    image_root=image_root,
    ann_root=ann_root,
    image_size=224,
    no_aug=True,
    return_sample_idx=True,
)
train_dataset, _, _ = create_dataset(args)
if index_mode == "pair_all":
    indices = list(range(len(train_dataset)))
elif index_mode == "image_unique":
    indices = []
    seen = set()
    for sample_idx in range(len(train_dataset)):
        meta = train_dataset.get_pair_metadata(sample_idx)
        key = meta.get("raw_image_id", meta.get("image", meta.get("img_id")))
        if key in seen:
            continue
        seen.add(key)
        indices.append(sample_idx)
else:
    raise ValueError(f"Unsupported index_mode: {index_mode}")
path = Path(output_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(
        {
            "selected_indices": indices,
            "selection_mode": index_mode,
            "num_train_pairs": len(train_dataset),
            "num_selected_records": len(indices),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(len(indices))
PY
}

stage_log "Full-dataset retrieval upper-bound start: datasets=${DATASETS} model=${MODEL_TAG} seeds=${SEEDS}"
stage_log "  index_mode=${FULL_RETRIEVAL_INDEX_MODE} subset_tag=${SUBSET_TAG}"
stage_log "  train config: epochs=${FULL_EPOCHS} batch_train=${BATCH_TRAIN} batch_test=${BATCH_TEST} lr_img=${FULL_LR_IMG} lr_txt=${FULL_LR_TXT} image_trainable=${FULL_IMAGE_TRAINABLE} text_trainable=${FULL_TEXT_TRAINABLE}"
stage_log "  train_root=${TRAIN_ROOT}"
stage_log "  report_dir=${REPORT_DIR}"

for dataset in ${DATASETS}; do
  image_root="$(get_image_root "${dataset}")"
  full_indices_path="${INDICES_ROOT}/${dataset}/train/${MODEL_TAG}/${FULL_RETRIEVAL_INDEX_MODE}/seed_0/selected_indices.json"
  if [[ -f "${full_indices_path}" ]]; then
    subset_size="$(python - "${full_indices_path}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
indices = payload["selected_indices"] if isinstance(payload, dict) else payload
print(len(indices))
PY
)"
    stage_log "Skip full indices: ${dataset} existing ${full_indices_path} size=${subset_size}"
  else
    stage_log "Generate full indices: dataset=${dataset} mode=${FULL_RETRIEVAL_INDEX_MODE}"
    subset_size="$(generate_full_indices "${dataset}" "${image_root}" "${full_indices_path}" "${FULL_RETRIEVAL_INDEX_MODE}")"
    stage_log "Full indices ready: dataset=${dataset} mode=${FULL_RETRIEVAL_INDEX_MODE} size=${subset_size}"
  fi

  for seed in ${SEEDS}; do
    metrics_path="${TRAIN_ROOT}/${dataset}/${MODEL_TAG}/size_$(printf '%04d' "${subset_size}")/${SUBSET_TAG}/seed_${seed}/metrics.json"
    if [[ -f "${metrics_path}" ]]; then
      stage_log "Skip full train/eval: dataset=${dataset} seed=${seed} metrics=${metrics_path}"
      continue
    fi

    stage_log "Full train/eval start: dataset=${dataset} size=${subset_size} seed=${seed}"
    train_extra_args=()
    if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
      train_extra_args+=(--no_aug)
    fi
    if [[ "${ENABLE_IMAGE_ENCODER_DATA_PARALLEL}" == "1" ]]; then
      train_extra_args+=(--enable_image_encoder_data_parallel)
    fi
    if [[ -n "${IMAGE_ENCODER_DATA_PARALLEL_DEVICE_IDS}" ]]; then
      train_extra_args+=(--image_encoder_data_parallel_device_ids "${IMAGE_ENCODER_DATA_PARALLEL_DEVICE_IDS}")
    fi

    python "${PROJECT_ROOT}/run_subset_train.py" \
      --dataset "${dataset}" \
      --image_root "${image_root}" \
      --ann_root "${ANN_ROOT}" \
      --selected_indices_path "${full_indices_path}" \
      --subset_size "${subset_size}" \
      --subset_tag "${SUBSET_TAG}" \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --output_root "${TRAIN_ROOT}" \
      --batch_size_train "${BATCH_TRAIN}" \
      --batch_size_test "${BATCH_TEST}" \
      --text_batch_size "${TEXT_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${FULL_EPOCHS}" \
      --eval_interval "${FULL_EVAL_INTERVAL}" \
      --lr_teacher_img "${FULL_LR_IMG}" \
      --lr_teacher_txt "${FULL_LR_TXT}" \
      --weight_decay "${FULL_WEIGHT_DECAY}" \
      --lr_decay_gamma "${FULL_LR_DECAY_GAMMA}" \
      --image_trainable "${FULL_IMAGE_TRAINABLE}" \
      --text_trainable "${FULL_TEXT_TRAINABLE}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      "${train_extra_args[@]}" \
      > "${REPORT_DIR}/${dataset}_seed${seed}_full_train.log" 2>&1
    stage_log "Full train/eval done: dataset=${dataset} seed=${seed}"
  done
done

stage_log "Aggregate full-dataset retrieval table"
python - "${TRAIN_ROOT}" "${REPORT_DIR}" "${MODEL_TAG}" "${SUBSET_TAG}" "${DATASETS}" "${SEEDS}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

train_root = Path(sys.argv[1])
report_dir = Path(sys.argv[2])
model_tag = sys.argv[3]
subset_tag = sys.argv[4]
datasets = [item for item in sys.argv[5].split() if item.strip()]
seeds = [int(item) for item in sys.argv[6].split() if item.strip()]

metric_keys = ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall"]
raw_rows = []
missing = []

for dataset in datasets:
    model_dir = train_root / dataset / model_tag
    size_dirs = sorted(model_dir.glob("size_*")) if model_dir.exists() else []
    for seed in seeds:
        candidates = [path / subset_tag / f"seed_{seed}" / "metrics.json" for path in size_dirs]
        candidates = [path for path in candidates if path.exists()]
        if not candidates:
            missing.append(str(model_dir / "size_*/" / subset_tag / f"seed_{seed}" / "metrics.json"))
            continue
        metrics_path = max(candidates, key=lambda path: path.stat().st_mtime)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "dataset": dataset,
            "model_tag": model_tag,
            "subset_tag": subset_tag,
            "budget_tag": metrics_path.parents[2].name,
            "subset_size": int(payload.get("subset_size", 0)),
            "seed": int(payload.get("seed", seed)),
            "best_epoch": int(payload.get("best_epoch", -1)),
            "metrics_path": str(metrics_path),
        }
        for key in metric_keys:
            row[key] = float(payload[key])
        raw_rows.append(row)

def safe_std(values):
    return 0.0 if len(values) <= 1 else float(statistics.stdev(values))

summary_rows = []
groups = {}
for row in raw_rows:
    key = (row["dataset"], row["model_tag"], row["subset_tag"], row["budget_tag"], row["subset_size"])
    groups.setdefault(key, []).append(row)
for key, rows in sorted(groups.items()):
    dataset, model_tag, subset_tag, budget_tag, subset_size = key
    summary = {
        "dataset": dataset,
        "model_tag": model_tag,
        "subset_tag": subset_tag,
        "budget_tag": budget_tag,
        "subset_size": subset_size,
        "num_runs": len(rows),
    }
    for metric in metric_keys:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = float(sum(values) / len(values))
        summary[f"{metric}_std"] = safe_std(values)
    summary_rows.append(summary)

raw_fields = ["dataset", "model_tag", "subset_tag", "budget_tag", "subset_size", "seed", "best_epoch", *metric_keys, "metrics_path"]
summary_fields = [
    "dataset", "model_tag", "subset_tag", "budget_tag", "subset_size", "num_runs",
    *[f"{metric}_{suffix}" for metric in metric_keys for suffix in ("mean", "std")],
]

def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

write_csv(report_dir / "full_dataset_retrieval_raw.csv", raw_rows, raw_fields)
write_csv(report_dir / "full_dataset_retrieval_summary.csv", summary_rows, summary_fields)
(report_dir / "missing_metrics.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

md_fields = ["dataset", "subset_size", "num_runs", "i2t_r1_mean", "i2t_r5_mean", "i2t_r10_mean", "t2i_r1_mean", "t2i_r5_mean", "t2i_r10_mean", "mean_recall_mean"]
lines = ["| " + " | ".join(md_fields) + " |", "| " + " | ".join(["---"] * len(md_fields)) + " |"]
for row in summary_rows:
    values = []
    for field in md_fields:
        value = row.get(field, "")
        if isinstance(value, float):
            value = f"{value:.4f}"
        values.append(str(value))
    lines.append("| " + " | ".join(values) + " |")
(report_dir / "full_dataset_retrieval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"saved raw csv: {report_dir / 'full_dataset_retrieval_raw.csv'}")
print(f"saved summary csv: {report_dir / 'full_dataset_retrieval_summary.csv'}")
print(f"saved summary md: {report_dir / 'full_dataset_retrieval_summary.md'}")
print(f"collected runs: {len(raw_rows)}")
PY

stage_log "Full-dataset retrieval upper-bound done"
stage_log "  report_dir=${REPORT_DIR}"
stage_log "  summary_csv=${REPORT_DIR}/full_dataset_retrieval_summary.csv"
stage_log "  summary_md=${REPORT_DIR}/full_dataset_retrieval_summary.md"
