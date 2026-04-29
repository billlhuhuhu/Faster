#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SUPP_DATASET="${SUPP_DATASET:-flickr}"
SUPP_METHODS="${SUPP_METHODS:-ours random lors repblend}"
SUPP_BUDGETS="${SUPP_BUDGETS:-100 200 500}"
SUPP_RATIOS="${SUPP_RATIOS:-0.01 0.02 0.03}"
SUPP_SEED="${SUPP_SEED:-0}"
SUPP_EVAL_BACKBONES="${SUPP_EVAL_BACKBONES:-nfnet resnet50 vit_b16}"
SUPP_TEXT_ENCODER="${SUPP_TEXT_ENCODER:-bert}"
SUPP_RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SUPP_ROOT="${SUPP_ROOT:-artifacts/supplemental_arch_energy}"
SUPP_REPORT_DIR="${SUPP_ROOT}/reports/${SUPP_DATASET}_${SUPP_RUN_TIMESTAMP}"
SUPP_LOG_DIR="${SUPP_ROOT}/logs/${SUPP_DATASET}_${SUPP_RUN_TIMESTAMP}"
SUPP_MANIFEST="${SUPP_REPORT_DIR}/manifest.jsonl"
mkdir -p "${SUPP_REPORT_DIR}" "${SUPP_LOG_DIR}"
: > "${SUPP_MANIFEST}"

IMAGE_ROOT="$(get_image_root "${SUPP_DATASET}")"
SOURCE_MODEL_TAG="nfnet_${SUPP_TEXT_ENCODER}"
OURS_SELECTION_ROOT="${OURS_SELECTION_ROOT:-artifacts/subset_selection_dense_sift_bovw}"
RANDOM_SELECTION_ROOT="${RANDOM_SELECTION_ROOT:-artifacts/subset_selection_random_baseline}"
RANDOM_FEATURE_CACHE_ROOT="${RANDOM_FEATURE_CACHE_ROOT:-artifacts/feature_cache_dense_sift_bovw}"
OURS_TRAIN_ROOT="${OURS_TRAIN_ROOT:-${SUPP_ROOT}/subset_train_ours_crossarch}"
RANDOM_TRAIN_ROOT="${RANDOM_TRAIN_ROOT:-${SUPP_ROOT}/subset_train_random_crossarch}"
REPBLEND_RESULTS_CSV="${REPBLEND_RESULTS_CSV:-}"
SUPP_GPU_COUNT="${SUPP_GPU_COUNT:-$(python - <<PY
devices = "${CUDA_VISIBLE_DEVICES:-}".strip()
print(max(len([x for x in devices.split(",") if x.strip()]), 1))
PY
)}"

method_enabled() {
  local name="$1"
  [[ " ${SUPP_METHODS} " == *" ${name} "* ]]
}

ratio_to_tag() {
  local ratio="$1"
  python - "${ratio}" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
}

append_manifest() {
  python - "$SUPP_MANIFEST" "$@" <<'PY'
import json
import sys
path = sys.argv[1]
keys = sys.argv[2::2]
vals = sys.argv[3::2]
item = dict(zip(keys, vals))
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(item, ensure_ascii=False) + "\n")
PY
}

run_timed() {
  local method="$1"
  local stage="$2"
  local budget_type="$3"
  local budget_value="$4"
  local budget_tag="$5"
  local eval_backbone="$6"
  local metrics_path="$7"
  local log_path="$8"
  shift 8
  if [[ -f "${metrics_path}" ]]; then
    stage_log "Skip ${method}/${stage}: existing ${metrics_path}"
    append_manifest method "${method}" dataset "${SUPP_DATASET}" budget_type "${budget_type}" budget_value "${budget_value}" budget_tag "${budget_tag}" eval_backbone "${eval_backbone}" stage "${stage}" seconds "0" gpu_count "${SUPP_GPU_COUNT}" metrics_path "${metrics_path}" log_path "${log_path}" skipped "1"
    return 0
  fi
  stage_log "Run ${method}/${stage}: ${budget_tag} backbone=${eval_backbone}"
  local start end elapsed
  start="$(date +%s)"
  "$@" > "${log_path}" 2>&1
  end="$(date +%s)"
  elapsed="$((end - start))"
  append_manifest method "${method}" dataset "${SUPP_DATASET}" budget_type "${budget_type}" budget_value "${budget_value}" budget_tag "${budget_tag}" eval_backbone "${eval_backbone}" stage "${stage}" seconds "${elapsed}" gpu_count "${SUPP_GPU_COUNT}" metrics_path "${metrics_path}" log_path "${log_path}" skipped "0"
}

run_real_subset_crossarch() {
  local method="$1"
  local selection_root="$2"
  local train_root="$3"
  local budget_type="$4"
  local budget_value="$5"
  local budget_tag="$6"
  local selection_method_tag="$7"
  local selected_indices_path="${selection_root}/${SUPP_DATASET}/train/${SOURCE_MODEL_TAG}/${budget_tag}/${selection_method_tag}/seed_${SUPP_SEED}/selected_indices.json"
  if [[ ! -f "${selected_indices_path}" ]]; then
    if [[ "${method}" == "random" ]]; then
      stage_log "Random selected indices missing; generating ${budget_tag} seed=${SUPP_SEED}"
      random_args=()
      if [[ "${budget_type}" == "size" ]]; then
        random_args+=(--budget_size "${budget_value}")
      else
        random_args+=(--budget_ratio "${budget_value}")
      fi
      python "${PROJECT_ROOT}/run_random_subset_selection.py" \
        --dataset "${SUPP_DATASET}" \
        --split train \
        --image_encoder nfnet \
        --text_encoder "${SUPP_TEXT_ENCODER}" \
        --feature_cache_root "${RANDOM_FEATURE_CACHE_ROOT}" \
        --output_root "${selection_root}" \
        "${random_args[@]}" \
        --selection_method random \
        --random_state "${SUPP_SEED}" \
        > "${SUPP_LOG_DIR}/random_select_${budget_tag}_seed${SUPP_SEED}.log" 2>&1
    fi
  fi
  if [[ ! -f "${selected_indices_path}" ]]; then
    stage_log "Missing ${method} selected indices after generation attempt: ${selected_indices_path}"
    return 0
  fi
  for eval_backbone in ${SUPP_EVAL_BACKBONES}; do
    local eval_model_tag
    eval_model_tag="$(sanitize_component "${eval_backbone}")_${SUPP_TEXT_ENCODER}"
    local metrics_path="${train_root}/${SUPP_DATASET}/${eval_model_tag}/${budget_tag}/${method}_crossarch/seed_${SUPP_SEED}/metrics.json"
    local log_path="${SUPP_LOG_DIR}/${method}_${budget_tag}_${eval_backbone}.log"
    local budget_args=()
    if [[ "${budget_type}" == "size" ]]; then
      budget_args+=(--subset_size "${budget_value}")
    else
      budget_args+=(--subset_ratio "${budget_value}")
    fi
    train_extra=()
    if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
      train_extra+=(--no_aug)
    fi
    run_timed "${method}" "train_eval" "${budget_type}" "${budget_value}" "${budget_tag}" "${eval_backbone}" "${metrics_path}" "${log_path}" \
      python "${PROJECT_ROOT}/run_subset_train.py" \
        --dataset "${SUPP_DATASET}" \
        --image_root "${IMAGE_ROOT}" \
        --ann_root "${ANN_ROOT}" \
        --selected_indices_path "${selected_indices_path}" \
        "${budget_args[@]}" \
        --subset_tag "${method}_crossarch" \
        --image_encoder "${eval_backbone}" \
        --text_encoder "${SUPP_TEXT_ENCODER}" \
        --output_root "${train_root}" \
        --batch_size_train "${BATCH_TRAIN}" \
        --batch_size_test "${BATCH_TEST}" \
        --text_batch_size "${TEXT_BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --epochs "${EPOCHS}" \
        --eval_interval "${EVAL_INTERVAL}" \
        --seed "${SUPP_SEED}" \
        --device "${DEVICE}" \
        "${train_extra[@]}"
  done
}

stage_log "Supplemental arch/energy start: dataset=${SUPP_DATASET} methods=${SUPP_METHODS}"

if method_enabled ours; then
  for budget in ${SUPP_BUDGETS}; do
    run_real_subset_crossarch "ours" "${OURS_SELECTION_ROOT}" "${OURS_TRAIN_ROOT}" "size" "${budget}" "$(format_budget_tag "${budget}")" "proxy_opt_lsrc"
  done
  for ratio in ${SUPP_RATIOS}; do
    run_real_subset_crossarch "ours" "${OURS_SELECTION_ROOT}" "${OURS_TRAIN_ROOT}" "ratio" "${ratio}" "$(ratio_to_tag "${ratio}")" "proxy_opt_lsrc"
  done
fi

if method_enabled random; then
  for budget in ${SUPP_BUDGETS}; do
    run_real_subset_crossarch "random" "${RANDOM_SELECTION_ROOT}" "${RANDOM_TRAIN_ROOT}" "size" "${budget}" "$(format_budget_tag "${budget}")" "random"
  done
  for ratio in ${SUPP_RATIOS}; do
    run_real_subset_crossarch "random" "${RANDOM_SELECTION_ROOT}" "${RANDOM_TRAIN_ROOT}" "ratio" "${ratio}" "$(ratio_to_tag "${ratio}")" "random"
  done
fi

if method_enabled lors; then
  stage_log "LoRS ratio cross-architecture run start"
  local_start="$(date +%s)"
  LORS_RATIO_DATASET="${SUPP_DATASET}" \
  LORS_RATIO_VALUES="${SUPP_RATIOS}" \
  LORS_RATIO_EVAL_BACKBONES="${SUPP_EVAL_BACKBONES}" \
  bash "${SCRIPT_DIR}/run_lors_ratio_crossarch.sh" > "${SUPP_LOG_DIR}/lors_ratio_crossarch.log" 2>&1
  local_end="$(date +%s)"
  latest_lors_csv="$(find "${REPORT_ROOT}" -path "*lors_ratio_crossarch_${SUPP_DATASET}_*/lors_ratio_crossarch.csv" -type f 2>/dev/null | sort | tail -n 1)"
  if [[ -n "${latest_lors_csv}" ]]; then
    python - "${SUPP_MANIFEST}" "${latest_lors_csv}" "$((local_end - local_start))" "${SUPP_GPU_COUNT}" <<'PY'
import csv, json, sys
manifest, csv_path, seconds, gpu_count = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(csv_path, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
seconds_per_row = float(seconds) / max(len(rows), 1)
with open(manifest, "a", encoding="utf-8") as out:
    for row in rows:
        ratio = row.get("ratio", "")
        budget_tag = f"ratio_{int(round(float(ratio) * 100)):02d}" if ratio else ""
        item = {
            "method": "lors",
            "dataset": row.get("dataset", ""),
            "budget_type": "ratio",
            "budget_value": ratio,
            "budget_tag": budget_tag,
            "eval_backbone": row.get("eval_backbone", ""),
            "stage": "distill_eval",
            "seconds": seconds_per_row,
            "gpu_count": gpu_count,
            "source": csv_path,
            "mean_recall": row.get("r_mean", ""),
            "i2t_r1": row.get("txt_r1", ""),
            "i2t_r5": row.get("txt_r5", ""),
            "i2t_r10": row.get("txt_r10", ""),
            "t2i_r1": row.get("img_r1", ""),
            "t2i_r5": row.get("img_r5", ""),
            "t2i_r10": row.get("img_r10", ""),
        }
        out.write(json.dumps(item, ensure_ascii=False) + "\n")
PY
  else
    stage_log "LoRS csv not found; skipped LoRS table rows"
  fi
fi

external_args=()
if method_enabled repblend; then
  if [[ -n "${REPBLEND_RESULTS_CSV}" && -f "${REPBLEND_RESULTS_CSV}" ]]; then
    external_args+=(--external_csv "${REPBLEND_RESULTS_CSV}")
  else
    stage_log "RepBlend code/results not found. Set REPBLEND_RESULTS_CSV=/path/to/repblend_results.csv to include it."
  fi
fi

python "${PROJECT_ROOT}/tools/build_supplemental_arch_energy_tables.py" \
  --manifest_jsonl "${SUPP_MANIFEST}" \
  --output_dir "${SUPP_REPORT_DIR}" \
  "${external_args[@]}"

stage_log "Supplemental tables done:"
stage_log "  architecture_bias=${SUPP_REPORT_DIR}/architecture_bias.csv"
stage_log "  energy_efficiency=${SUPP_REPORT_DIR}/energy_efficiency.csv"
