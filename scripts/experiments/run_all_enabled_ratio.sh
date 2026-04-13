#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASET="${ALL_ENABLED_RATIO_DATASET:-flickr}"
SEED="${ALL_ENABLED_RATIO_SEED:-0}"
BACKBONE="${ALL_ENABLED_RATIO_BACKBONE:-nfnet}"
TEXT_ENCODER="${ALL_ENABLED_RATIO_TEXT_ENCODER:-bert}"
RATIOS_STR="${ALL_ENABLED_RATIO_RATIOS:-0.1 0.2 0.3}"
TRAIN_BACKBONES_STR="${ALL_ENABLED_RATIO_TRAIN_BACKBONES:-nfnet resnet50 vit_b16}"
VARIANT="${ALL_ENABLED_RATIO_VARIANT:-all_enabled_ratio}"
MODEL_TAG="$(sanitize_component "${BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
FUSION_TAG="k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_IMAGE}")_a$(sanitize_component "${ALPHA}")"
IMAGE_ROOT="$(get_image_root "${DATASET}")"

# Reuse existing all-enabled cross-modal artifacts when available; ratios only affect selection/train.
ALL_ENABLED_RATIO_CROSS_ROOT="${ALL_ENABLED_RATIO_CROSS_ROOT:-artifacts/cross_modal_topology_ablation}"
ALL_ENABLED_RATIO_SELECTION_ROOT="${ALL_ENABLED_RATIO_SELECTION_ROOT:-artifacts/subset_selection_ratio}"
ALL_ENABLED_RATIO_TRAIN_ROOT="${ALL_ENABLED_RATIO_TRAIN_ROOT:-artifacts/subset_train_ratio}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/all_enabled_ratio_${DATASET}_${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}" "${REPORT_ROOT}"
REPORT_CSV_PATH="${REPORT_ROOT}/all_enabled_ratio_${DATASET}_seed${SEED}_${TIMESTAMP}.csv"
REPORT_MISSING_PATH="${REPORT_ROOT}/all_enabled_ratio_${DATASET}_seed${SEED}_${TIMESTAMP}_missing.txt"

format_ratio_tag_local() {
  local ratio="$1"
  python - <<PY
ratio = float("${ratio}")
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
}

run_cross_stage() {
  local cross_root="${ALL_ENABLED_RATIO_CROSS_ROOT}/all_enabled"
  local summary_path="${cross_root}/${DATASET}/train/${MODEL_TAG}/${FUSION_TAG}/summary.json"
  local log_path="${RUN_LOG_DIR}/cross.log"

  if [[ -f "${summary_path}" ]]; then
    stage_log "Skip cross-modal (all-enabled): ${summary_path} already exists"
    return 0
  fi

  if [[ "${ENABLE_SELECTION_EFFICIENCY_BENCHMARK}" == "1" ]]; then
    stage_log "Skip standalone cross-modal (all-enabled): benchmark mode will wrap cross+selection together"
    return 0
  fi

  local wavelet_extra_args=()
  if [[ -n "${WAVELET_FUSION_WEIGHT_A_SCALES}" ]]; then
    wavelet_extra_args+=(--wavelet_fusion_weight_a_scales "${WAVELET_FUSION_WEIGHT_A_SCALES}")
  fi
  if [[ -n "${WAVELET_FUSION_WEIGHT_B_SCALES}" ]]; then
    wavelet_extra_args+=(--wavelet_fusion_weight_b_scales "${WAVELET_FUSION_WEIGHT_B_SCALES}")
  fi

  stage_log "Cross-modal start: variant=all-enabled correction=bidirectional fusion=confidence_aware"
  python "${PROJECT_ROOT}/run_cross_modal_topology.py" \
    --dataset "${DATASET}" \
    --split train \
    --image_encoder "${BACKBONE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --topology_root "${TOPOLOGY_ROOT}" \
    --output_root "${cross_root}" \
    --metric "${TOPOLOGY_METRIC_IMAGE}" \
    --image_metric "${TOPOLOGY_METRIC_IMAGE}" \
    --text_metric "${TOPOLOGY_METRIC_TEXT}" \
    --k "${K_NEIGHBORS}" \
    --alpha "${ALPHA}" \
    --correction_mode bidirectional \
    --correction_score_mode "${CORRECTION_SCORE_MODE}" \
    --collapse_score_mode "${COLLAPSE_SCORE_MODE}" \
    --collapse_score_weight_edge "${COLLAPSE_SCORE_WEIGHT_EDGE}" \
    --collapse_score_weight_a2b "${COLLAPSE_SCORE_WEIGHT_A2B}" \
    --collapse_score_weight_b2a "${COLLAPSE_SCORE_WEIGHT_B2A}" \
    --collapse_score_weight_nbr2nbr "${COLLAPSE_SCORE_WEIGHT_NBR2NBR}" \
    --collapse_neighbor_topk "${COLLAPSE_NEIGHBOR_TOPK}" \
    --fusion_domain_mode "${FUSION_DOMAIN_MODE}" \
    --fusion_mode confidence_aware \
    --wavelet_fusion_scales "${WAVELET_FUSION_SCALES}" \
    --wavelet_fusion_impl "${WAVELET_FUSION_IMPL}" \
    --wavelet_fusion_probe_mode "${WAVELET_FUSION_PROBE_MODE}" \
    --wavelet_fusion_probe_dim "${WAVELET_FUSION_PROBE_DIM}" \
    --wavelet_fusion_weight_mode "${WAVELET_FUSION_WEIGHT_MODE}" \
    --wavelet_latent_lambda_sparse "${WAVELET_LATENT_LAMBDA_SPARSE}" \
    --wavelet_latent_lambda_sym "${WAVELET_LATENT_LAMBDA_SYM}" \
    --wavelet_latent_lambda_nonneg "${WAVELET_LATENT_LAMBDA_NONNEG}" \
    --wavelet_latent_reconstruction_mode "${WAVELET_LATENT_RECONSTRUCTION_MODE}" \
    --wavelet_latent_postprocess_topk "${WAVELET_LATENT_POSTPROCESS_TOPK}" \
    --wavelet_latent_postprocess_threshold "${WAVELET_LATENT_POSTPROCESS_THRESHOLD}" \
    --tau_g 0.5 \
    --correction_eps 1e-8 \
    --lambda_f 1.0 \
    --mu_f 1.0 \
    --fusion_eps 1e-8 \
    --num_eigs "${CROSS_MODAL_NUM_EIGS}" \
    --spectral_embedding_dim "${CROSS_MODAL_EMBED_DIM}" \
    --spectrum_solver_mode normalized_adjacency_largest \
    "${wavelet_extra_args[@]}" \
    > "${log_path}" 2>&1
}

run_selection_and_train() {
  local ratio="$1"
  local ratio_tag
  ratio_tag="$(format_ratio_tag_local "${ratio}")"
  local cross_root="${ALL_ENABLED_RATIO_CROSS_ROOT}/all_enabled"
  local selection_root="${ALL_ENABLED_RATIO_SELECTION_ROOT}/${VARIANT}"
  local selection_method_tag="proxy_opt_lsrc"
  local selected_indices_path="${selection_root}/${DATASET}/train/${MODEL_TAG}/${ratio_tag}/${selection_method_tag}/seed_${SEED}/selected_indices.json"
  local select_log="${RUN_LOG_DIR}/${ratio_tag}_select.log"
  local benchmark_output_dir="${SELECTION_EFFICIENCY_ROOT}/all_enabled_ratio/${DATASET}/${MODEL_TAG}/${ratio_tag}/seed_${SEED}"
  local benchmark_summary_path="${benchmark_output_dir}/selection_efficiency_summary.json"

  if [[ ! -f "${selected_indices_path}" || ( "${ENABLE_SELECTION_EFFICIENCY_BENCHMARK}" == "1" && ! -f "${benchmark_summary_path}" ) ]]; then
    stage_log "Selection start: variant=${VARIANT} ratio=${ratio}"
    local selection_extra_args=(
      --enable_lsrc
      --proxy_projection_dim "${PROXY_PROJECTION_DIM}"
      --proxy_num_frequencies "${PROXY_NUM_FREQUENCIES}"
      --proxy_num_steps "${PROXY_NUM_STEPS}"
      --proxy_target_batch_size "${PROXY_TARGET_BATCH_SIZE}"
      --proxy_batch_size "${PROXY_BATCH_SIZE}"
      --lsrc_k 32
      --lsrc_tau_r 1.0
      --lsrc_tau_c 1.0
      --lsrc_eta 0.5
      --lsrc_beta 0.5
      --lambda_lsrc_cov 0.1
      --lambda_lsrc_rel 0.05
      --lsrc_eps 1e-8
      --lsrc_batch_size "${PROXY_LSRC_BATCH_SIZE}"
    )

    if [[ "${ENABLE_SELECTION_EFFICIENCY_BENCHMARK}" == "1" ]]; then
      local benchmark_cmd=(
        python "${PROJECT_ROOT}/run_selection_efficiency_benchmark.py"
        --dataset "${DATASET}"
        --split train
        --image_encoder "${BACKBONE}"
        --text_encoder "${TEXT_ENCODER}"
        --feature_cache_root "${FEATURE_CACHE_ROOT}"
        --topology_root "${TOPOLOGY_ROOT}"
        --cross_output_root "${cross_root}"
        --selection_output_root "${selection_root}"
        --metric "${TOPOLOGY_METRIC_IMAGE}"
        --image_metric "${TOPOLOGY_METRIC_IMAGE}"
        --text_metric "${TOPOLOGY_METRIC_TEXT}"
        --k "${K_NEIGHBORS}"
        --alpha "${ALPHA}"
        --correction_mode bidirectional
        --correction_score_mode "${CORRECTION_SCORE_MODE}"
        --collapse_score_mode "${COLLAPSE_SCORE_MODE}"
        --collapse_score_weight_edge "${COLLAPSE_SCORE_WEIGHT_EDGE}"
        --collapse_score_weight_a2b "${COLLAPSE_SCORE_WEIGHT_A2B}"
        --collapse_score_weight_b2a "${COLLAPSE_SCORE_WEIGHT_B2A}"
        --collapse_score_weight_nbr2nbr "${COLLAPSE_SCORE_WEIGHT_NBR2NBR}"
        --collapse_neighbor_topk "${COLLAPSE_NEIGHBOR_TOPK}"
        --fusion_domain_mode "${FUSION_DOMAIN_MODE}"
        --fusion_mode confidence_aware
        --wavelet_fusion_scales "${WAVELET_FUSION_SCALES}"
        --wavelet_fusion_impl "${WAVELET_FUSION_IMPL}"
        --wavelet_fusion_probe_mode "${WAVELET_FUSION_PROBE_MODE}"
        --wavelet_fusion_probe_dim "${WAVELET_FUSION_PROBE_DIM}"
        --wavelet_fusion_weight_mode "${WAVELET_FUSION_WEIGHT_MODE}"
        --wavelet_latent_lambda_sparse "${WAVELET_LATENT_LAMBDA_SPARSE}"
        --wavelet_latent_lambda_sym "${WAVELET_LATENT_LAMBDA_SYM}"
        --wavelet_latent_lambda_nonneg "${WAVELET_LATENT_LAMBDA_NONNEG}"
        --wavelet_latent_reconstruction_mode "${WAVELET_LATENT_RECONSTRUCTION_MODE}"
        --wavelet_latent_postprocess_topk "${WAVELET_LATENT_POSTPROCESS_TOPK}"
        --wavelet_latent_postprocess_threshold "${WAVELET_LATENT_POSTPROCESS_THRESHOLD}"
        --tau_g 0.5
        --correction_eps 1e-8
        --lambda_f 1.0
        --mu_f 1.0
        --fusion_eps 1e-8
        --num_eigs "${CROSS_MODAL_NUM_EIGS}"
        --spectral_embedding_dim "${CROSS_MODAL_EMBED_DIM}"
        --spectrum_solver_mode normalized_adjacency_largest
        --budget_ratio "${ratio}"
        --selection_method proxy_opt
        --reference_embedding_mode "${SUBSET_REFERENCE_EMBEDDING_MODE}"
        --spectral_weight "${SUBSET_SPECTRAL_WEIGHT}"
        --random_state "${SEED}"
        --device "${DEVICE}"
        --proxy_projection_dim "${PROXY_PROJECTION_DIM}"
        --geometry_weight "${PROXY_GEOMETRY_WEIGHT}"
        --matching_cost_mode "${MATCHING_COST_MODE}"
        --proxy_objective_mode "${PROXY_OBJECTIVE_MODE}"
        --proxy_num_frequencies "${PROXY_NUM_FREQUENCIES}"
        --proxy_num_steps "${PROXY_NUM_STEPS}"
        --proxy_target_batch_size "${PROXY_TARGET_BATCH_SIZE}"
        --proxy_batch_size "${PROXY_BATCH_SIZE}"
        --lambda_div "${PROXY_LAMBDA_DIV}"
        --lambda_match "${PROXY_LAMBDA_MATCH}"
        --lambda_graph "${PROXY_LAMBDA_GRAPH}"
        --lambda_phase "${PROXY_LAMBDA_PHASE}"
        --num_freq_pool "${PROXY_NUM_FREQ_POOL}"
        --tau_min "${PROXY_TAU_MIN}"
        --tau_max "${PROXY_TAU_MAX}"
        --variant_name "${VARIANT}"
        --benchmark_output_dir "${benchmark_output_dir}"
        --enable_selection_efficiency_benchmark
        --energy_backend "${SELECTION_EFFICIENCY_BACKEND}"
        --poll_interval_ms "${SELECTION_EFFICIENCY_POLL_INTERVAL_MS}"
      )
      if [[ -n "${SELECTION_EFFICIENCY_BASELINE_SUMMARY}" ]]; then
        benchmark_cmd+=(--baseline_summary "${SELECTION_EFFICIENCY_BASELINE_SUMMARY}")
      fi
      if [[ "${PROXY_USE_PDAS}" == "1" ]]; then
        benchmark_cmd+=(--use_pdas)
      fi
      if [[ "${PROXY_USE_PDCFD}" == "1" ]]; then
        benchmark_cmd+=(--use_pdcfd)
      fi
      if [[ "${PROXY_USE_DPP}" == "1" ]]; then
        benchmark_cmd+=(--use_dpp)
      fi
      if [[ -n "${WAVELET_FUSION_WEIGHT_A_SCALES}" ]]; then
        benchmark_cmd+=(--wavelet_fusion_weight_a_scales "${WAVELET_FUSION_WEIGHT_A_SCALES}")
      fi
      if [[ -n "${WAVELET_FUSION_WEIGHT_B_SCALES}" ]]; then
        benchmark_cmd+=(--wavelet_fusion_weight_b_scales "${WAVELET_FUSION_WEIGHT_B_SCALES}")
      fi
      "${benchmark_cmd[@]}" "${selection_extra_args[@]}" > "${select_log}" 2>&1
    else
      python "${PROJECT_ROOT}/run_subset_selection.py" \
        --dataset "${DATASET}" \
        --split train \
        --image_encoder "${BACKBONE}" \
        --text_encoder "${TEXT_ENCODER}" \
        --feature_cache_root "${FEATURE_CACHE_ROOT}" \
        --cross_modal_root "${cross_root}" \
        --output_root "${selection_root}" \
        --metric "${TOPOLOGY_METRIC_IMAGE}" \
        --k "${K_NEIGHBORS}" \
        --alpha "${ALPHA}" \
        --budget_ratio "${ratio}" \
        --selection_method proxy_opt \
        --reference_embedding_mode "${SUBSET_REFERENCE_EMBEDDING_MODE}" \
        --spectral_weight "${SUBSET_SPECTRAL_WEIGHT}" \
        --random_state "${SEED}" \
        --device "${DEVICE}" \
        --geometry_weight "${PROXY_GEOMETRY_WEIGHT}" \
        --matching_cost_mode "${MATCHING_COST_MODE}" \
        --proxy_objective_mode "${PROXY_OBJECTIVE_MODE}" \
        --lambda_div "${PROXY_LAMBDA_DIV}" \
        --lambda_match "${PROXY_LAMBDA_MATCH}" \
        --lambda_graph "${PROXY_LAMBDA_GRAPH}" \
        --lambda_phase "${PROXY_LAMBDA_PHASE}" \
        --num_freq_pool "${PROXY_NUM_FREQ_POOL}" \
        --tau_min "${PROXY_TAU_MIN}" \
        --tau_max "${PROXY_TAU_MAX}" \
        $( [[ "${PROXY_USE_PDAS}" == "1" ]] && echo --use_pdas ) \
        $( [[ "${PROXY_USE_PDCFD}" == "1" ]] && echo --use_pdcfd ) \
        $( [[ "${PROXY_USE_DPP}" == "1" ]] && echo --use_dpp ) \
        "${selection_extra_args[@]}" \
        > "${select_log}" 2>&1
    fi
  else
    stage_log "Skip selection (${ratio_tag}): ${selected_indices_path} already exists"
  fi

  local train_backbone
  for train_backbone in ${TRAIN_BACKBONES_STR}; do
    local train_model_tag
    train_model_tag="$(sanitize_component "${train_backbone}")_$(sanitize_component "${TEXT_ENCODER}")"
    local train_root="${ALL_ENABLED_RATIO_TRAIN_ROOT}/${VARIANT}"
    local metrics_path="${train_root}/${DATASET}/${train_model_tag}/${ratio_tag}/${VARIANT}/seed_${SEED}/metrics.json"
    local train_log="${RUN_LOG_DIR}/${ratio_tag}_train_${train_model_tag}.log"

    if [[ -f "${metrics_path}" ]]; then
      stage_log "Skip train (${ratio_tag}, ${train_backbone}): ${metrics_path} already exists"
      continue
    fi

    stage_log "Train start: variant=${VARIANT} ratio=${ratio} backbone=${train_backbone}"
    local train_extra_args=()
    if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
      train_extra_args+=(--no_aug)
    fi

    python "${PROJECT_ROOT}/run_subset_train.py" \
      --dataset "${DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --selected_indices_path "${selected_indices_path}" \
      --subset_ratio "${ratio}" \
      --subset_tag "${VARIANT}" \
      --image_encoder "${train_backbone}" \
      --text_encoder "${TEXT_ENCODER}" \
      --output_root "${train_root}" \
      --batch_size_train "${BATCH_TRAIN}" \
      --batch_size_test "${BATCH_TEST}" \
      --text_batch_size "${TEXT_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      "${train_extra_args[@]}" \
      > "${train_log}" 2>&1
  done
}

stage_log "All-enabled ratio pipeline start: dataset=${DATASET} ratios=${RATIOS_STR} seed=${SEED} train_backbones=${TRAIN_BACKBONES_STR}"

run_cross_stage

for ratio in ${RATIOS_STR}; do
  run_selection_and_train "${ratio}"
done

stage_log "Aggregate start: ${REPORT_CSV_PATH}"
python - <<PY
import csv
import json
from pathlib import Path

dataset = "${DATASET}"
seed = int("${SEED}")
text_encoder = "${TEXT_ENCODER}"
ratios = "${RATIOS_STR}".split()
train_backbones = "${TRAIN_BACKBONES_STR}".split()
train_root = Path(r"${ALL_ENABLED_RATIO_TRAIN_ROOT}") / "${VARIANT}"
report_csv_path = Path(r"${REPORT_CSV_PATH}")
report_missing_path = Path(r"${REPORT_MISSING_PATH}")


def sanitize_name(name):
    return str(name).replace("\\\\", "-").replace("/", "-").replace(" ", "_")


def ratio_tag(ratio_str):
    return f"ratio_{int(round(float(ratio_str) * 100)):02d}"


rows = []
missing = []

for ratio in ratios:
    budget_tag = ratio_tag(ratio)
    for backbone in train_backbones:
        model_tag = f"{sanitize_name(backbone)}_{sanitize_name(text_encoder)}"
        metrics_path = train_root / dataset / model_tag / budget_tag / "${VARIANT}" / f"seed_{seed}" / "metrics.json"
        row = {
            "dataset": dataset,
            "subset_ratio": float(ratio),
            "ratio_tag": budget_tag,
            "image_encoder": backbone,
            "text_encoder": text_encoder,
            "variant": "${VARIANT}",
            "i2t_r1": None,
            "i2t_r5": None,
            "i2t_r10": None,
            "t2i_r1": None,
            "t2i_r5": None,
            "t2i_r10": None,
            "mean_recall": None,
            "metrics_path": str(metrics_path),
        }
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            row.update({
                "i2t_r1": metrics.get("i2t_r1"),
                "i2t_r5": metrics.get("i2t_r5"),
                "i2t_r10": metrics.get("i2t_r10"),
                "t2i_r1": metrics.get("t2i_r1"),
                "t2i_r5": metrics.get("t2i_r5"),
                "t2i_r10": metrics.get("t2i_r10"),
                "mean_recall": metrics.get("mean_recall"),
            })
        else:
            missing.append(str(metrics_path))
        rows.append(row)

report_csv_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

with open(report_missing_path, "w", encoding="utf-8") as f:
    if missing:
        for item in missing:
            f.write(item + "\\n")
    else:
        f.write("")

print(f"saved csv: {report_csv_path}")
if missing:
    print(f"missing count: {len(missing)}")
    print(f"missing list: {report_missing_path}")
else:
    print("all 9 metrics found")
PY

stage_log "All-enabled ratio pipeline completed"
