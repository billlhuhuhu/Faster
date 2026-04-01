#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASET="${ALL_ENABLED_RATIO_DATASET:-flickr}"
SEED="${ALL_ENABLED_RATIO_SEED:-0}"
BACKBONE="${ALL_ENABLED_RATIO_BACKBONE:-nfnet}"
TEXT_ENCODER="${ALL_ENABLED_RATIO_TEXT_ENCODER:-bert}"
RATIOS_STR="${ALL_ENABLED_RATIO_RATIOS:-0.1 0.2 0.3}"
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
    --fusion_mode confidence_aware \
    --tau_g 0.5 \
    --correction_eps 1e-8 \
    --lambda_f 1.0 \
    --mu_f 1.0 \
    --fusion_eps 1e-8 \
    --num_eigs "${CROSS_MODAL_NUM_EIGS}" \
    --spectral_embedding_dim "${CROSS_MODAL_EMBED_DIM}" \
    --spectrum_solver_mode normalized_adjacency_largest \
    > "${log_path}" 2>&1
}

run_selection_and_train() {
  local ratio="$1"
  local ratio_tag
  ratio_tag="$(format_ratio_tag_local "${ratio}")"
  local cross_root="${ALL_ENABLED_RATIO_CROSS_ROOT}/all_enabled"
  local selection_root="${ALL_ENABLED_RATIO_SELECTION_ROOT}/${VARIANT}"
  local train_root="${ALL_ENABLED_RATIO_TRAIN_ROOT}/${VARIANT}"
  local selection_method_tag="proxy_opt_lsrc"
  local selected_indices_path="${selection_root}/${DATASET}/train/${MODEL_TAG}/${ratio_tag}/${selection_method_tag}/seed_${SEED}/selected_indices.json"
  local metrics_path="${train_root}/${DATASET}/${MODEL_TAG}/${ratio_tag}/${VARIANT}/seed_${SEED}/metrics.json"
  local select_log="${RUN_LOG_DIR}/${ratio_tag}_select.log"
  local train_log="${RUN_LOG_DIR}/${ratio_tag}_train.log"
  local benchmark_output_dir="${SELECTION_EFFICIENCY_ROOT}/all_enabled_ratio/${DATASET}/${MODEL_TAG}/${ratio_tag}/seed_${SEED}"
  local benchmark_summary_path="${benchmark_output_dir}/selection_efficiency_summary.json"

  if [[ ! -f "${selected_indices_path}" || ( "${ENABLE_SELECTION_EFFICIENCY_BENCHMARK}" == "1" && ! -f "${benchmark_summary_path}" ) ]]; then
    stage_log "Selection start: variant=${VARIANT} ratio=${ratio}"
    local selection_extra_args=(
      --enable_lsrc
      --lsrc_k 32
      --lsrc_tau_r 1.0
      --lsrc_tau_c 1.0
      --lsrc_eta 0.5
      --lsrc_beta 0.5
      --lambda_lsrc_cov 0.1
      --lambda_lsrc_rel 0.05
      --lsrc_eps 1e-8
      --lsrc_batch_size 4096
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
        --fusion_mode confidence_aware
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
        --geometry_weight "${PROXY_GEOMETRY_WEIGHT}"
        --matching_cost_mode "${MATCHING_COST_MODE}"
        --proxy_objective_mode "${PROXY_OBJECTIVE_MODE}"
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

  if [[ -f "${metrics_path}" ]]; then
    stage_log "Skip train (${ratio_tag}): ${metrics_path} already exists"
    return 0
  fi

  stage_log "Train start: variant=${VARIANT} ratio=${ratio}"
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
    --image_encoder "${BACKBONE}" \
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
}

stage_log "All-enabled ratio pipeline start: dataset=${DATASET} ratios=${RATIOS_STR} seed=${SEED}"

run_cross_stage

for ratio in ${RATIOS_STR}; do
  run_selection_and_train "${ratio}"
done

stage_log "All-enabled ratio pipeline completed"
