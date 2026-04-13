#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASET="${ABLATION_DATASET:-flickr}"
SEED="${ABLATION_SEED:-0}"
BACKBONE="${ABLATION_BACKBONE:-nfnet}"
TEXT_ENCODER="${ABLATION_TEXT_ENCODER:-bert}"
BUDGETS_STR="${ABLATION_BUDGETS:-100 200 500}"
MODEL_TAG="$(sanitize_component "${BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
FUSION_TAG="k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_IMAGE}")_a$(sanitize_component "${ALPHA}")"
IMAGE_ROOT="$(get_image_root "${DATASET}")"

ABLATION_CROSS_ROOT="${ABLATION_CROSS_ROOT:-artifacts/cross_modal_topology_ablation}"
ABLATION_SELECTION_ROOT="${ABLATION_SELECTION_ROOT:-artifacts/subset_selection_ablation}"
ABLATION_TRAIN_ROOT="${ABLATION_TRAIN_ROOT:-artifacts/subset_train_ablation}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/direction_ablation_${DATASET}_${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}" "${REPORT_ROOT}"

run_cross_stage() {
  local variant="$1"
  local correction_mode="$2"
  local fusion_mode="$3"
  local cross_root="${ABLATION_CROSS_ROOT}/${variant}"
  local summary_path="${cross_root}/${DATASET}/train/${MODEL_TAG}/${FUSION_TAG}/summary.json"
  local log_path="${RUN_LOG_DIR}/${variant}_cross.log"

  if [[ -f "${summary_path}" ]]; then
    stage_log "Skip cross-modal (${variant}): ${summary_path} already exists"
    return 0
  fi

  if [[ "${ENABLE_SELECTION_EFFICIENCY_BENCHMARK}" == "1" ]]; then
    stage_log "Skip standalone cross-modal (${variant}): benchmark mode will wrap cross+selection together"
    return 0
  fi

  local wavelet_extra_args=()
  if [[ -n "${WAVELET_FUSION_WEIGHT_A_SCALES}" ]]; then
    wavelet_extra_args+=(--wavelet_fusion_weight_a_scales "${WAVELET_FUSION_WEIGHT_A_SCALES}")
  fi
  if [[ -n "${WAVELET_FUSION_WEIGHT_B_SCALES}" ]]; then
    wavelet_extra_args+=(--wavelet_fusion_weight_b_scales "${WAVELET_FUSION_WEIGHT_B_SCALES}")
  fi

  stage_log "Cross-modal start: variant=${variant} correction=${correction_mode} fusion=${fusion_mode}"
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
    --correction_mode "${correction_mode}" \
    --correction_score_mode "${CORRECTION_SCORE_MODE}" \
    --collapse_score_mode "${COLLAPSE_SCORE_MODE}" \
    --collapse_score_weight_edge "${COLLAPSE_SCORE_WEIGHT_EDGE}" \
    --collapse_score_weight_a2b "${COLLAPSE_SCORE_WEIGHT_A2B}" \
    --collapse_score_weight_b2a "${COLLAPSE_SCORE_WEIGHT_B2A}" \
    --collapse_score_weight_nbr2nbr "${COLLAPSE_SCORE_WEIGHT_NBR2NBR}" \
    --collapse_neighbor_topk "${COLLAPSE_NEIGHBOR_TOPK}" \
    --fusion_domain_mode "${FUSION_DOMAIN_MODE}" \
    --fusion_mode "${fusion_mode}" \
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
  local variant="$1"
  local cross_root="$2"
  local budget="$3"
  local enable_lsrc="$4"
  local correction_mode="$5"
  local fusion_mode="$6"
  local budget_tag
  budget_tag="$(format_budget_tag "${budget}")"

  local selection_root="${ABLATION_SELECTION_ROOT}/${variant}"
  local train_root="${ABLATION_TRAIN_ROOT}/${variant}"
  local selection_method_tag="proxy_opt"
  if [[ "${enable_lsrc}" == "1" ]]; then
    selection_method_tag="proxy_opt_lsrc"
  fi

  local selected_indices_path="${selection_root}/${DATASET}/train/${MODEL_TAG}/${budget_tag}/${selection_method_tag}/seed_${SEED}/selected_indices.json"
  local metrics_path="${train_root}/${DATASET}/${MODEL_TAG}/${budget_tag}/${variant}/seed_${SEED}/metrics.json"
  local select_log="${RUN_LOG_DIR}/${variant}_${budget_tag}_select.log"
  local train_log="${RUN_LOG_DIR}/${variant}_${budget_tag}_train.log"
  local benchmark_output_dir="${SELECTION_EFFICIENCY_ROOT}/direction_ablation/${variant}/${DATASET}/${MODEL_TAG}/${budget_tag}/seed_${SEED}"
  local benchmark_summary_path="${benchmark_output_dir}/selection_efficiency_summary.json"

  if [[ ! -f "${selected_indices_path}" || ( "${ENABLE_SELECTION_EFFICIENCY_BENCHMARK}" == "1" && ! -f "${benchmark_summary_path}" ) ]]; then
    stage_log "Selection start: variant=${variant} budget=${budget}"
    local selection_extra_args=()
    if [[ "${enable_lsrc}" == "1" ]]; then
      selection_extra_args+=(
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
    fi

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
        --correction_mode "${correction_mode}"
        --correction_score_mode "${CORRECTION_SCORE_MODE}"
        --collapse_score_mode "${COLLAPSE_SCORE_MODE}"
        --collapse_score_weight_edge "${COLLAPSE_SCORE_WEIGHT_EDGE}"
        --collapse_score_weight_a2b "${COLLAPSE_SCORE_WEIGHT_A2B}"
        --collapse_score_weight_b2a "${COLLAPSE_SCORE_WEIGHT_B2A}"
        --collapse_score_weight_nbr2nbr "${COLLAPSE_SCORE_WEIGHT_NBR2NBR}"
        --collapse_neighbor_topk "${COLLAPSE_NEIGHBOR_TOPK}"
        --fusion_domain_mode "${FUSION_DOMAIN_MODE}"
        --fusion_mode "${fusion_mode}"
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
        --budget_size "${budget}"
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
        --variant_name "${variant}"
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
        --budget_size "${budget}" \
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
    stage_log "Skip selection (${variant}, ${budget}): ${selected_indices_path} already exists"
  fi

  if [[ -f "${metrics_path}" ]]; then
    stage_log "Skip train (${variant}, ${budget}): ${metrics_path} already exists"
    return 0
  fi

  stage_log "Train start: variant=${variant} budget=${budget}"
  train_extra_args=()
  if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
    train_extra_args+=(--no_aug)
  fi

  python "${PROJECT_ROOT}/run_subset_train.py" \
    --dataset "${DATASET}" \
    --image_root "${IMAGE_ROOT}" \
    --ann_root "${ANN_ROOT}" \
    --selected_indices_path "${selected_indices_path}" \
    --subset_size "${budget}" \
    --subset_tag "${variant}" \
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

stage_log "Direction ablation start: dataset=${DATASET} budgets=${BUDGETS_STR} seed=${SEED}"

# dir1: bidirectional correction only
run_cross_stage "dir1_bidir_only" "bidirectional" "intersection"

# dir2: confidence-aware fusion only
run_cross_stage "dir2_conf_only" "directional" "confidence_aware"

# dir3: LSRC only, reuse the old directional/intersection setting but regenerate
# cross-modal artifacts with the current code so corrected image/text graphs exist.
run_cross_stage "dir3_lsrc_only" "directional" "intersection"

# all enabled: bidirectional correction + confidence-aware fusion
run_cross_stage "all_enabled" "bidirectional" "confidence_aware"

for budget in ${BUDGETS_STR}; do
  run_selection_and_train "dir1_bidir_only" "${ABLATION_CROSS_ROOT}/dir1_bidir_only" "${budget}" "0" "bidirectional" "intersection"
  run_selection_and_train "dir2_conf_only" "${ABLATION_CROSS_ROOT}/dir2_conf_only" "${budget}" "0" "directional" "confidence_aware"
  run_selection_and_train "dir3_lsrc_only" "${ABLATION_CROSS_ROOT}/dir3_lsrc_only" "${budget}" "1" "directional" "intersection"
  run_selection_and_train "all_enabled" "${ABLATION_CROSS_ROOT}/all_enabled" "${budget}" "1" "bidirectional" "confidence_aware"
done

stage_log "Aggregating direction ablation metrics"
python "${PROJECT_ROOT}/tools/aggregate_direction_ablation.py" \
  --dataset "${DATASET}" \
  --backbone "${BACKBONE}" \
  --text_encoder "${TEXT_ENCODER}" \
  --seed "${SEED}" \
  --base_train_root "${SUBSET_TRAIN_ROOT}" \
  --ablation_train_root "${ABLATION_TRAIN_ROOT}" \
  --report_root "${REPORT_ROOT}" \
  --budgets ${BUDGETS_STR}

stage_log "Direction ablation completed"
