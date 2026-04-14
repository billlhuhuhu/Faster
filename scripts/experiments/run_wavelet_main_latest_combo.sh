#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${WAVELET_MAIN_LATEST_DATASET:-flickr}"
BACKBONE="${WAVELET_MAIN_LATEST_BACKBONE:-nfnet}"
TEXT_ENCODER="${WAVELET_MAIN_LATEST_TEXT_ENCODER:-bert}"
VARIANT="${WAVELET_MAIN_LATEST_VARIANT:-wavelet_main_latest}"
SEEDS_STR="${WAVELET_MAIN_LATEST_SEEDS:-0}"
read -r -a SEEDS <<< "${SEEDS_STR}"
BUDGETS_STR="${WAVELET_MAIN_LATEST_BUDGETS:-100 200 500}"
read -r -a BUDGETS <<< "${BUDGETS_STR}"
RATIOS_STR="${WAVELET_MAIN_LATEST_RATIOS:-0.01}"
read -r -a RATIOS <<< "${RATIOS_STR}"

CROSS_OUTPUT_ROOT="${WAVELET_MAIN_LATEST_CROSS_OUTPUT_ROOT:-artifacts/cross_modal_topology_wavelet_main_latest}"
SELECTION_OUTPUT_ROOT="${WAVELET_MAIN_LATEST_SELECTION_OUTPUT_ROOT:-artifacts/subset_selection_wavelet_main_latest}"
TRAIN_OUTPUT_ROOT="${WAVELET_MAIN_LATEST_TRAIN_OUTPUT_ROOT:-artifacts/subset_train_wavelet_main_latest}"
REPORT_NAME="${WAVELET_MAIN_LATEST_REPORT_NAME:-wavelet_main_latest_combo}"

CORRECTION_MODE="${WAVELET_MAIN_LATEST_CORRECTION_MODE:-bidirectional}"
FUSION_MODE="${WAVELET_MAIN_LATEST_FUSION_MODE:-confidence_aware}"
ENABLE_LOCAL_NODE_CONFIDENCE="${WAVELET_MAIN_LATEST_ENABLE_LOCAL_NODE_CONFIDENCE:-0}"

PROXY_LOSS_TYPE="${WAVELET_MAIN_LATEST_PROXY_LOSS_TYPE:-wavelet_main}"
PROXY_INIT_METHOD="${WAVELET_MAIN_LATEST_PROXY_INIT_METHOD:-kmeans}"
PROXY_LR="${WAVELET_MAIN_LATEST_PROXY_LR:-0.05}"
PROXY_REG_WEIGHT="${WAVELET_MAIN_LATEST_PROXY_REG_WEIGHT:-0.01}"
PROXY_NUM_STEPS="${WAVELET_MAIN_LATEST_PROXY_NUM_STEPS:-200}"
PROXY_TARGET_BATCH_SIZE="${WAVELET_MAIN_LATEST_PROXY_TARGET_BATCH_SIZE:-4096}"
PROXY_BATCH_SIZE="${WAVELET_MAIN_LATEST_PROXY_BATCH_SIZE:-4096}"
PROXY_USE_DPP="${WAVELET_MAIN_LATEST_USE_DPP:-1}"
PROXY_DIVERSITY_SIGMA="${WAVELET_MAIN_LATEST_DIVERSITY_SIGMA:-1.0}"
PROXY_GEOMETRY_WEIGHT="${WAVELET_MAIN_LATEST_GEOMETRY_WEIGHT:-1.0}"

LAMBDA_MAIN="${WAVELET_MAIN_LATEST_LAMBDA_MAIN:-1.0}"
LAMBDA_LSRC="${WAVELET_MAIN_LATEST_LAMBDA_LSRC:-0.1}"
LAMBDA_REG="${WAVELET_MAIN_LATEST_LAMBDA_REG:-1.0}"
REG_ALPHA_DIV="${WAVELET_MAIN_LATEST_REG_ALPHA_DIV:-1.0}"
REG_BETA_TOPO="${WAVELET_MAIN_LATEST_REG_BETA_TOPO:-1.0}"
REG_GAMMA_INIT="${WAVELET_MAIN_LATEST_REG_GAMMA_INIT:-1.0}"

ENABLE_LSRC="${WAVELET_MAIN_LATEST_ENABLE_LSRC:-1}"
KEEP_LSRC="${WAVELET_MAIN_LATEST_KEEP_LSRC:-1}"
LSRC_K="${WAVELET_MAIN_LATEST_LSRC_K:-32}"
LSRC_TAU_R="${WAVELET_MAIN_LATEST_LSRC_TAU_R:-1.0}"
LSRC_TAU_C="${WAVELET_MAIN_LATEST_LSRC_TAU_C:-1.0}"
LSRC_ETA="${WAVELET_MAIN_LATEST_LSRC_ETA:-0.5}"
LSRC_BETA="${WAVELET_MAIN_LATEST_LSRC_BETA:-0.5}"
LSRC_BATCH_SIZE="${WAVELET_MAIN_LATEST_LSRC_BATCH_SIZE:-4096}"
LSRC_COVERAGE_MODE="${WAVELET_MAIN_LATEST_LSRC_COVERAGE_MODE:-mean}"
LSRC_REL_LOSS_MODE="${WAVELET_MAIN_LATEST_LSRC_REL_LOSS_MODE:-weight_mean}"
LSRC_USE_GLOBAL_CONFIDENCE="${WAVELET_MAIN_LATEST_LSRC_USE_GLOBAL_CONFIDENCE:-0}"

WAVELET_SCALES="${WAVELET_MAIN_LATEST_WAVELET_SCALES:-1,2,4}"
WAVELET_DISTANCE_TYPE="${WAVELET_MAIN_LATEST_WAVELET_DISTANCE_TYPE:-swd}"
WAVELET_SCHEDULE="${WAVELET_MAIN_LATEST_WAVELET_SCHEDULE:-coarse_to_fine}"
WAVELET_SWD_NUM_PROJECTIONS="${WAVELET_MAIN_LATEST_WAVELET_SWD_NUM_PROJECTIONS:-64}"
WAVELET_SWD_P="${WAVELET_MAIN_LATEST_WAVELET_SWD_P:-2.0}"
WAVELET_MAIN_SCALES="${WAVELET_MAIN_LATEST_MAIN_SCALES:-1,2,4}"
WAVELET_MAIN_SCALE_WEIGHTS="${WAVELET_MAIN_LATEST_MAIN_SCALE_WEIGHTS:-}"
WAVELET_MAIN_SWD_NUM_PROJECTIONS="${WAVELET_MAIN_LATEST_MAIN_SWD_NUM_PROJECTIONS:-64}"
WAVELET_COV_WEIGHT="${WAVELET_MAIN_LATEST_COV_WEIGHT:-0.5}"
WAVELET_EDGE_WEIGHT="${WAVELET_MAIN_LATEST_EDGE_WEIGHT:-0.25}"
WAVELET_CURRICULUM_SCHEDULE="${WAVELET_MAIN_LATEST_CURRICULUM_SCHEDULE:-coarse_to_fine}"

MATCHING_COST_MODE="${WAVELET_MAIN_LATEST_MATCHING_COST_MODE:-candidate_topk}"
COST_ALPHA_DIFF="${WAVELET_MAIN_LATEST_COST_ALPHA_DIFF:-0.25}"
COST_BETA_WAVELET="${WAVELET_MAIN_LATEST_COST_BETA_WAVELET:-1.0}"
MATCHING_WAVELET_WEIGHT="${WAVELET_MAIN_LATEST_MATCHING_WAVELET_WEIGHT:-1.0}"
COST_GAMMA_TOPO="${WAVELET_MAIN_LATEST_COST_GAMMA_TOPO:-0.1}"
COST_ETA_LSRC="${WAVELET_MAIN_LATEST_COST_ETA_LSRC:-0.1}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/${REPORT_NAME}_${DATASET}_${RUN_TIMESTAMP}"
REPORT_DIR="${REPORT_ROOT}/${REPORT_NAME}_${DATASET}_${RUN_TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"
mkdir -p "${REPORT_DIR}"

MODEL_TAG="$(sanitize_component "${BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
IMAGE_ROOT="$(get_image_root "${DATASET}")"
FUSION_TAG="k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_IMAGE}")_a$(sanitize_component "${ALPHA}")"
FEATURE_CACHE_DIR="${FEATURE_CACHE_ROOT}/${DATASET}/train/${MODEL_TAG}"
IMAGE_TOPOLOGY_DIR="${TOPOLOGY_ROOT}/${DATASET}/train/${MODEL_TAG}/image/k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_IMAGE}")"
TEXT_TOPOLOGY_DIR="${TOPOLOGY_ROOT}/${DATASET}/train/${MODEL_TAG}/text/k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_TEXT}")"
CROSS_SUMMARY_PATH="${CROSS_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${FUSION_TAG}/summary.json"

format_ratio_tag_local() {
  local ratio="$1"
  python - <<PY
ratio = float("${ratio}")
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
}

selection_method_tag() {
  if [[ "${KEEP_LSRC}" == "1" || "${ENABLE_LSRC}" == "1" ]]; then
    echo "proxy_opt_lsrc"
  else
    echo "proxy_opt"
  fi
}

run_precompute_if_needed() {
  local topology_extra_args=()
  local cross_extra_args=()

  if [[ -n "${TOPOLOGY_MULTI_SCALE_KS}" ]]; then
    topology_extra_args+=(--multi_scale_ks "${TOPOLOGY_MULTI_SCALE_KS}")
    cross_extra_args+=(--multi_scale_ks "${TOPOLOGY_MULTI_SCALE_KS}")
  fi
  if [[ "${TOPOLOGY_FAISS_USE_GPU}" == "1" ]]; then
    topology_extra_args+=(--faiss_use_gpu)
  fi
  if [[ "${TOPOLOGY_USE_MST_CONNECTIVITY}" == "1" ]]; then
    topology_extra_args+=(--use_mst_connectivity)
  fi
  if [[ "${ENABLE_LOCAL_NODE_CONFIDENCE}" == "1" ]]; then
    cross_extra_args+=(--enable_local_node_confidence)
  fi
  if [[ -n "${WAVELET_FUSION_WEIGHT_A_SCALES}" ]]; then
    cross_extra_args+=(--wavelet_fusion_weight_a_scales "${WAVELET_FUSION_WEIGHT_A_SCALES}")
  fi
  if [[ -n "${WAVELET_FUSION_WEIGHT_B_SCALES}" ]]; then
    cross_extra_args+=(--wavelet_fusion_weight_b_scales "${WAVELET_FUSION_WEIGHT_B_SCALES}")
  fi

  if [[ ! -f "${FEATURE_CACHE_DIR}/img_features_selection.pt" || ! -f "${FEATURE_CACHE_DIR}/txt_features_selection.pt" || ! -f "${FEATURE_CACHE_DIR}/sample_meta.json" ]]; then
    stage_log "Feature cache start: dataset=${DATASET} backbone=${BACKBONE}"
    python "${PROJECT_ROOT}/run_feature_cache.py" \
      --dataset "${DATASET}" \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --selection_image_repr_method "${SELECTION_IMAGE_REPR_METHOD}" \
      --selection_text_repr_method "${SELECTION_TEXT_REPR_METHOD}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --cache_root "${FEATURE_CACHE_ROOT}" \
      --selection_image_size "${SELECTION_IMAGE_SIZE}" \
      --selection_raw_resize_size "${SELECTION_RAW_RESIZE_SIZE}" \
      --selection_raw_pca_dim "${SELECTION_RAW_PCA_DIM}" \
      --selection_image_batch_size "${SELECTION_IMAGE_BATCH_SIZE}" \
      --selection_text_batch_size "${SELECTION_TEXT_BATCH_SIZE}" \
      --hog_orientations "${HOG_ORIENTATIONS}" \
      --hog_pixels_per_cell "${HOG_PIXELS_PER_CELL}" \
      --hog_cells_per_block "${HOG_CELLS_PER_BLOCK}" \
      --color_hist_bins "${COLOR_HIST_BINS}" \
      --color_space "${COLOR_SPACE}" \
      --device "${DEVICE}" \
      > "${RUN_LOG_DIR}/feature_cache.log" 2>&1
    stage_log "Feature cache done"
  else
    stage_log "Skip feature cache: existing cache found at ${FEATURE_CACHE_DIR}"
  fi

  if [[ ! -f "${IMAGE_TOPOLOGY_DIR}/summary.json" ]]; then
    stage_log "Topology graph start: modality=image"
    python "${PROJECT_ROOT}/run_topology_graph.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --modality image \
      --feature_cache_root "${FEATURE_CACHE_ROOT}" \
      --output_root "${TOPOLOGY_ROOT}" \
      --metric "${TOPOLOGY_METRIC_IMAGE}" \
      --knn_k "${K_NEIGHBORS}" \
      --graph_reduce_method "${TOPOLOGY_GRAPH_REDUCE_METHOD}" \
      --graph_feature_dim "${TOPOLOGY_GRAPH_FEATURE_DIM}" \
      --num_eigs 32 \
      --spectral_embedding_dim 32 \
      --n_jobs "${TOPOLOGY_N_JOBS}" \
      --knn_backend "${TOPOLOGY_KNN_BACKEND}" \
      --mst_weight_scale "${TOPOLOGY_MST_WEIGHT_SCALE}" \
      "${topology_extra_args[@]}" \
      > "${RUN_LOG_DIR}/topology_image.log" 2>&1
    stage_log "Topology graph done: modality=image"
  else
    stage_log "Skip topology graph: existing image graph found at ${IMAGE_TOPOLOGY_DIR}"
  fi

  if [[ ! -f "${TEXT_TOPOLOGY_DIR}/summary.json" ]]; then
    stage_log "Topology graph start: modality=text"
    python "${PROJECT_ROOT}/run_topology_graph.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --modality text \
      --feature_cache_root "${FEATURE_CACHE_ROOT}" \
      --output_root "${TOPOLOGY_ROOT}" \
      --metric "${TOPOLOGY_METRIC_TEXT}" \
      --knn_k "${K_NEIGHBORS}" \
      --graph_reduce_method "${TOPOLOGY_GRAPH_REDUCE_METHOD}" \
      --graph_feature_dim "${TOPOLOGY_GRAPH_FEATURE_DIM}" \
      --num_eigs 32 \
      --spectral_embedding_dim 32 \
      --n_jobs "${TOPOLOGY_N_JOBS}" \
      --knn_backend "${TOPOLOGY_KNN_BACKEND}" \
      --mst_weight_scale "${TOPOLOGY_MST_WEIGHT_SCALE}" \
      "${topology_extra_args[@]}" \
      > "${RUN_LOG_DIR}/topology_text.log" 2>&1
    stage_log "Topology graph done: modality=text"
  else
    stage_log "Skip topology graph: existing text graph found at ${TEXT_TOPOLOGY_DIR}"
  fi

  if [[ -f "${CROSS_SUMMARY_PATH}" ]]; then
    stage_log "Skip cross-modal: existing summary found at ${CROSS_SUMMARY_PATH}"
    return 0
  fi

  stage_log "Cross-modal start: correction=${CORRECTION_MODE} fusion_domain=${FUSION_DOMAIN_MODE}"
  python "${PROJECT_ROOT}/run_cross_modal_topology.py" \
    --dataset "${DATASET}" \
    --split train \
    --image_encoder "${BACKBONE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --topology_root "${TOPOLOGY_ROOT}" \
    --output_root "${CROSS_OUTPUT_ROOT}" \
    --metric "${TOPOLOGY_METRIC_IMAGE}" \
    --image_metric "${TOPOLOGY_METRIC_IMAGE}" \
    --text_metric "${TOPOLOGY_METRIC_TEXT}" \
    --k "${K_NEIGHBORS}" \
    --alpha "${ALPHA}" \
    --correction_mode "${CORRECTION_MODE}" \
    --correction_score_mode "${CORRECTION_SCORE_MODE}" \
    --collapse_score_mode "${COLLAPSE_SCORE_MODE}" \
    --collapse_score_weight_edge "${COLLAPSE_SCORE_WEIGHT_EDGE}" \
    --collapse_score_weight_a2b "${COLLAPSE_SCORE_WEIGHT_A2B}" \
    --collapse_score_weight_b2a "${COLLAPSE_SCORE_WEIGHT_B2A}" \
    --collapse_score_weight_nbr2nbr "${COLLAPSE_SCORE_WEIGHT_NBR2NBR}" \
    --collapse_neighbor_topk "${COLLAPSE_NEIGHBOR_TOPK}" \
    --fusion_domain_mode "${FUSION_DOMAIN_MODE}" \
    --fusion_mode "${FUSION_MODE}" \
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
    --num_eigs "${CROSS_MODAL_NUM_EIGS}" \
    --spectral_embedding_dim "${CROSS_MODAL_EMBED_DIM}" \
    "${cross_extra_args[@]}" \
    > "${RUN_LOG_DIR}/cross_modal.log" 2>&1
  stage_log "Cross-modal done"
}

run_selection_abs() {
  local budget="$1"
  local seed="$2"
  local budget_tag
  local method_tag
  local selected_indices_path
  local select_log
  local train_log
  local metrics_path
  local selection_extra_args=()
  local train_extra_args=()

  budget_tag="$(format_budget_tag "${budget}")"
  method_tag="$(selection_method_tag)"
  selected_indices_path="${SELECTION_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${budget_tag}/${method_tag}/seed_${seed}/selected_indices.json"
  metrics_path="${TRAIN_OUTPUT_ROOT}/${DATASET}/${MODEL_TAG}/${budget_tag}/${VARIANT}/seed_${seed}/metrics.json"
  select_log="${RUN_LOG_DIR}/${budget_tag}_seed${seed}_select.log"
  train_log="${RUN_LOG_DIR}/${budget_tag}_seed${seed}_train.log"

  if [[ "${PROXY_USE_DPP}" == "1" ]]; then
    selection_extra_args+=(--use_dpp)
  fi
  if [[ "${ENABLE_LSRC}" == "1" ]]; then
    selection_extra_args+=(--enable_lsrc)
  fi
  if [[ "${KEEP_LSRC}" == "1" ]]; then
    selection_extra_args+=(--keep_lsrc)
  else
    selection_extra_args+=(--disable_lsrc)
  fi
  if [[ "${LSRC_USE_GLOBAL_CONFIDENCE}" == "1" ]]; then
    selection_extra_args+=(--lsrc_use_global_confidence)
  fi
  if [[ -n "${WAVELET_MAIN_SCALE_WEIGHTS}" ]]; then
    selection_extra_args+=(--wavelet_main_scale_weights "${WAVELET_MAIN_SCALE_WEIGHTS}")
  fi

  if [[ ! -f "${selected_indices_path}" ]]; then
    stage_log "Selection start: budget=${budget} seed=${seed}"
    python "${PROJECT_ROOT}/run_subset_selection.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --feature_cache_root "${FEATURE_CACHE_ROOT}" \
      --cross_modal_root "${CROSS_OUTPUT_ROOT}" \
      --output_root "${SELECTION_OUTPUT_ROOT}" \
      --metric "${TOPOLOGY_METRIC_IMAGE}" \
      --k "${K_NEIGHBORS}" \
      --alpha "${ALPHA}" \
      --budget_size "${budget}" \
      --selection_method proxy_opt \
      --reference_embedding_mode "${SUBSET_REFERENCE_EMBEDDING_MODE}" \
      --spectral_weight "${SUBSET_SPECTRAL_WEIGHT}" \
      --random_state "${seed}" \
      --device "${DEVICE}" \
      --proxy_projection_dim "${PROXY_PROJECTION_DIM}" \
      --proxy_init_method "${PROXY_INIT_METHOD}" \
      --proxy_loss_type "${PROXY_LOSS_TYPE}" \
      --proxy_lr "${PROXY_LR}" \
      --proxy_num_steps "${PROXY_NUM_STEPS}" \
      --proxy_reg_weight "${PROXY_REG_WEIGHT}" \
      --proxy_target_batch_size "${PROXY_TARGET_BATCH_SIZE}" \
      --proxy_batch_size "${PROXY_BATCH_SIZE}" \
      --wavelet_scales "${WAVELET_SCALES}" \
      --wavelet_distance_type "${WAVELET_DISTANCE_TYPE}" \
      --wavelet_schedule "${WAVELET_SCHEDULE}" \
      --wavelet_swd_num_projections "${WAVELET_SWD_NUM_PROJECTIONS}" \
      --wavelet_swd_p "${WAVELET_SWD_P}" \
      --lambda_main "${LAMBDA_MAIN}" \
      --wavelet_main_scales "${WAVELET_MAIN_SCALES}" \
      --wavelet_main_swd_num_projections "${WAVELET_MAIN_SWD_NUM_PROJECTIONS}" \
      --wavelet_cov_weight "${WAVELET_COV_WEIGHT}" \
      --wavelet_edge_weight "${WAVELET_EDGE_WEIGHT}" \
      --wavelet_curriculum_schedule "${WAVELET_CURRICULUM_SCHEDULE}" \
      --lambda_lsrc "${LAMBDA_LSRC}" \
      --lambda_reg "${LAMBDA_REG}" \
      --reg_alpha_div "${REG_ALPHA_DIV}" \
      --reg_beta_topo "${REG_BETA_TOPO}" \
      --reg_gamma_init "${REG_GAMMA_INIT}" \
      --lsrc_k "${LSRC_K}" \
      --lsrc_tau_r "${LSRC_TAU_R}" \
      --lsrc_tau_c "${LSRC_TAU_C}" \
      --lsrc_eta "${LSRC_ETA}" \
      --lsrc_beta "${LSRC_BETA}" \
      --lsrc_batch_size "${LSRC_BATCH_SIZE}" \
      --lsrc_coverage_mode "${LSRC_COVERAGE_MODE}" \
      --lsrc_rel_loss_mode "${LSRC_REL_LOSS_MODE}" \
      --matching_cost_mode "${MATCHING_COST_MODE}" \
      --cost_alpha_diff "${COST_ALPHA_DIFF}" \
      --cost_beta_wavelet "${COST_BETA_WAVELET}" \
      --matching_wavelet_weight "${MATCHING_WAVELET_WEIGHT}" \
      --cost_gamma_topo "${COST_GAMMA_TOPO}" \
      --cost_eta_lsrc "${COST_ETA_LSRC}" \
      --geometry_weight "${PROXY_GEOMETRY_WEIGHT}" \
      --diversity_sigma "${PROXY_DIVERSITY_SIGMA}" \
      "${selection_extra_args[@]}" \
      > "${select_log}" 2>&1
    stage_log "Selection done: budget=${budget} seed=${seed}"
  else
    stage_log "Skip selection: existing selected_indices found at ${selected_indices_path}"
  fi

  if [[ ! -f "${metrics_path}" ]]; then
    if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
      train_extra_args+=(--no_aug)
    fi
    stage_log "Train start: budget=${budget} seed=${seed}"
    python "${PROJECT_ROOT}/run_subset_train.py" \
      --dataset "${DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --selected_indices_path "${selected_indices_path}" \
      --subset_size "${budget}" \
      --subset_tag "${VARIANT}" \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --output_root "${TRAIN_OUTPUT_ROOT}" \
      --batch_size_train "${BATCH_TRAIN}" \
      --batch_size_test "${BATCH_TEST}" \
      --text_batch_size "${TEXT_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      "${train_extra_args[@]}" \
      > "${train_log}" 2>&1
    stage_log "Train done: budget=${budget} seed=${seed}"
  else
    stage_log "Skip train: existing metrics found at ${metrics_path}"
  fi
}

run_selection_ratio() {
  local ratio="$1"
  local seed="$2"
  local ratio_tag
  local method_tag
  local selected_indices_path
  local select_log
  local train_log
  local metrics_path
  local selection_extra_args=()
  local train_extra_args=()

  ratio_tag="$(format_ratio_tag_local "${ratio}")"
  method_tag="$(selection_method_tag)"
  selected_indices_path="${SELECTION_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${ratio_tag}/${method_tag}/seed_${seed}/selected_indices.json"
  metrics_path="${TRAIN_OUTPUT_ROOT}/${DATASET}/${MODEL_TAG}/${ratio_tag}/${VARIANT}/seed_${seed}/metrics.json"
  select_log="${RUN_LOG_DIR}/${ratio_tag}_seed${seed}_select.log"
  train_log="${RUN_LOG_DIR}/${ratio_tag}_seed${seed}_train.log"

  if [[ "${PROXY_USE_DPP}" == "1" ]]; then
    selection_extra_args+=(--use_dpp)
  fi
  if [[ "${ENABLE_LSRC}" == "1" ]]; then
    selection_extra_args+=(--enable_lsrc)
  fi
  if [[ "${KEEP_LSRC}" == "1" ]]; then
    selection_extra_args+=(--keep_lsrc)
  else
    selection_extra_args+=(--disable_lsrc)
  fi
  if [[ "${LSRC_USE_GLOBAL_CONFIDENCE}" == "1" ]]; then
    selection_extra_args+=(--lsrc_use_global_confidence)
  fi
  if [[ -n "${WAVELET_MAIN_SCALE_WEIGHTS}" ]]; then
    selection_extra_args+=(--wavelet_main_scale_weights "${WAVELET_MAIN_SCALE_WEIGHTS}")
  fi

  if [[ ! -f "${selected_indices_path}" ]]; then
    stage_log "Selection start: ratio=${ratio} seed=${seed}"
    python "${PROJECT_ROOT}/run_subset_selection.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --feature_cache_root "${FEATURE_CACHE_ROOT}" \
      --cross_modal_root "${CROSS_OUTPUT_ROOT}" \
      --output_root "${SELECTION_OUTPUT_ROOT}" \
      --metric "${TOPOLOGY_METRIC_IMAGE}" \
      --k "${K_NEIGHBORS}" \
      --alpha "${ALPHA}" \
      --budget_ratio "${ratio}" \
      --selection_method proxy_opt \
      --reference_embedding_mode "${SUBSET_REFERENCE_EMBEDDING_MODE}" \
      --spectral_weight "${SUBSET_SPECTRAL_WEIGHT}" \
      --random_state "${seed}" \
      --device "${DEVICE}" \
      --proxy_projection_dim "${PROXY_PROJECTION_DIM}" \
      --proxy_init_method "${PROXY_INIT_METHOD}" \
      --proxy_loss_type "${PROXY_LOSS_TYPE}" \
      --proxy_lr "${PROXY_LR}" \
      --proxy_num_steps "${PROXY_NUM_STEPS}" \
      --proxy_reg_weight "${PROXY_REG_WEIGHT}" \
      --proxy_target_batch_size "${PROXY_TARGET_BATCH_SIZE}" \
      --proxy_batch_size "${PROXY_BATCH_SIZE}" \
      --wavelet_scales "${WAVELET_SCALES}" \
      --wavelet_distance_type "${WAVELET_DISTANCE_TYPE}" \
      --wavelet_schedule "${WAVELET_SCHEDULE}" \
      --wavelet_swd_num_projections "${WAVELET_SWD_NUM_PROJECTIONS}" \
      --wavelet_swd_p "${WAVELET_SWD_P}" \
      --lambda_main "${LAMBDA_MAIN}" \
      --wavelet_main_scales "${WAVELET_MAIN_SCALES}" \
      --wavelet_main_swd_num_projections "${WAVELET_MAIN_SWD_NUM_PROJECTIONS}" \
      --wavelet_cov_weight "${WAVELET_COV_WEIGHT}" \
      --wavelet_edge_weight "${WAVELET_EDGE_WEIGHT}" \
      --wavelet_curriculum_schedule "${WAVELET_CURRICULUM_SCHEDULE}" \
      --lambda_lsrc "${LAMBDA_LSRC}" \
      --lambda_reg "${LAMBDA_REG}" \
      --reg_alpha_div "${REG_ALPHA_DIV}" \
      --reg_beta_topo "${REG_BETA_TOPO}" \
      --reg_gamma_init "${REG_GAMMA_INIT}" \
      --lsrc_k "${LSRC_K}" \
      --lsrc_tau_r "${LSRC_TAU_R}" \
      --lsrc_tau_c "${LSRC_TAU_C}" \
      --lsrc_eta "${LSRC_ETA}" \
      --lsrc_beta "${LSRC_BETA}" \
      --lsrc_batch_size "${LSRC_BATCH_SIZE}" \
      --lsrc_coverage_mode "${LSRC_COVERAGE_MODE}" \
      --lsrc_rel_loss_mode "${LSRC_REL_LOSS_MODE}" \
      --matching_cost_mode "${MATCHING_COST_MODE}" \
      --cost_alpha_diff "${COST_ALPHA_DIFF}" \
      --cost_beta_wavelet "${COST_BETA_WAVELET}" \
      --matching_wavelet_weight "${MATCHING_WAVELET_WEIGHT}" \
      --cost_gamma_topo "${COST_GAMMA_TOPO}" \
      --cost_eta_lsrc "${COST_ETA_LSRC}" \
      --geometry_weight "${PROXY_GEOMETRY_WEIGHT}" \
      --diversity_sigma "${PROXY_DIVERSITY_SIGMA}" \
      "${selection_extra_args[@]}" \
      > "${select_log}" 2>&1
    stage_log "Selection done: ratio=${ratio} seed=${seed}"
  else
    stage_log "Skip selection: existing selected_indices found at ${selected_indices_path}"
  fi

  if [[ ! -f "${metrics_path}" ]]; then
    if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
      train_extra_args+=(--no_aug)
    fi
    stage_log "Train start: ratio=${ratio} seed=${seed}"
    python "${PROJECT_ROOT}/run_subset_train.py" \
      --dataset "${DATASET}" \
      --image_root "${IMAGE_ROOT}" \
      --ann_root "${ANN_ROOT}" \
      --selected_indices_path "${selected_indices_path}" \
      --subset_ratio "${ratio}" \
      --subset_tag "${VARIANT}" \
      --image_encoder "${BACKBONE}" \
      --text_encoder "${TEXT_ENCODER}" \
      --output_root "${TRAIN_OUTPUT_ROOT}" \
      --batch_size_train "${BATCH_TRAIN}" \
      --batch_size_test "${BATCH_TEST}" \
      --text_batch_size "${TEXT_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      "${train_extra_args[@]}" \
      > "${train_log}" 2>&1
    stage_log "Train done: ratio=${ratio} seed=${seed}"
  else
    stage_log "Skip train: existing metrics found at ${metrics_path}"
  fi
}

cd "${PROJECT_ROOT}"
run_precompute_if_needed

stage_log "Wavelet-main latest combo start: dataset=${DATASET} budgets=${BUDGETS[*]} ratios=${RATIOS[*]} seeds=${SEEDS[*]}"

for budget in "${BUDGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_selection_abs "${budget}" "${seed}"
  done
done

for ratio in "${RATIOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_selection_ratio "${ratio}" "${seed}"
  done
done

RAW_CSV_PATH="${REPORT_DIR}/wavelet_main_latest_combo_raw.csv"
SUMMARY_CSV_PATH="${REPORT_DIR}/wavelet_main_latest_combo_summary.csv"
MISSING_TXT_PATH="${REPORT_DIR}/missing_metrics.txt"

python - "${TRAIN_OUTPUT_ROOT}" "${DATASET}" "${MODEL_TAG}" "${VARIANT}" "${RAW_CSV_PATH}" "${SUMMARY_CSV_PATH}" "${MISSING_TXT_PATH}" "${BUDGETS[*]}" "${RATIOS[*]}" "${SEEDS[*]}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


subset_train_root = Path(sys.argv[1])
dataset = sys.argv[2]
model_tag = sys.argv[3]
variant = sys.argv[4]
raw_csv_path = Path(sys.argv[5])
summary_csv_path = Path(sys.argv[6])
missing_txt_path = Path(sys.argv[7])
budgets = [item for item in sys.argv[8].split() if item.strip()]
ratios = [item for item in sys.argv[9].split() if item.strip()]
seeds = [item for item in sys.argv[10].split() if item.strip()]

raw_rows = []
missing = []

targets = []
for budget in budgets:
    targets.append(("abs", f"size_{int(budget):04d}", str(int(budget))))
for ratio in ratios:
    ratio_value = float(ratio)
    targets.append(("ratio", f"ratio_{int(round(ratio_value * 100)):02d}", f"{ratio_value:.6f}"))

for budget_type, budget_tag, budget_value in targets:
    for seed in seeds:
        metrics_path = subset_train_root / dataset / model_tag / budget_tag / variant / f"seed_{int(seed)}" / "metrics.json"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        raw_rows.append(
            {
                "dataset": dataset,
                "model_tag": model_tag,
                "variant": variant,
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
    key = (row["dataset"], row["model_tag"], row["variant"], row["budget_type"], row["budget_tag"], row["budget_value"])
    grouped.setdefault(key, []).append(row)

summary_rows = []
for key in sorted(grouped.keys()):
    dataset, model_tag, variant, budget_type, budget_tag, budget_value = key
    rows = grouped[key]
    summary = {
        "dataset": dataset,
        "model_tag": model_tag,
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

stage_log "Wavelet-main latest combo completed. Logs saved to ${RUN_LOG_DIR}"
stage_log "Wavelet-main latest combo report dir: ${REPORT_DIR}"
stage_log "Wavelet-main latest combo raw csv: ${RAW_CSV_PATH}"
stage_log "Wavelet-main latest combo summary csv: ${SUMMARY_CSV_PATH}"
