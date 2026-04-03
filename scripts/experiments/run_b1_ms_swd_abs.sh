#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${B1_MS_SWD_DATASET:-flickr}"
BACKBONE="${B1_MS_SWD_BACKBONE:-nfnet}"
TEXT_ENCODER="${B1_MS_SWD_TEXT_ENCODER:-bert}"
VARIANT="${B1_MS_SWD_VARIANT:-b1_ms_swd}"
SEEDS_STR="${B1_MS_SWD_SEEDS:-0}"
read -r -a SEEDS <<< "${SEEDS_STR}"
BUDGETS_STR="${B1_MS_SWD_BUDGETS:-100 200 500}"
read -r -a BUDGETS <<< "${BUDGETS_STR}"

# Cross-modal artifacts can be safely reused because this script only changes
# the selection/training scheme, not feature extraction or unified graph construction.
CROSS_OUTPUT_ROOT="${B1_MS_SWD_CROSS_OUTPUT_ROOT:-artifacts/cross_modal_topology_b1}"
SELECTION_OUTPUT_ROOT="${B1_MS_SWD_SELECTION_OUTPUT_ROOT:-artifacts/subset_selection_b1_ms_swd_abs}"
TRAIN_OUTPUT_ROOT="${B1_MS_SWD_TRAIN_OUTPUT_ROOT:-artifacts/subset_train_b1_ms_swd_abs}"
REPORT_NAME="${B1_MS_SWD_REPORT_NAME:-b1_ms_swd_abs}"

CORRECTION_MODE="${B1_MS_SWD_CORRECTION_MODE:-bidirectional}"
FUSION_MODE="${B1_MS_SWD_FUSION_MODE:-confidence_aware}"
EMBEDDING_TYPE="${B1_MS_SWD_EMBEDDING_TYPE:-diffusion}"
DIFFUSION_DIM="${B1_MS_SWD_DIFFUSION_DIM:-32}"
DIFFUSION_TIME="${B1_MS_SWD_DIFFUSION_TIME:-1.0}"
DIFFUSION_EIG_SOLVER="${B1_MS_SWD_DIFFUSION_EIG_SOLVER:-auto}"
ENABLE_LOCAL_NODE_CONFIDENCE="${B1_MS_SWD_ENABLE_LOCAL_NODE_CONFIDENCE:-0}"

PROXY_LOSS_TYPE="${B1_MS_SWD_PROXY_LOSS_TYPE:-diffusion_ms_swd}"
PROXY_INIT_METHOD="${B1_MS_SWD_PROXY_INIT_METHOD:-kmeans}"
PROXY_LR="${B1_MS_SWD_PROXY_LR:-0.05}"
PROXY_REG_WEIGHT="${B1_MS_SWD_PROXY_REG_WEIGHT:-0.01}"
ENABLE_WAVELET_MULTISCALE="${B1_MS_SWD_ENABLE_WAVELET_MULTISCALE:-1}"
USE_DPP="${B1_MS_SWD_USE_DPP:-1}"
WAVELET_SCALES="${B1_MS_SWD_WAVELET_SCALES:-1,2,4}"
WAVELET_DISTANCE_TYPE="${B1_MS_SWD_WAVELET_DISTANCE_TYPE:-swd}"
WAVELET_SCHEDULE="${B1_MS_SWD_WAVELET_SCHEDULE:-coarse_to_fine}"
WAVELET_LOSS_WEIGHT="${B1_MS_SWD_WAVELET_LOSS_WEIGHT:-0.1}"
SWD_NUM_PROJECTIONS="${B1_MS_SWD_NUM_PROJECTIONS:-64}"
SWD_P="${B1_MS_SWD_P:-2.0}"
SWD_USE_FIXED_PROJECTIONS="${B1_MS_SWD_USE_FIXED_PROJECTIONS:-0}"
WAVELET_SWD_NUM_PROJECTIONS="${B1_MS_SWD_WAVELET_NUM_PROJECTIONS:-64}"
WAVELET_SWD_P="${B1_MS_SWD_WAVELET_P:-2.0}"

ENABLE_LSRC="${B1_MS_SWD_ENABLE_LSRC:-1}"
LSRC_K="${B1_MS_SWD_LSRC_K:-32}"
LSRC_TAU_R="${B1_MS_SWD_LSRC_TAU_R:-1.0}"
LSRC_TAU_C="${B1_MS_SWD_LSRC_TAU_C:-1.0}"
LSRC_ETA="${B1_MS_SWD_LSRC_ETA:-0.5}"
LSRC_BETA="${B1_MS_SWD_LSRC_BETA:-0.5}"
LSRC_BATCH_SIZE="${B1_MS_SWD_LSRC_BATCH_SIZE:-4096}"
LSRC_COVERAGE_MODE="${B1_MS_SWD_LSRC_COVERAGE_MODE:-mean}"
LSRC_REL_LOSS_MODE="${B1_MS_SWD_LSRC_REL_LOSS_MODE:-weight_mean}"
LSRC_USE_GLOBAL_CONFIDENCE="${B1_MS_SWD_LSRC_USE_GLOBAL_CONFIDENCE:-0}"

LAMBDA_DIFF="${B1_MS_SWD_LAMBDA_DIFF:-1.0}"
LAMBDA_MS="${B1_MS_SWD_LAMBDA_MS:-0.0}"
LAMBDA_LSRC="${B1_MS_SWD_LAMBDA_LSRC:-0.1}"
LSRC_MU="${B1_MS_SWD_LSRC_MU:-0.5}"
LAMBDA_REG="${B1_MS_SWD_LAMBDA_REG:-1.0}"
REG_ALPHA_DIV="${B1_MS_SWD_REG_ALPHA_DIV:-1.0}"
REG_BETA_TOPO="${B1_MS_SWD_REG_BETA_TOPO:-1.0}"
REG_GAMMA_INIT="${B1_MS_SWD_REG_GAMMA_INIT:-1.0}"

MATCHING_COST_MODE="${B1_MS_SWD_MATCHING_COST_MODE:-candidate_topk}"
COST_ALPHA_DIFF="${B1_MS_SWD_COST_ALPHA_DIFF:-1.0}"
COST_BETA_WAVELET="${B1_MS_SWD_COST_BETA_WAVELET:-0.25}"
COST_GAMMA_TOPO="${B1_MS_SWD_COST_GAMMA_TOPO:-0.1}"
COST_ETA_LSRC="${B1_MS_SWD_COST_ETA_LSRC:-0.1}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/${REPORT_NAME}_${DATASET}_${RUN_TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

MODEL_TAG="$(sanitize_component "${BACKBONE}")_$(sanitize_component "${TEXT_ENCODER}")"
IMAGE_ROOT="$(get_image_root "${DATASET}")"
FUSION_TAG="k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_IMAGE}")_a$(sanitize_component "${ALPHA}")"
FEATURE_CACHE_DIR="${FEATURE_CACHE_ROOT}/${DATASET}/train/${MODEL_TAG}"
IMAGE_TOPOLOGY_DIR="${TOPOLOGY_ROOT}/${DATASET}/train/${MODEL_TAG}/image/k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_IMAGE}")"
TEXT_TOPOLOGY_DIR="${TOPOLOGY_ROOT}/${DATASET}/train/${MODEL_TAG}/text/k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_TEXT}")"
CROSS_SUMMARY_PATH="${CROSS_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${FUSION_TAG}/summary.json"

run_precompute_if_needed() {
  local topology_extra_args=()
  if [[ -n "${TOPOLOGY_MULTI_SCALE_KS}" ]]; then
    topology_extra_args+=(--multi_scale_ks "${TOPOLOGY_MULTI_SCALE_KS}")
  fi
  if [[ "${TOPOLOGY_FAISS_USE_GPU}" == "1" ]]; then
    topology_extra_args+=(--faiss_use_gpu)
  fi
  if [[ "${TOPOLOGY_USE_MST_CONNECTIVITY}" == "1" ]]; then
    topology_extra_args+=(--use_mst_connectivity)
  fi

  if [[ -f "${CROSS_SUMMARY_PATH}" ]]; then
    stage_log "Skip precompute: existing cross-modal summary found at ${CROSS_SUMMARY_PATH}"
    return 0
  fi

  if [[ -f "${FEATURE_CACHE_DIR}/img_features_selection.pt" && -f "${FEATURE_CACHE_DIR}/txt_features_selection.pt" && -f "${FEATURE_CACHE_DIR}/sample_meta.json" ]]; then
    stage_log "Skip feature cache: existing cache found at ${FEATURE_CACHE_DIR}"
  else
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
  fi

  if [[ -f "${IMAGE_TOPOLOGY_DIR}/summary.json" ]]; then
    stage_log "Skip topology graph: existing image graph found at ${IMAGE_TOPOLOGY_DIR}"
  else
    stage_log "Topology graph start: dataset=${DATASET} modality=image"
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
  fi

  if [[ -f "${TEXT_TOPOLOGY_DIR}/summary.json" ]]; then
    stage_log "Skip topology graph: existing text graph found at ${TEXT_TOPOLOGY_DIR}"
  else
    stage_log "Topology graph start: dataset=${DATASET} modality=text"
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
  fi

  local cross_extra_args=()
  if [[ -n "${TOPOLOGY_MULTI_SCALE_KS}" ]]; then
    cross_extra_args+=(--multi_scale_ks "${TOPOLOGY_MULTI_SCALE_KS}")
  fi
  if [[ "${ENABLE_LOCAL_NODE_CONFIDENCE}" == "1" ]]; then
    cross_extra_args+=(--enable_local_node_confidence)
  fi

  stage_log "Cross-modal start: embedding=${EMBEDDING_TYPE} correction=${CORRECTION_MODE} fusion=${FUSION_MODE}"
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
    --fusion_mode "${FUSION_MODE}" \
    --num_eigs "${CROSS_MODAL_NUM_EIGS}" \
    --spectral_embedding_dim "${CROSS_MODAL_EMBED_DIM}" \
    --embedding_type "${EMBEDDING_TYPE}" \
    --diffusion_dim "${DIFFUSION_DIM}" \
    --diffusion_time "${DIFFUSION_TIME}" \
    --diffusion_eig_solver "${DIFFUSION_EIG_SOLVER}" \
    "${cross_extra_args[@]}" \
    > "${RUN_LOG_DIR}/cross_modal.log" 2>&1
  stage_log "Cross-modal done"
}

run_one_budget_seed() {
  local budget="$1"
  local seed="$2"
  local budget_tag
  local selection_method_tag="proxy_opt"
  local selection_log
  local train_log
  local selected_indices_path
  local metrics_path
  local selection_extra_args=()
  local train_extra_args=()

  budget_tag="$(format_budget_tag "${budget}")"
  if [[ "${ENABLE_LSRC}" == "1" ]]; then
    selection_method_tag="${selection_method_tag}_lsrc"
  fi

  selected_indices_path="${SELECTION_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${budget_tag}/${selection_method_tag}/seed_${seed}/selected_indices.json"
  metrics_path="${TRAIN_OUTPUT_ROOT}/${DATASET}/${MODEL_TAG}/${budget_tag}/${VARIANT}/seed_${seed}/metrics.json"
  selection_log="${RUN_LOG_DIR}/${budget_tag}_seed${seed}_select.log"
  train_log="${RUN_LOG_DIR}/${budget_tag}_seed${seed}_train.log"

  if [[ ! -f "${selected_indices_path}" ]]; then
    if [[ "${ENABLE_WAVELET_MULTISCALE}" == "1" ]]; then
      selection_extra_args+=(--use_wavelet_multiscale)
    fi
    if [[ "${ENABLE_LSRC}" == "1" ]]; then
      selection_extra_args+=(--enable_lsrc)
    fi
    if [[ "${USE_DPP}" == "1" ]]; then
      selection_extra_args+=(--use_dpp)
    fi
    if [[ "${LSRC_USE_GLOBAL_CONFIDENCE}" == "1" ]]; then
      selection_extra_args+=(--lsrc_use_global_confidence)
    fi
    if [[ "${SWD_USE_FIXED_PROJECTIONS}" == "1" ]]; then
      selection_extra_args+=(--swd_use_fixed_projections)
    fi

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
      --swd_num_projections "${SWD_NUM_PROJECTIONS}" \
      --swd_p "${SWD_P}" \
      --wavelet_scales "${WAVELET_SCALES}" \
      --wavelet_loss_weight "${WAVELET_LOSS_WEIGHT}" \
      --wavelet_distance_type "${WAVELET_DISTANCE_TYPE}" \
      --wavelet_swd_num_projections "${WAVELET_SWD_NUM_PROJECTIONS}" \
      --wavelet_swd_p "${WAVELET_SWD_P}" \
      --wavelet_schedule "${WAVELET_SCHEDULE}" \
      --lambda_div "${PROXY_LAMBDA_DIV}" \
      --lambda_match "${PROXY_LAMBDA_MATCH}" \
      --lambda_graph "${PROXY_LAMBDA_GRAPH}" \
      --diversity_sigma "${PROXY_DIVERSITY_SIGMA}" \
      --lambda_diff "${LAMBDA_DIFF}" \
      --lambda_ms "${LAMBDA_MS}" \
      --lambda_lsrc "${LAMBDA_LSRC}" \
      --lsrc_mu "${LSRC_MU}" \
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
      --cost_gamma_topo "${COST_GAMMA_TOPO}" \
      --cost_eta_lsrc "${COST_ETA_LSRC}" \
      "${selection_extra_args[@]}" \
      > "${selection_log}" 2>&1
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

cd "${PROJECT_ROOT}"
run_precompute_if_needed

stage_log "B1 MS-SWD absolute-budget sweep start: dataset=${DATASET} budgets=${BUDGETS[*]} seeds=${SEEDS[*]} variant=${VARIANT}"
for budget in "${BUDGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one_budget_seed "${budget}" "${seed}"
  done
done

python "${PROJECT_ROOT}/tools/aggregate_main_table_metrics.py" \
  --subset_train_root "${TRAIN_OUTPUT_ROOT}" \
  --output_root "${REPORT_ROOT}" \
  --report_name "${REPORT_NAME}" \
  --datasets "${DATASET}" \
  --backbone "${BACKBONE}" \
  --methods "${VARIANT}" \
  --budget_sizes "${BUDGETS[@]}" \
  --seeds "${SEEDS[@]}"

stage_log "B1 MS-SWD absolute-budget sweep completed. Logs saved to ${RUN_LOG_DIR}"
