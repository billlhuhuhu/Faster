#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASETS=("flickr" "coco")
METHODS=("ours_baseline" "ours_full")
SEEDS=("${SEEDS_DEFAULT[@]}")
BUDGETS=("${BUDGETS_ABS_DEFAULT[@]}")
BACKBONE="nfnet"
TEXT_ENCODER="bert"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/main_table_abs_${RUN_TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

run_precompute_for_dataset() {
  local dataset="$1"
  local image_root
  local topology_extra_args=()
  image_root="$(get_image_root "${dataset}")"

  if [[ -n "${TOPOLOGY_MULTI_SCALE_KS}" ]]; then
    topology_extra_args+=(--multi_scale_ks "${TOPOLOGY_MULTI_SCALE_KS}")
  fi
  if [[ "${TOPOLOGY_FAISS_USE_GPU}" == "1" ]]; then
    topology_extra_args+=(--faiss_use_gpu)
  fi
  if [[ "${TOPOLOGY_USE_MST_CONNECTIVITY}" == "1" ]]; then
    topology_extra_args+=(--use_mst_connectivity)
  fi

  stage_log "Precompute start: dataset=${dataset}"

  python "${PROJECT_ROOT}/run_feature_cache.py" \
    --dataset "${dataset}" \
    --image_encoder "${BACKBONE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --selection_image_repr_method "${SELECTION_IMAGE_REPR_METHOD}" \
    --selection_text_repr_method "${SELECTION_TEXT_REPR_METHOD}" \
    --image_root "${image_root}" \
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
    --device "${DEVICE}"

  python "${PROJECT_ROOT}/run_topology_graph.py" \
    --dataset "${dataset}" \
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
    "${topology_extra_args[@]}"

  python "${PROJECT_ROOT}/run_topology_graph.py" \
    --dataset "${dataset}" \
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
    "${topology_extra_args[@]}"

  python "${PROJECT_ROOT}/run_cross_modal_topology.py" \
    --dataset "${dataset}" \
    --split train \
    --image_encoder "${BACKBONE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --topology_root "${TOPOLOGY_ROOT}" \
    --output_root "${CROSS_MODAL_ROOT}" \
    --metric "${TOPOLOGY_METRIC_IMAGE}" \
    --image_metric "${TOPOLOGY_METRIC_IMAGE}" \
    --text_metric "${TOPOLOGY_METRIC_TEXT}" \
    --k "${K_NEIGHBORS}" \
    --alpha "${ALPHA}" \
    --num_eigs "${CROSS_MODAL_NUM_EIGS}" \
    --spectral_embedding_dim "${CROSS_MODAL_EMBED_DIM}"

  stage_log "Precompute done: dataset=${dataset}"
}

run_one_experiment() {
  local dataset="$1"
  local method_name="$2"
  local budget="$3"
  local seed="$4"
  local selection_method="$5"
  local image_root
  local budget_tag
  local selected_indices_path
  local train_log
  local select_log
  local selection_extra_args=()
  local train_extra_args=()

  image_root="$(get_image_root "${dataset}")"
  budget_tag="$(format_budget_tag "${budget}")"
  selected_indices_path="${SUBSET_SELECTION_ROOT}/${dataset}/train/${BACKBONE}_${TEXT_ENCODER}/${budget_tag}/${selection_method}/seed_${seed}/selected_indices.json"
  select_log="${RUN_LOG_DIR}/${dataset}_${method_name}_${budget_tag}_seed${seed}_select.log"
  train_log="${RUN_LOG_DIR}/${dataset}_${method_name}_${budget_tag}_seed${seed}_train.log"

  if [[ "${PROXY_USE_PDCFD}" == "1" ]]; then
    selection_extra_args+=(--use_pdcfd)
  fi
  if [[ "${PROXY_USE_PDAS}" == "1" ]]; then
    selection_extra_args+=(--use_pdas)
  fi
  if [[ "${PROXY_USE_DPP}" == "1" ]]; then
    selection_extra_args+=(--use_dpp)
  fi
  if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
    train_extra_args+=(--no_aug)
  fi

  stage_log "Selection start: dataset=${dataset} method=${method_name} budget=${budget} seed=${seed}"
  python "${PROJECT_ROOT}/run_subset_selection.py" \
    --dataset "${dataset}" \
    --split train \
    --image_encoder "${BACKBONE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --feature_cache_root "${FEATURE_CACHE_ROOT}" \
    --cross_modal_root "${CROSS_MODAL_ROOT}" \
    --output_root "${SUBSET_SELECTION_ROOT}" \
    --metric "${TOPOLOGY_METRIC_IMAGE}" \
    --k "${K_NEIGHBORS}" \
    --alpha "${ALPHA}" \
    --budget_size "${budget}" \
    --selection_method "${selection_method}" \
    --reference_embedding_mode "${SUBSET_REFERENCE_EMBEDDING_MODE}" \
    --spectral_weight "${SUBSET_SPECTRAL_WEIGHT}" \
    --random_state "${seed}" \
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
    "${selection_extra_args[@]}" \
    > "${select_log}" 2>&1
  stage_log "Selection done: dataset=${dataset} method=${method_name} budget=${budget} seed=${seed}"

  stage_log "Training start: dataset=${dataset} method=${method_name} budget=${budget} seed=${seed}"
  python "${PROJECT_ROOT}/run_subset_train.py" \
    --dataset "${dataset}" \
    --image_root "${image_root}" \
    --ann_root "${ANN_ROOT}" \
    --selected_indices_path "${selected_indices_path}" \
    --subset_size "${budget}" \
    --subset_tag "${method_name}" \
    --image_encoder "${BACKBONE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --output_root "${SUBSET_TRAIN_ROOT}" \
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
  stage_log "Training done: dataset=${dataset} method=${method_name} budget=${budget} seed=${seed}"
}

for dataset in "${DATASETS[@]}"; do
  run_precompute_for_dataset "${dataset}"
done

for dataset in "${DATASETS[@]}"; do
  for method_name in "${METHODS[@]}"; do
    selection_method="$(selection_method_from_name "${method_name}")"
    if [[ -z "${selection_method}" ]]; then
      echo "Unsupported method mapping: ${method_name}" >&2
      exit 1
    fi
    for budget in "${BUDGETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one_experiment "${dataset}" "${method_name}" "${budget}" "${seed}" "${selection_method}"
      done
    done
  done
done

python "${PROJECT_ROOT}/tools/aggregate_main_table_metrics.py" \
  --subset_train_root "${SUBSET_TRAIN_ROOT}" \
  --output_root "${REPORT_ROOT}" \
  --report_name "main_table_abs" \
  --datasets "${DATASETS[@]}" \
  --backbone "${BACKBONE}" \
  --methods "${METHODS[@]}" \
  --budget_sizes "${BUDGETS[@]}" \
  --seeds "${SEEDS[@]}"

stage_log "Main-table absolute-budget runs completed. Logs saved to ${RUN_LOG_DIR}"
