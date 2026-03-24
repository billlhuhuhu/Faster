#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="${1:?usage: run_pipeline.sh <dataset> <image_encoder> <subset_ratio> <seed> <method>}"
IMAGE_ENCODER="${2:?usage: run_pipeline.sh <dataset> <image_encoder> <subset_ratio> <seed> <method>}"
SUBSET_RATIO="${3:?usage: run_pipeline.sh <dataset> <image_encoder> <subset_ratio> <seed> <method>}"
SEED="${4:?usage: run_pipeline.sh <dataset> <image_encoder> <subset_ratio> <seed> <method>}"
METHOD="${5:?usage: run_pipeline.sh <dataset> <image_encoder> <subset_ratio> <seed> <method>}"

SELECTION_METHOD="$(selection_method_from_name "$METHOD")"
if [[ -z "${SELECTION_METHOD}" ]]; then
  echo "Method ${METHOD} is not runnable by this script yet." >&2
  echo "Supported methods: ours_baseline, ours_full" >&2
  exit 2
fi

IMAGE_ROOT="$(get_image_root "$DATASET")"
MODEL_TAG="$(sanitize_component "${IMAGE_ENCODER}")_bert"
RATIO_INT="$(python - <<PY
ratio = float("${SUBSET_RATIO}")
print(int(round(ratio * 100)))
PY
)"
RATIO_TAG="ratio_$(printf '%02d' "${RATIO_INT}")"

cd "$PROJECT_ROOT"

stage_log "Stage 2/8 feature cache: dataset=${DATASET} backbone=${IMAGE_ENCODER}"
python run_feature_cache.py \
  --dataset "$DATASET" \
  --image_encoder "$IMAGE_ENCODER" \
  --text_encoder bert \
  --selection_image_repr_method "$SELECTION_IMAGE_REPR_METHOD" \
  --selection_text_repr_method "$SELECTION_TEXT_REPR_METHOD" \
  --selection_image_size "$SELECTION_IMAGE_SIZE" \
  --selection_raw_resize_size "$SELECTION_RAW_RESIZE_SIZE" \
  --selection_raw_pca_dim "$SELECTION_RAW_PCA_DIM" \
  --selection_image_batch_size "$SELECTION_IMAGE_BATCH_SIZE" \
  --selection_text_batch_size "$SELECTION_TEXT_BATCH_SIZE" \
  --selection_random_state "$SELECTION_RANDOM_STATE" \
  --hog_orientations "$HOG_ORIENTATIONS" \
  --hog_pixels_per_cell "$HOG_PIXELS_PER_CELL" \
  --hog_cells_per_block "$HOG_CELLS_PER_BLOCK" \
  --color_hist_bins "$COLOR_HIST_BINS" \
  --color_space "$COLOR_SPACE" \
  --image_root "$IMAGE_ROOT" \
  --ann_root "$ANN_ROOT" \
  --cache_root "$FEATURE_CACHE_ROOT" \
  --batch_size "$BATCH_FEATURE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE"
stage_log "Stage 2/8 feature cache done"

stage_log "Stage 3/8 topology graph (image): metric=${TOPOLOGY_METRIC_IMAGE}"
IMAGE_TOPOLOGY_CMD=(
  python run_topology_graph.py
  --dataset "$DATASET"
  --split train
  --image_encoder "$IMAGE_ENCODER"
  --text_encoder bert
  --modality image
  --feature_cache_root "$FEATURE_CACHE_ROOT"
  --output_root "$TOPOLOGY_ROOT"
  --metric "$TOPOLOGY_METRIC_IMAGE"
  --k "$K_NEIGHBORS"
  --num_eigs 32
  --spectral_embedding_dim "$CROSS_MODAL_EMBED_DIM"
  --n_jobs "$TOPOLOGY_N_JOBS"
  --knn_backend "$TOPOLOGY_KNN_BACKEND"
  --graph_reduce_method "$TOPOLOGY_GRAPH_REDUCE_METHOD"
  --graph_feature_dim "$TOPOLOGY_GRAPH_FEATURE_DIM"
)
if [[ -n "${TOPOLOGY_MULTI_SCALE_KS}" ]]; then
  IMAGE_TOPOLOGY_CMD+=(--multi_scale_ks "$TOPOLOGY_MULTI_SCALE_KS")
fi
if [[ "${TOPOLOGY_USE_MST_CONNECTIVITY}" == "1" ]]; then
  IMAGE_TOPOLOGY_CMD+=(--use_mst_connectivity --mst_weight_scale "$TOPOLOGY_MST_WEIGHT_SCALE")
fi
if [[ "${TOPOLOGY_FAISS_USE_GPU}" == "1" ]]; then
  IMAGE_TOPOLOGY_CMD+=(--faiss_use_gpu)
fi
"${IMAGE_TOPOLOGY_CMD[@]}"
stage_log "Stage 3/8 topology graph (image) done"

stage_log "Stage 3/8 topology graph (text): metric=${TOPOLOGY_METRIC_TEXT}"
TEXT_TOPOLOGY_CMD=(
  python run_topology_graph.py
  --dataset "$DATASET"
  --split train
  --image_encoder "$IMAGE_ENCODER"
  --text_encoder bert
  --modality text
  --feature_cache_root "$FEATURE_CACHE_ROOT"
  --output_root "$TOPOLOGY_ROOT"
  --metric "$TOPOLOGY_METRIC_TEXT"
  --k "$K_NEIGHBORS"
  --num_eigs 32
  --spectral_embedding_dim "$CROSS_MODAL_EMBED_DIM"
  --n_jobs "$TOPOLOGY_N_JOBS"
  --knn_backend "$TOPOLOGY_KNN_BACKEND"
  --graph_reduce_method "$TOPOLOGY_GRAPH_REDUCE_METHOD"
  --graph_feature_dim "$TOPOLOGY_GRAPH_FEATURE_DIM"
)
if [[ -n "${TOPOLOGY_MULTI_SCALE_KS}" ]]; then
  TEXT_TOPOLOGY_CMD+=(--multi_scale_ks "$TOPOLOGY_MULTI_SCALE_KS")
fi
if [[ "${TOPOLOGY_USE_MST_CONNECTIVITY}" == "1" ]]; then
  TEXT_TOPOLOGY_CMD+=(--use_mst_connectivity --mst_weight_scale "$TOPOLOGY_MST_WEIGHT_SCALE")
fi
if [[ "${TOPOLOGY_FAISS_USE_GPU}" == "1" ]]; then
  TEXT_TOPOLOGY_CMD+=(--faiss_use_gpu)
fi
"${TEXT_TOPOLOGY_CMD[@]}"
stage_log "Stage 3/8 topology graph (text) done"

stage_log "Stage 4/8 cross-modal topology"
CROSS_MODAL_CMD=(
  python run_cross_modal_topology.py
  --dataset "$DATASET"
  --split train
  --image_encoder "$IMAGE_ENCODER"
  --text_encoder bert
  --topology_root "$TOPOLOGY_ROOT"
  --output_root "$CROSS_MODAL_ROOT"
  --metric "$TOPOLOGY_METRIC_IMAGE"
  --image_metric "$TOPOLOGY_METRIC_IMAGE"
  --text_metric "$TOPOLOGY_METRIC_TEXT"
  --k "$K_NEIGHBORS"
  --alpha "$ALPHA"
  --num_eigs "$CROSS_MODAL_NUM_EIGS"
  --spectral_embedding_dim "$CROSS_MODAL_EMBED_DIM"
)
if [[ -n "${TOPOLOGY_MULTI_SCALE_KS}" ]]; then
  CROSS_MODAL_CMD+=(--multi_scale_ks "$TOPOLOGY_MULTI_SCALE_KS")
fi
"${CROSS_MODAL_CMD[@]}"
stage_log "Stage 4/8 cross-modal topology done"

stage_log "Stage 5/8 subset selection: method=${SELECTION_METHOD}"
SUBSET_SELECTION_CMD=(
  python run_subset_selection.py
  --dataset "$DATASET"
  --split train
  --image_encoder "$IMAGE_ENCODER"
  --text_encoder bert
  --feature_cache_root "$FEATURE_CACHE_ROOT"
  --cross_modal_root "$CROSS_MODAL_ROOT"
  --output_root "$SUBSET_SELECTION_ROOT"
  --metric "$TOPOLOGY_METRIC_IMAGE"
  --k "$K_NEIGHBORS"
  --alpha "$ALPHA"
  --budget_ratio "$SUBSET_RATIO"
  --selection_method "$SELECTION_METHOD"
  --device "$DEVICE"
  --reference_embedding_mode "$SUBSET_REFERENCE_EMBEDDING_MODE"
  --spectral_weight "$SUBSET_SPECTRAL_WEIGHT"
  --proxy_objective_mode "$PROXY_OBJECTIVE_MODE"
  --lambda_phase "$PROXY_LAMBDA_PHASE"
  --pdas_num_stages "$PROXY_PDAS_NUM_STAGES"
  --pdas_schedule_mode "$PROXY_PDAS_SCHEDULE"
  --num_freq_pool "$PROXY_NUM_FREQ_POOL"
  --tau_min "$PROXY_TAU_MIN"
  --tau_max "$PROXY_TAU_MAX"
  --lambda_div "$PROXY_LAMBDA_DIV"
  --lambda_match "$PROXY_LAMBDA_MATCH"
  --lambda_graph "$PROXY_LAMBDA_GRAPH"
  --diversity_sigma "$PROXY_DIVERSITY_SIGMA"
  --geometry_weight "$PROXY_GEOMETRY_WEIGHT"
  --matching_cost_mode "$MATCHING_COST_MODE"
)
if [[ "${PROXY_USE_PDAS}" == "1" ]]; then
  SUBSET_SELECTION_CMD+=(--use_pdas)
fi
if [[ "${PROXY_USE_PDCFD}" == "1" ]]; then
  SUBSET_SELECTION_CMD+=(--use_pdcfd)
fi
if [[ "${PROXY_USE_DPP}" == "1" ]]; then
  SUBSET_SELECTION_CMD+=(--use_dpp)
fi
"${SUBSET_SELECTION_CMD[@]}"
stage_log "Stage 5/8 subset selection done"

SELECTED_INDICES_PATH="${SUBSET_SELECTION_ROOT}/${DATASET}/train/${MODEL_TAG}/${RATIO_TAG}"

if [[ "$METHOD" == "ours_full" ]]; then
  SELECTED_INDICES_PATH="${SELECTED_INDICES_PATH}/proxy_opt/selected_indices.json"
else
  SELECTED_INDICES_PATH="${SELECTED_INDICES_PATH}/selected_indices.json"
fi

stage_log "Stage 6/8 selected indices resolved: ${SELECTED_INDICES_PATH}"

stage_log "Stage 7/8 subset training: selected_indices=${SELECTED_INDICES_PATH}"
if [[ "${TRAIN_NO_AUG}" == "1" ]]; then
  python run_subset_train.py \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --ann_root "$ANN_ROOT" \
    --selected_indices_path "$SELECTED_INDICES_PATH" \
    --subset_ratio "$SUBSET_RATIO" \
    --subset_tag "$METHOD" \
    --image_encoder "$IMAGE_ENCODER" \
    --text_encoder bert \
    --output_root "$SUBSET_TRAIN_ROOT" \
    --batch_size_train "$BATCH_TRAIN" \
    --batch_size_test "$BATCH_TEST" \
    --text_batch_size "$TEXT_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --eval_interval "$EVAL_INTERVAL" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --no_aug
else
  python run_subset_train.py \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --ann_root "$ANN_ROOT" \
    --selected_indices_path "$SELECTED_INDICES_PATH" \
    --subset_ratio "$SUBSET_RATIO" \
    --subset_tag "$METHOD" \
    --image_encoder "$IMAGE_ENCODER" \
    --text_encoder bert \
    --output_root "$SUBSET_TRAIN_ROOT" \
    --batch_size_train "$BATCH_TRAIN" \
    --batch_size_test "$BATCH_TEST" \
    --text_batch_size "$TEXT_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --eval_interval "$EVAL_INTERVAL" \
    --seed "$SEED" \
    --device "$DEVICE"
fi
stage_log "Stage 7/8 subset training done"
stage_log "Stage 8/8 pipeline completed: dataset=${DATASET} backbone=${IMAGE_ENCODER} ratio=${SUBSET_RATIO} seed=${SEED} method=${METHOD}"
