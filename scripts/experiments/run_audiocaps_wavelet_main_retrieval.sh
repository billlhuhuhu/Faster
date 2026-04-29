#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET="audiocaps"
AUDIOCAPS_ROOT="${AUDIOCAPS_ROOT:-/home/hzx/Faster/data/AudioCaps}"
AUDIO_ENCODER="${AUDIOCAPS_AUDIO_ENCODER:-logmel_stats}"
TEXT_ENCODER="${AUDIOCAPS_TEXT_ENCODER:-bert}"
TEXT_FEATURE_MODE="${AUDIOCAPS_TEXT_FEATURE_MODE:-bert}"
VARIANT="${AUDIOCAPS_VARIANT:-wavelet_main_audiocaps_logmel}"
RUN_OURS="${AUDIOCAPS_RUN_OURS:-1}"
RUN_RANDOM="${AUDIOCAPS_RUN_RANDOM:-1}"
RUN_FULL="${AUDIOCAPS_RUN_FULL:-1}"
SEEDS_STR="${AUDIOCAPS_SEEDS:-0}"
read -r -a SEEDS <<< "${SEEDS_STR}"
BUDGETS_STR="${AUDIOCAPS_BUDGETS:-100 200 500}"
read -r -a BUDGETS <<< "${BUDGETS_STR}"
RATIOS_STR="${AUDIOCAPS_RATIOS:-0.01 0.02 0.03}"
read -r -a RATIOS <<< "${RATIOS_STR}"

FEATURE_CACHE_ROOT="${AUDIOCAPS_FEATURE_CACHE_ROOT:-artifacts_audiocaps/feature_cache}"
TOPOLOGY_ROOT="${AUDIOCAPS_TOPOLOGY_ROOT:-artifacts_audiocaps/topology_graph}"
CROSS_OUTPUT_ROOT="${AUDIOCAPS_CROSS_OUTPUT_ROOT:-artifacts_audiocaps/cross_modal_topology}"
SELECTION_OUTPUT_ROOT="${AUDIOCAPS_SELECTION_OUTPUT_ROOT:-artifacts_audiocaps/subset_selection}"
RANDOM_SELECTION_OUTPUT_ROOT="${AUDIOCAPS_RANDOM_SELECTION_OUTPUT_ROOT:-artifacts_audiocaps/subset_selection_random}"
RETRIEVAL_OUTPUT_ROOT="${AUDIOCAPS_RETRIEVAL_OUTPUT_ROOT:-artifacts_audiocaps/subset_retrieval}"
REPORT_ROOT="${AUDIOCAPS_REPORT_ROOT:-artifacts_audiocaps/reports}"
EXPERIMENT_LOG_ROOT="${AUDIOCAPS_LOG_ROOT:-artifacts_audiocaps/experiment_logs}"

AUDIO_SAMPLE_RATE="${AUDIOCAPS_AUDIO_SAMPLE_RATE:-16000}"
AUDIO_N_MELS="${AUDIOCAPS_AUDIO_N_MELS:-64}"
AUDIO_N_FFT="${AUDIOCAPS_AUDIO_N_FFT:-1024}"
AUDIO_HOP_LENGTH="${AUDIOCAPS_AUDIO_HOP_LENGTH:-320}"
AUDIO_MAX_DURATION_SEC="${AUDIOCAPS_AUDIO_MAX_DURATION_SEC:-10.0}"

K_NEIGHBORS="${AUDIOCAPS_K_NEIGHBORS:-15}"
TOPOLOGY_METRIC_AUDIO="${AUDIOCAPS_TOPOLOGY_METRIC_AUDIO:-euclidean}"
TOPOLOGY_METRIC_TEXT="${AUDIOCAPS_TOPOLOGY_METRIC_TEXT:-cosine}"
TOPOLOGY_KNN_BACKEND="${AUDIOCAPS_TOPOLOGY_KNN_BACKEND:-auto}"
TOPOLOGY_N_JOBS="${AUDIOCAPS_TOPOLOGY_N_JOBS:-32}"
TOPOLOGY_GRAPH_REDUCE_METHOD="${AUDIOCAPS_TOPOLOGY_GRAPH_REDUCE_METHOD:-pca}"
TOPOLOGY_GRAPH_FEATURE_DIM="${AUDIOCAPS_TOPOLOGY_GRAPH_FEATURE_DIM:-256}"

CORRECTION_MODE="${AUDIOCAPS_CORRECTION_MODE:-bidirectional}"
FUSION_MODE="${AUDIOCAPS_FUSION_MODE:-confidence_aware}"
FUSION_DOMAIN_MODE="${AUDIOCAPS_FUSION_DOMAIN_MODE:-wavelet_latent}"
WAVELET_FUSION_WEIGHT_MODE="${AUDIOCAPS_WAVELET_FUSION_WEIGHT_MODE:-collapse_aware}"
WAVELET_FUSION_ENTROPY_TEMPERATURE="${AUDIOCAPS_WAVELET_FUSION_ENTROPY_TEMPERATURE:-1.0}"

PROXY_LOSS_TYPE="${AUDIOCAPS_PROXY_LOSS_TYPE:-wavelet_main}"
PROXY_INIT_METHOD="${AUDIOCAPS_PROXY_INIT_METHOD:-kmeans}"
PROXY_LR="${AUDIOCAPS_PROXY_LR:-0.05}"
PROXY_NUM_STEPS="${AUDIOCAPS_PROXY_NUM_STEPS:-200}"
PROXY_TARGET_BATCH_SIZE="${AUDIOCAPS_PROXY_TARGET_BATCH_SIZE:-2048}"
PROXY_BATCH_SIZE="${AUDIOCAPS_PROXY_BATCH_SIZE:-2048}"
LSRC_BATCH_SIZE="${AUDIOCAPS_LSRC_BATCH_SIZE:-2048}"
MATCHING_TOP_K="${AUDIOCAPS_MATCHING_TOP_K:-64}"
MATCHING_CANDIDATE_BATCH_SIZE="${AUDIOCAPS_MATCHING_CANDIDATE_BATCH_SIZE:-128}"
LAMBDA_MAIN="${AUDIOCAPS_LAMBDA_MAIN:-1.0}"
LAMBDA_LSRC="${AUDIOCAPS_LAMBDA_LSRC:-0.1}"
LAMBDA_REG="${AUDIOCAPS_LAMBDA_REG:-1.0}"
WAVELET_MAIN_SCALES="${AUDIOCAPS_WAVELET_MAIN_SCALES:-1,2,4}"
WAVELET_MAIN_SWD_NUM_PROJECTIONS="${AUDIOCAPS_WAVELET_MAIN_SWD_NUM_PROJECTIONS:-64}"
WAVELET_COV_WEIGHT="${AUDIOCAPS_WAVELET_COV_WEIGHT:-0.5}"
WAVELET_EDGE_WEIGHT="${AUDIOCAPS_WAVELET_EDGE_WEIGHT:-0.25}"
WAVELET_CURRICULUM_SCHEDULE="${AUDIOCAPS_WAVELET_CURRICULUM_SCHEDULE:-coarse_to_fine}"
SELECTION_USE_TORCHRUN="${AUDIOCAPS_SELECTION_USE_TORCHRUN:-0}"
SELECTION_CUDA_VISIBLE_DEVICES="${AUDIOCAPS_SELECTION_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${AUDIOCAPS_SELECTION_NPROC_PER_NODE:-}" && -n "${SELECTION_CUDA_VISIBLE_DEVICES}" ]]; then
  SELECTION_NPROC_PER_NODE="$(python - <<PY
devices = "${SELECTION_CUDA_VISIBLE_DEVICES}".strip()
print(max(len([item for item in devices.split(",") if item.strip()]), 1))
PY
)"
else
  SELECTION_NPROC_PER_NODE="${AUDIOCAPS_SELECTION_NPROC_PER_NODE:-1}"
fi

RETRIEVAL_EPOCHS="${AUDIOCAPS_RETRIEVAL_EPOCHS:-30}"
RETRIEVAL_BATCH_SIZE="${AUDIOCAPS_RETRIEVAL_BATCH_SIZE:-512}"
RETRIEVAL_EVAL_BATCH_SIZE="${AUDIOCAPS_RETRIEVAL_EVAL_BATCH_SIZE:-2048}"
RETRIEVAL_LR="${AUDIOCAPS_RETRIEVAL_LR:-1e-3}"
RETRIEVAL_EMBED_DIM="${AUDIOCAPS_RETRIEVAL_EMBED_DIM:-256}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${EXPERIMENT_LOG_ROOT}/${VARIANT}_${RUN_TIMESTAMP}"
REPORT_DIR="${REPORT_ROOT}/${VARIANT}_${RUN_TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}" "${REPORT_DIR}"

MODEL_TAG="$(sanitize_component "${AUDIO_ENCODER}")_$(sanitize_component "${TEXT_ENCODER}")"
FUSION_TAG="k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_AUDIO}")_a$(sanitize_component "${ALPHA}")"
TRAIN_FEATURE_DIR="${FEATURE_CACHE_ROOT}/${DATASET}/train/${MODEL_TAG}"
EVAL_SPLIT="${AUDIOCAPS_EVAL_SPLIT:-}"
if [[ -z "${EVAL_SPLIT}" ]]; then
  EVAL_SPLIT="$(python - <<PY
from src.audiocaps_dataset import find_annotation_file
root = "${AUDIOCAPS_ROOT}"
for split in ["test", "val", "validation"]:
    try:
        find_annotation_file(root, split)
        print(split)
        break
    except Exception:
        pass
else:
    print("test")
PY
)"
fi
EVAL_FEATURE_DIR="${FEATURE_CACHE_ROOT}/${DATASET}/${EVAL_SPLIT}/${MODEL_TAG}"
IMAGE_TOPOLOGY_DIR="${TOPOLOGY_ROOT}/${DATASET}/train/${MODEL_TAG}/image/k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_AUDIO}")"
TEXT_TOPOLOGY_DIR="${TOPOLOGY_ROOT}/${DATASET}/train/${MODEL_TAG}/text/k${K_NEIGHBORS}_$(sanitize_component "${TOPOLOGY_METRIC_TEXT}")"
CROSS_SUMMARY_PATH="${CROSS_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${FUSION_TAG}/summary.json"

format_ratio_tag_local() {
  local ratio="$1"
  python - <<PY
ratio = float("${ratio}")
print(f"ratio_{int(round(ratio * 100)):02d}")
PY
}

stage_log "AudioCaps wavelet-main retrieval start: root=${AUDIOCAPS_ROOT} eval_split=${EVAL_SPLIT} budgets=${BUDGETS_STR} ratios=${RATIOS_STR} ours=${RUN_OURS} random=${RUN_RANDOM} full=${RUN_FULL}"

build_feature_cache() {
  local split="$1"
  local feature_dir="${FEATURE_CACHE_ROOT}/${DATASET}/${split}/${MODEL_TAG}"
  if [[ -f "${feature_dir}/img_features_selection.pt" && -f "${feature_dir}/txt_features_selection.pt" ]]; then
    stage_log "Skip AudioCaps feature cache: existing ${feature_dir}"
    return 0
  fi
  stage_log "AudioCaps feature cache start: split=${split}"
  python "${PROJECT_ROOT}/run_audiocaps_feature_cache.py" \
    --data_root "${AUDIOCAPS_ROOT}" \
    --split "${split}" \
    --cache_root "${FEATURE_CACHE_ROOT}" \
    --audio_encoder "${AUDIO_ENCODER}" \
    --text_encoder "${TEXT_ENCODER}" \
    --audio_feature_mode logmel_stats \
    --text_feature_mode "${TEXT_FEATURE_MODE}" \
    --audio_sample_rate "${AUDIO_SAMPLE_RATE}" \
    --audio_n_mels "${AUDIO_N_MELS}" \
    --audio_n_fft "${AUDIO_N_FFT}" \
    --audio_hop_length "${AUDIO_HOP_LENGTH}" \
    --audio_max_duration_sec "${AUDIO_MAX_DURATION_SEC}" \
    --text_batch_size "${SELECTION_TEXT_BATCH_SIZE}" \
    --device "${DEVICE}" \
    > "${RUN_LOG_DIR}/feature_cache_${split}.log" 2>&1
  stage_log "AudioCaps feature cache done: split=${split}"
}

run_precompute_if_needed() {
  build_feature_cache train
  build_feature_cache "${EVAL_SPLIT}"

  if [[ ! -f "${IMAGE_TOPOLOGY_DIR}/summary.json" ]]; then
    stage_log "Audio topology graph start"
    python "${PROJECT_ROOT}/run_topology_graph.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${AUDIO_ENCODER}" \
      --text_encoder "${TEXT_ENCODER}" \
      --modality image \
      --feature_cache_root "${FEATURE_CACHE_ROOT}" \
      --output_root "${TOPOLOGY_ROOT}" \
      --metric "${TOPOLOGY_METRIC_AUDIO}" \
      --knn_k "${K_NEIGHBORS}" \
      --graph_reduce_method "${TOPOLOGY_GRAPH_REDUCE_METHOD}" \
      --graph_feature_dim "${TOPOLOGY_GRAPH_FEATURE_DIM}" \
      --num_eigs 32 \
      --spectral_embedding_dim 32 \
      --n_jobs "${TOPOLOGY_N_JOBS}" \
      --knn_backend "${TOPOLOGY_KNN_BACKEND}" \
      > "${RUN_LOG_DIR}/topology_audio.log" 2>&1
    stage_log "Audio topology graph done"
  else
    stage_log "Skip audio topology graph: ${IMAGE_TOPOLOGY_DIR}"
  fi

  if [[ ! -f "${TEXT_TOPOLOGY_DIR}/summary.json" ]]; then
    stage_log "Text topology graph start"
    python "${PROJECT_ROOT}/run_topology_graph.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${AUDIO_ENCODER}" \
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
      > "${RUN_LOG_DIR}/topology_text.log" 2>&1
    stage_log "Text topology graph done"
  else
    stage_log "Skip text topology graph: ${TEXT_TOPOLOGY_DIR}"
  fi

  if [[ ! -f "${CROSS_SUMMARY_PATH}" ]]; then
    stage_log "Cross-modal topology start"
    python "${PROJECT_ROOT}/run_cross_modal_topology.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${AUDIO_ENCODER}" \
      --text_encoder "${TEXT_ENCODER}" \
      --topology_root "${TOPOLOGY_ROOT}" \
      --output_root "${CROSS_OUTPUT_ROOT}" \
      --metric "${TOPOLOGY_METRIC_AUDIO}" \
      --image_metric "${TOPOLOGY_METRIC_AUDIO}" \
      --text_metric "${TOPOLOGY_METRIC_TEXT}" \
      --k "${K_NEIGHBORS}" \
      --alpha "${ALPHA}" \
      --correction_mode "${CORRECTION_MODE}" \
      --correction_score_mode "${CORRECTION_SCORE_MODE}" \
      --collapse_score_mode "${COLLAPSE_SCORE_MODE}" \
      --collapse_neighbor_topk "${COLLAPSE_NEIGHBOR_TOPK}" \
      --asymmetric_correction_lambda "${ASYMMETRIC_CORRECTION_LAMBDA}" \
      --corrected_image_added_topk "${CORRECTED_IMAGE_ADDED_TOPK}" \
      --fusion_domain_mode "${FUSION_DOMAIN_MODE}" \
      --fusion_mode "${FUSION_MODE}" \
      --wavelet_fusion_scales "${WAVELET_FUSION_SCALES}" \
      --wavelet_fusion_impl "${WAVELET_FUSION_IMPL}" \
      --wavelet_fusion_probe_mode "${WAVELET_FUSION_PROBE_MODE}" \
      --wavelet_fusion_probe_dim "${WAVELET_FUSION_PROBE_DIM}" \
      --wavelet_fusion_weight_mode "${WAVELET_FUSION_WEIGHT_MODE}" \
      --wavelet_fusion_entropy_temperature "${WAVELET_FUSION_ENTROPY_TEMPERATURE}" \
      --wavelet_latent_postprocess_topk "${WAVELET_LATENT_POSTPROCESS_TOPK}" \
      --num_eigs "${CROSS_MODAL_NUM_EIGS}" \
      --spectral_embedding_dim "${CROSS_MODAL_EMBED_DIM}" \
      > "${RUN_LOG_DIR}/cross_modal.log" 2>&1
    stage_log "Cross-modal topology done"
  else
    stage_log "Skip cross-modal topology: ${CROSS_SUMMARY_PATH}"
  fi
}

run_retrieval_with_indices() {
  local subset_mode="$1"
  local budget_tag="$2"
  local seed="$3"
  local selected_indices_path="$4"
  local retrieval_dir="${RETRIEVAL_OUTPUT_ROOT}/${DATASET}/${MODEL_TAG}/${budget_tag}/${subset_mode}_${VARIANT}/seed_${seed}"
  local retrieval_log="${RUN_LOG_DIR}/${budget_tag}_${subset_mode}_seed${seed}_retrieval.log"

  if [[ ! -f "${retrieval_dir}/metrics.json" ]]; then
    stage_log "AudioCaps retrieval start: mode=${subset_mode} ${budget_tag} seed=${seed}"
    local retrieval_args=(
      --train_feature_dir "${TRAIN_FEATURE_DIR}"
      --eval_feature_dir "${EVAL_FEATURE_DIR}"
      --output_dir "${retrieval_dir}"
      --subset_mode "${subset_mode}"
      --embed_dim "${RETRIEVAL_EMBED_DIM}"
      --epochs "${RETRIEVAL_EPOCHS}"
      --batch_size "${RETRIEVAL_BATCH_SIZE}"
      --eval_batch_size "${RETRIEVAL_EVAL_BATCH_SIZE}"
      --lr "${RETRIEVAL_LR}"
      --device "${DEVICE}"
    )
    if [[ -n "${selected_indices_path}" ]]; then
      retrieval_args+=(--selected_indices_path "${selected_indices_path}")
    fi
    python "${PROJECT_ROOT}/run_audiocaps_subset_retrieval.py" "${retrieval_args[@]}" \
      > "${retrieval_log}" 2>&1
    stage_log "AudioCaps retrieval done: mode=${subset_mode} ${budget_tag} seed=${seed}"
  else
    stage_log "Skip retrieval: existing ${retrieval_dir}/metrics.json"
  fi
}

run_ours_selection_and_retrieval() {
  local budget_arg="$1"
  local budget_value="$2"
  local budget_tag="$3"
  local seed="$4"
  local selected_indices_path="${SELECTION_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${budget_tag}/proxy_opt_lsrc/seed_${seed}/selected_indices.json"
  local selection_log="${RUN_LOG_DIR}/${budget_tag}_seed${seed}_select.log"

  if [[ ! -f "${selected_indices_path}" ]]; then
    stage_log "Selection start: ${budget_tag} seed=${seed}"
    local launcher=(python)
    if [[ "${SELECTION_USE_TORCHRUN}" == "1" && "${SELECTION_NPROC_PER_NODE}" -gt 1 ]]; then
      launcher=(torchrun --standalone --nproc_per_node "${SELECTION_NPROC_PER_NODE}")
    fi
    "${launcher[@]}" "${PROJECT_ROOT}/run_subset_selection.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${AUDIO_ENCODER}" \
      --text_encoder "${TEXT_ENCODER}" \
      --feature_cache_root "${FEATURE_CACHE_ROOT}" \
      --cross_modal_root "${CROSS_OUTPUT_ROOT}" \
      --output_root "${SELECTION_OUTPUT_ROOT}" \
      --metric "${TOPOLOGY_METRIC_AUDIO}" \
      --k "${K_NEIGHBORS}" \
      --alpha "${ALPHA}" \
      "${budget_arg}" "${budget_value}" \
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
      --proxy_target_batch_size "${PROXY_TARGET_BATCH_SIZE}" \
      --proxy_batch_size "${PROXY_BATCH_SIZE}" \
      --lambda_main "${LAMBDA_MAIN}" \
      --wavelet_main_scales "${WAVELET_MAIN_SCALES}" \
      --wavelet_main_swd_num_projections "${WAVELET_MAIN_SWD_NUM_PROJECTIONS}" \
      --wavelet_cov_weight "${WAVELET_COV_WEIGHT}" \
      --wavelet_edge_weight "${WAVELET_EDGE_WEIGHT}" \
      --wavelet_curriculum_schedule "${WAVELET_CURRICULUM_SCHEDULE}" \
      --lambda_lsrc "${LAMBDA_LSRC}" \
      --lambda_reg "${LAMBDA_REG}" \
      --lsrc_k "${LSRC_K}" \
      --lsrc_batch_size "${LSRC_BATCH_SIZE}" \
      --matching_top_k "${MATCHING_TOP_K}" \
      --matching_candidate_batch_size "${MATCHING_CANDIDATE_BATCH_SIZE}" \
      --enable_lsrc \
      --keep_lsrc \
      --use_dpp \
      > "${selection_log}" 2>&1
    stage_log "Selection done: ${budget_tag} seed=${seed}"
  else
    stage_log "Skip selection: existing ${selected_indices_path}"
  fi

  run_retrieval_with_indices "ours" "${budget_tag}" "${seed}" "${selected_indices_path}"
}

run_random_selection_and_retrieval() {
  local budget_arg="$1"
  local budget_value="$2"
  local budget_tag="$3"
  local seed="$4"
  local selected_indices_path="${RANDOM_SELECTION_OUTPUT_ROOT}/${DATASET}/train/${MODEL_TAG}/${budget_tag}/random/seed_${seed}/selected_indices.json"
  local selection_log="${RUN_LOG_DIR}/${budget_tag}_random_seed${seed}_select.log"

  if [[ ! -f "${selected_indices_path}" ]]; then
    stage_log "Random selection start: ${budget_tag} seed=${seed}"
    python "${PROJECT_ROOT}/run_random_subset_selection.py" \
      --dataset "${DATASET}" \
      --split train \
      --image_encoder "${AUDIO_ENCODER}" \
      --text_encoder "${TEXT_ENCODER}" \
      --feature_cache_root "${FEATURE_CACHE_ROOT}" \
      --output_root "${RANDOM_SELECTION_OUTPUT_ROOT}" \
      "${budget_arg}" "${budget_value}" \
      --selection_method random \
      --random_state "${seed}" \
      > "${selection_log}" 2>&1
    stage_log "Random selection done: ${budget_tag} seed=${seed}"
  else
    stage_log "Skip random selection: existing ${selected_indices_path}"
  fi

  run_retrieval_with_indices "random" "${budget_tag}" "${seed}" "${selected_indices_path}"
}

run_precompute_if_needed
if [[ "${RUN_FULL}" == "1" ]]; then
  run_retrieval_with_indices "full" "full" "0" ""
fi
for seed in "${SEEDS[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    if [[ "${RUN_OURS}" == "1" ]]; then
      run_ours_selection_and_retrieval "--budget_size" "${budget}" "$(format_budget_tag "${budget}")" "${seed}"
    fi
    if [[ "${RUN_RANDOM}" == "1" ]]; then
      run_random_selection_and_retrieval "--budget_size" "${budget}" "$(format_budget_tag "${budget}")" "${seed}"
    fi
  done
  for ratio in "${RATIOS[@]}"; do
    if [[ "${RUN_OURS}" == "1" ]]; then
      run_ours_selection_and_retrieval "--budget_ratio" "${ratio}" "$(format_ratio_tag_local "${ratio}")" "${seed}"
    fi
    if [[ "${RUN_RANDOM}" == "1" ]]; then
      run_random_selection_and_retrieval "--budget_ratio" "${ratio}" "$(format_ratio_tag_local "${ratio}")" "${seed}"
    fi
  done
done

REPORT_PATH="${REPORT_DIR}/audiocaps_retrieval_summary.csv"
python - <<PY
import csv, json
from pathlib import Path
root = Path("${RETRIEVAL_OUTPUT_ROOT}") / "${DATASET}" / "${MODEL_TAG}"
rows = []
for metrics_path in root.glob("*/*${VARIANT}/seed_*/metrics.json"):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    budget_tag = metrics_path.parents[2].name
    seed = metrics_path.parent.name.replace("seed_", "")
    rows.append({
        "dataset": "audiocaps",
        "subset_mode": data.get("subset_mode", metrics_path.parents[1].name.split("_", 1)[0]),
        "budget": budget_tag,
        "seed": seed,
        "subset_size": data.get("subset_size"),
        "a2t_r1": data.get("a2t_r1"),
        "a2t_r5": data.get("a2t_r5"),
        "a2t_r10": data.get("a2t_r10"),
        "t2a_r1": data.get("t2a_r1"),
        "t2a_r5": data.get("t2a_r5"),
        "t2a_r10": data.get("t2a_r10"),
        "mean_recall": data.get("mean_recall"),
        "metrics_path": str(metrics_path),
    })
rows.sort(key=lambda r: (r["budget"], r["seed"]))
out = Path("${REPORT_PATH}")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset","budget","seed"])
    writer.writeheader()
    writer.writerows(rows)
print(f"saved AudioCaps retrieval table: {out}")
PY

stage_log "AudioCaps wavelet-main retrieval done"
stage_log "Report: ${REPORT_PATH}"
