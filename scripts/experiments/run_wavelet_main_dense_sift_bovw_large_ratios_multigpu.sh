#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WAVELET_MAIN_LATEST_DATASET="${WAVELET_MAIN_BOVW_LARGE_DATASET:-flickr}"
export WAVELET_MAIN_LATEST_BACKBONE="${WAVELET_MAIN_BOVW_LARGE_BACKBONE:-nfnet}"
export WAVELET_MAIN_LATEST_TEXT_ENCODER="${WAVELET_MAIN_BOVW_LARGE_TEXT_ENCODER:-bert}"
export WAVELET_MAIN_LATEST_VARIANT="${WAVELET_MAIN_BOVW_LARGE_VARIANT:-wavelet_main_dense_sift_bovw_large_ratios_multigpu}"
export WAVELET_MAIN_LATEST_BUDGETS="${WAVELET_MAIN_BOVW_LARGE_BUDGETS:-}"
export WAVELET_MAIN_LATEST_RATIOS="${WAVELET_MAIN_BOVW_LARGE_RATIOS:-0.05 0.10 0.15 0.20}"
export WAVELET_MAIN_LATEST_REPORT_NAME="${WAVELET_MAIN_BOVW_LARGE_REPORT_NAME:-dense_sift_bovw_large_ratios_multigpu}"

# Reuse the existing dense_sift_bovw precompute artifacts by default. The train
# root is separated so these high-budget multi-GPU runs do not overwrite earlier
# small-budget metrics.
export FEATURE_CACHE_ROOT="${WAVELET_MAIN_BOVW_LARGE_FEATURE_CACHE_ROOT:-artifacts/feature_cache_dense_sift_bovw}"
export TOPOLOGY_ROOT="${WAVELET_MAIN_BOVW_LARGE_TOPOLOGY_ROOT:-artifacts/topology_graph_dense_sift_bovw}"
export WAVELET_MAIN_LATEST_CROSS_OUTPUT_ROOT="${WAVELET_MAIN_BOVW_LARGE_CROSS_OUTPUT_ROOT:-artifacts/cross_modal_topology_dense_sift_bovw}"
export WAVELET_MAIN_LATEST_SELECTION_OUTPUT_ROOT="${WAVELET_MAIN_BOVW_LARGE_SELECTION_OUTPUT_ROOT:-artifacts/subset_selection_dense_sift_bovw_large_ratios}"
export WAVELET_MAIN_LATEST_TRAIN_OUTPUT_ROOT="${WAVELET_MAIN_BOVW_LARGE_TRAIN_OUTPUT_ROOT:-artifacts/subset_train_dense_sift_bovw_large_ratios_multigpu}"

export SELECTION_IMAGE_REPR_METHOD="${WAVELET_MAIN_BOVW_LARGE_IMAGE_REPR_METHOD:-dense_sift_bovw}"
export BOVW_CODEBOOK_SIZE="${WAVELET_MAIN_BOVW_LARGE_CODEBOOK_SIZE:-512}"
export DENSE_SIFT_STEP="${WAVELET_MAIN_BOVW_LARGE_STEP:-8}"
export DENSE_SIFT_PATCH="${WAVELET_MAIN_BOVW_LARGE_PATCH:-16}"
export BOVW_MAX_FIT_DESCRIPTORS="${WAVELET_MAIN_BOVW_LARGE_MAX_FIT_DESCRIPTORS:-200000}"
export BOVW_DESCRIPTORS_PER_IMAGE="${WAVELET_MAIN_BOVW_LARGE_DESCRIPTORS_PER_IMAGE:-200}"

export ENABLE_IMAGE_ENCODER_DATA_PARALLEL="${ENABLE_IMAGE_ENCODER_DATA_PARALLEL:-1}"
export IMAGE_ENCODER_DATA_PARALLEL_DEVICE_IDS="${IMAGE_ENCODER_DATA_PARALLEL_DEVICE_IDS:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Conservative defaults for large ratio runs. Override from the command line if
# your memory budget allows larger batches. Subset selection is still a single
# GPU proxy-optimization step, so these must be forwarded to the latest-combo
# variables that run_wavelet_main_latest_combo.sh actually consumes.
export BATCH_TRAIN="${BATCH_TRAIN:-64}"
export BATCH_TEST="${BATCH_TEST:-64}"
export TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-512}"
export WAVELET_MAIN_LATEST_PROXY_BATCH_SIZE="${WAVELET_MAIN_LATEST_PROXY_BATCH_SIZE:-${PROXY_BATCH_SIZE:-512}}"
export WAVELET_MAIN_LATEST_PROXY_TARGET_BATCH_SIZE="${WAVELET_MAIN_LATEST_PROXY_TARGET_BATCH_SIZE:-${PROXY_TARGET_BATCH_SIZE:-1024}}"
export WAVELET_MAIN_LATEST_LSRC_BATCH_SIZE="${WAVELET_MAIN_LATEST_LSRC_BATCH_SIZE:-${LSRC_BATCH_SIZE:-1024}}"
export WAVELET_MAIN_LATEST_MAIN_SWD_NUM_PROJECTIONS="${WAVELET_MAIN_LATEST_MAIN_SWD_NUM_PROJECTIONS:-32}"
export WAVELET_MAIN_LATEST_WAVELET_SWD_NUM_PROJECTIONS="${WAVELET_MAIN_LATEST_WAVELET_SWD_NUM_PROJECTIONS:-32}"

# Keep the old variable names populated for any external logging or wrappers.
export PROXY_BATCH_SIZE="${PROXY_BATCH_SIZE:-${WAVELET_MAIN_LATEST_PROXY_BATCH_SIZE}}"
export PROXY_TARGET_BATCH_SIZE="${PROXY_TARGET_BATCH_SIZE:-${WAVELET_MAIN_LATEST_PROXY_TARGET_BATCH_SIZE}}"
export LSRC_BATCH_SIZE="${LSRC_BATCH_SIZE:-${WAVELET_MAIN_LATEST_LSRC_BATCH_SIZE}}"

if [[ "${WAVELET_MAIN_BOVW_LARGE_SELECTION_PARALLEL:-1}" == "1" && -n "${WAVELET_MAIN_LATEST_RATIOS// }" ]]; then
  original_cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  selection_gpus_csv="${WAVELET_MAIN_BOVW_LARGE_SELECTION_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
  IFS=',' read -r -a selection_gpus <<< "${selection_gpus_csv}"
  read -r -a ratio_jobs <<< "${WAVELET_MAIN_LATEST_RATIOS}"
  max_parallel="${WAVELET_MAIN_BOVW_LARGE_SELECTION_PARALLEL_JOBS:-${#selection_gpus[@]}}"
  if [[ "${max_parallel}" -lt 1 ]]; then
    max_parallel=1
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Selection GPU-parallel mode: ratios=${WAVELET_MAIN_LATEST_RATIOS} gpus=${selection_gpus_csv} max_parallel=${max_parallel}"

  # Build or verify shared artifacts once before launching parallel selection
  # jobs, avoiding races in feature/topology/cross-modal precompute.
  (
    export WAVELET_MAIN_LATEST_BUDGETS=""
    export WAVELET_MAIN_LATEST_RATIOS=""
    export WAVELET_MAIN_LATEST_RUN_SELECTION=0
    export WAVELET_MAIN_LATEST_RUN_TRAIN=0
    export WAVELET_MAIN_LATEST_REPORT_NAME="${WAVELET_MAIN_LATEST_REPORT_NAME}_precompute"
    bash "${SCRIPT_DIR}/run_wavelet_main_latest_combo.sh"
  )

  pids=()
  launched=0
  for ratio in "${ratio_jobs[@]}"; do
    gpu="${selection_gpus[$((launched % ${#selection_gpus[@]}))]}"
    ratio_label="${ratio//./p}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launch selection: ratio=${ratio} gpu=${gpu}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export WAVELET_MAIN_LATEST_BUDGETS=""
      export WAVELET_MAIN_LATEST_RATIOS="${ratio}"
      export WAVELET_MAIN_LATEST_RUN_SELECTION=1
      export WAVELET_MAIN_LATEST_RUN_TRAIN=0
      export WAVELET_MAIN_LATEST_REPORT_NAME="${WAVELET_MAIN_LATEST_REPORT_NAME}_select_${ratio_label}"
      bash "${SCRIPT_DIR}/run_wavelet_main_latest_combo.sh"
    ) &
    pids+=("$!")
    launched=$((launched + 1))
    if [[ "${#pids[@]}" -ge "${max_parallel}" ]]; then
      for pid in "${pids[@]}"; do
        wait "${pid}"
      done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Selection GPU-parallel jobs completed. Start training/evaluation pass."
  if [[ -n "${original_cuda_visible_devices}" ]]; then
    export CUDA_VISIBLE_DEVICES="${original_cuda_visible_devices}"
  else
    unset CUDA_VISIBLE_DEVICES
  fi
  export WAVELET_MAIN_LATEST_RUN_SELECTION=0
  export WAVELET_MAIN_LATEST_RUN_TRAIN=1
fi

exec bash "${SCRIPT_DIR}/run_wavelet_main_latest_combo.sh"
