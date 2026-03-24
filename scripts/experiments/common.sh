#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-data}"
FLICKR_IMAGE_ROOT="${FLICKR_IMAGE_ROOT:-${DATA_ROOT}/Flickr30k}"
COCO_IMAGE_ROOT="${COCO_IMAGE_ROOT:-${DATA_ROOT}/COCO}"
ANN_ROOT="${ANN_ROOT:-data/Flickr30k_ann}"

FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-artifacts/feature_cache}"
TOPOLOGY_ROOT="${TOPOLOGY_ROOT:-artifacts/topology_graph}"
CROSS_MODAL_ROOT="${CROSS_MODAL_ROOT:-artifacts/cross_modal_topology}"
SUBSET_SELECTION_ROOT="${SUBSET_SELECTION_ROOT:-artifacts/subset_selection}"
SUBSET_TRAIN_ROOT="${SUBSET_TRAIN_ROOT:-artifacts/subset_train}"

DEVICE="${DEVICE:-cuda}"
BATCH_FEATURE="${BATCH_FEATURE:-64}"
BATCH_TRAIN="${BATCH_TRAIN:-64}"
BATCH_TEST="${BATCH_TEST:-128}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
TRAIN_NO_AUG="${TRAIN_NO_AUG:-1}"

K_NEIGHBORS="${K_NEIGHBORS:-15}"
TOPOLOGY_METRIC_IMAGE="${TOPOLOGY_METRIC_IMAGE:-euclidean}"
TOPOLOGY_METRIC_TEXT="${TOPOLOGY_METRIC_TEXT:-cosine}"
ALPHA="${ALPHA:-1.0}"

SEEDS_DEFAULT=(0 1 2)
RATIOS_DEFAULT=(0.05 0.1 0.2)

get_image_root() {
  local dataset="$1"
  if [[ "$dataset" == "flickr" ]]; then
    echo "$FLICKR_IMAGE_ROOT"
  elif [[ "$dataset" == "coco" ]]; then
    echo "$COCO_IMAGE_ROOT"
  else
    echo "Unknown dataset: $dataset" >&2
    return 1
  fi
}

selection_method_from_name() {
  local method="$1"
  if [[ "$method" == "ours_baseline" ]]; then
    echo "baseline"
  elif [[ "$method" == "ours_full" ]]; then
    echo "proxy_opt"
  else
    echo ""
  fi
}

sanitize_component() {
  local value="$1"
  value="${value//\\/-}"
  value="${value//\//-}"
  value="${value// /_}"
  echo "$value"
}
