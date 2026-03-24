#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SEEDS=("${SEEDS_DEFAULT[@]}")
DATASET="flickr"
BACKBONE="nfnet"
RATIO="0.1"
METHODS=("ours_baseline" "ours_full")

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    bash "${SCRIPT_DIR}/run_pipeline.sh" "$DATASET" "$BACKBONE" "$RATIO" "$seed" "$method"
  done
done

echo "Ablation table starter runs submitted."
echo "Planned ablations such as w/o_frequency_alignment still need dedicated flags before they can be added here."
