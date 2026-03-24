#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SEEDS=("${SEEDS_DEFAULT[@]}")
RATIOS=("${RATIOS_DEFAULT[@]}")
DATASETS=("flickr")
BACKBONES=("resnet-50" "vit-b/16")
METHODS=("ours_baseline" "ours_full")

for dataset in "${DATASETS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for method in "${METHODS[@]}"; do
          bash "${SCRIPT_DIR}/run_pipeline.sh" "$dataset" "$backbone" "$ratio" "$seed" "$method"
        done
      done
    done
  done
done

echo "Generalization table runs submitted."
