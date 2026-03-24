# Experiment Scripts

This directory contains bash templates for the first-round server experiments.

Files:

- `common.sh`: shared roots, defaults, and helper functions
- `run_pipeline.sh`: run one complete pipeline for one dataset/backbone/ratio/seed/method
- `run_main_table.sh`: main table matrix
- `run_generalization_table.sh`: generalization matrix
- `run_ablation_table.sh`: ablation starter matrix

Runnable methods right now:

- `ours_baseline`
- `ours_full`

Planned methods in the experiment matrix:

- `random`
- `kcenter`
- `lors`

Example:

```bash
bash scripts/experiments/run_pipeline.sh flickr nfnet 0.05 0 ours_full
```

Before running on server, export the correct dataset roots if needed:

```bash
export FLICKR_IMAGE_ROOT=/path/to/Flickr30k
export COCO_IMAGE_ROOT=/path/to/COCO
export ANN_ROOT=/path/to/Flickr30k_ann
export DEVICE=cuda
```
