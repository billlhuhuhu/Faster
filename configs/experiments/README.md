# Experiment Configs

This directory stores the first-round experiment matrix for server runs.

Files:

- `main_table.json`: main comparison table
- `generalization_table.json`: cross-backbone generalization table
- `ablation_table.json`: ablation table

Conventions:

- `dataset`: `flickr` or `coco`
- `image_encoder`: `nfnet`, `resnet-50`, `vit-b/16`
- `text_encoder`: always `bert` in v1
- `subset_ratio`: real subset ratio, one of `0.05`, `0.1`, `0.2`
- `seed`: random seed for repeated runs
- `status`:
  - `ready`: can be launched by the current scripts
  - `planned`: included in the matrix, but still needs a method-specific runner

Recommended output roots on server:

- `artifacts/feature_cache`
- `artifacts/topology_graph`
- `artifacts/cross_modal_topology`
- `artifacts/subset_selection`
- `artifacts/subset_train`

Recommended naming:

- selection outputs:
  - `artifacts/subset_selection/{dataset}/train/{backbone}_bert/ratio_{xx}/{method}/`
- training outputs:
  - `artifacts/subset_train/{dataset}/{backbone}_bert/ratio_{xx}/{method}/seed_{seed}/`
