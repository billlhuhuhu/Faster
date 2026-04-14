# Baseline Usage (`baselines/` independent framework)

## 1) Scope and isolation
- This package is independent from mainline `src/` pipeline.
- It reads existing feature caches and sample metadata; it does not modify mainline behavior.
- Final output is always pair-level `sample_idx` subset files.

## 2) Pair-level multimodal selection rule (uniform)
- Data unit: image-text pair sample.
- For each method we keep:
  - `score_img`
  - `score_txt`
  - optional `score_joint`
  - `score_pair` (fusion result)
- Default fusion strategy:
  - score normalization first
  - then `weighted_sum`

## 3) CLI entrypoints

### 3.1 Run one baseline
```bash
python -m baselines.runners.run_baseline_selection \
  --method entropy \
  --ratio 0.05 \
  --dataset_name flickr \
  --split train \
  --image_encoder nfnet \
  --text_encoder bert \
  --feature_source artifacts/feature_cache \
  --output_dir artifacts/baselines \
  --pair_score_fusion weighted_sum \
  --seed 0 \
  --device cpu
```

### 3.2 Sweep multiple baselines
```bash
python -m baselines.runners.benchmark_baselines \
  --dataset_name flickr \
  --image_encoder nfnet \
  --text_encoder bert \
  --feature_source artifacts/feature_cache \
  --output_dir artifacts/baselines \
  --methods entropy,el2n,grand,ccs-rand,ccs-herd,ccs-kcenter,ccs-forget,gradmatch,glister,dq,dfool,nms,adap_sne \
  --ratios 0.05,0.1,0.2 \
  --seed 0 \
  --device cpu
```

## 4) Output format (uniform)
Output directory:

`{output_dir}/{dataset}/{split}/{image_encoder}_{text_encoder}/ratio_xx/{method}/seed_x/`

Saved files:
- `selected_indices.json` (pair-level `sample_idx`)
- `selection_scores.npz` (`score_img`, `score_txt`, `score_joint`, `score_pair`)
- `baseline_summary.json`

## 5) Reproduction status quick map
- `faithful`: `rand`
- `faithful_but_practical`: `entropy`, `el2n`, `grand`, `gradmatch`, `glister`, `herd`, `kcenter`, `forget`
- `surrogate`: `dfool`
- `assumed_version`: `dq`, `nms`, `adap_sne`
- `project_specific_variant`: `ccs-rand`, `ccs-herd`, `ccs-kcenter`, `ccs-forget`

## 6) Ambiguity clarifications
- `DQ` is interpreted as Dataset Quantization adaptation (not generic data quality).
- `Dfool` is interpreted as DeepFool/DFAL-style boundary proximity baseline, implemented as practical surrogate.
- `NMS` means Near-Memory Sampling on Manifolds (not non-maximum suppression); hardware-specific parts omitted.
- `AdapSNE` is treated as an NMS successor and implemented as a practical surrogate inspired by entropy-guided adaptive manifold sampling.
- `CCS-*` names are project-specific coverage-wrapper variants, not claimed as official canonical sub-method names from one paper.

## 7) Main experiment aligned baseline protocol

### Aligned fields (for fair comparison)
- dataset: `flickr`
- image encoder: `nfnet`
- text encoder: `bert`
- budgets: `100`, `200`, `500`
- sample unit: pair-level `sample_idx`
- feature source: `artifacts/feature_cache/{dataset}/{split}/{image_encoder}_{text_encoder}`
- output format: `selected_indices.json`, `selection_scores.npz`, `baseline_summary.json`
- split protocol: `train`
- seed protocol: shared seed list from main-aligned config

### Not aligned by design
- mainline-specific internals are not reproduced in baselines:
  - diffusion objectives
  - wavelet matching internals
  - LSRC internals
  - grouped loss internals
  - fused matching internals

### Why this remains fair
- Baselines and main experiment use the same dataset, encoder pair, feature cache source, pair-level sample unit, and selection budgets.
- This isolates selection strategy differences while avoiding leakage from mainline method-specific objective engineering.

### Main-aligned run examples
Run all methods with aligned defaults:
```bash
python -m baselines.runners.run_main_aligned_baselines \
  --config baselines/configs/main_aligned_flickr_nfnet_bert.yaml
```

Override methods/budgets/seeds:
```bash
python -m baselines.runners.run_main_aligned_baselines \
  --config baselines/configs/main_aligned_flickr_nfnet_bert.yaml \
  --methods entropy el2n grand gradmatch glister ccs-rand ccs-herd ccs-kcenter ccs-forget dq dfool nms adap_sne \
  --budgets 100 200 500 \
  --seeds 0
```

Generate aligned benchmark tables:
```bash
python -m baselines.runners.benchmark_baselines \
  --config baselines/configs/main_aligned_flickr_nfnet_bert.yaml \
  --budgets 100 200 500 \
  --seeds 0
```

Shell wrappers (baseline-only env vars):
```bash
bash baselines/scripts/run_main_aligned_flickr_nfnet_bert.sh
bash baselines/scripts/run_baseline_table_flickr.sh
```
