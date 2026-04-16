# Method Mapping for `baselines/` (paper definitions + pair-level adaptation)

This document defines how each baseline shorthand is interpreted in this project, with explicit reproduction-status labels.

## 1) Repository evidence checked before mapping
- `README.md`
- `configs/experiments/main_table.json`
- `scripts/experiments/*.sh`
- `src/*.py`
- `tools/*.py`

Observed in-repo facts:
- Mainline pipeline currently runs `ours_baseline` / `ours_full`.
- `random`, `kcenter`, `lors` appear as planned comparison methods.
- No canonical in-repo definition text for `DQ`, `Dfool`, `NMS`, `AdapSNE`, or single-algorithm `CCS`.

## 2) Unified multimodal pair-level rule (applies to all methods)
- Data unit in this project: **pair-level image-text sample**.
- Final selection unit: **pair-level `sample_idx`**.
- For single-modality papers we adapt through:
  - `score_img(i)`
  - `score_txt(i)`
  - optional `score_joint(i)`
  - `score_pair(i) = fusion(score_img, score_txt, score_joint)`
- Default fusion:
  - normalize branch scores first
  - then weighted sum (`weighted_sum`)
- Geometry/coverage methods can operate on image/text/joint spaces.
- Training-dynamics / uncertainty methods compute branch scores then aggregate to pair level.

## 3) Reproduction status labels
- `faithful`
- `faithful_but_practical`
- `surrogate`
- `assumed_version`
- `project_specific_variant` (used for CCS wrapper variants in this repo)
- `adapted_counterexample`
- `surrogate_sample_level`
- `adapted_dynamic_baseline`

## 4) Method mapping table

| method | full name | paper / source | multimodal pair-level adaptation | reproduction_status | ambiguity note |
|---|---|---|---|---|---|
| `entropy` | Shannon entropy uncertainty sampling | classical uncertainty/active-learning definition | branch entropy (img/txt, optional joint) -> fused pair score | `faithful_but_practical` | none |
| `el2n` | Early-L2-Norm score | *Deep Learning on a Data Diet* | early-epoch error-vector L2 on branches -> pair score | `faithful_but_practical` | surrogate training loop |
| `grand` | Gradient-Norm score | *Deep Learning on a Data Diet* | early-epoch per-sample grad-norm proxy on branches -> pair score | `faithful_but_practical` | surrogate gradient computation |
| `gradmatch` | GRAD-MATCH | *GRAD-MATCH: Gradient Matching based Data Subset Selection...* | pair sample as unit, branch gradients combined to pair gradient rep | `faithful_but_practical` | faithful practical surrogate |
| `glister` | GLISTER | *GLISTER: Generalization based Data Subset Selection...* | pair sample as unit, branch info enters validation-gain proxy | `faithful_but_practical` | approximate online/taylor surrogate |
| `rand` / `ccs-rand` | Random baseline / CCS-rand wrapper | standard random baseline | uniform random over pair `sample_idx` | `faithful` (`rand`), `project_specific_variant` (`ccs-rand`) | CCS wrapper is project-defined |
| `herd` / `ccs-herd` | Herding-style exemplar mean approximation | herding/exemplar-selection tradition (e.g., iCaRL-style exemplar selection) | branch/joint feature mean-approx objective -> pair selection | `faithful_but_practical` (`herd`), `project_specific_variant` (`ccs-herd`) | not claimed as one single canonical coreset paper |
| `kcenter` / `ccs-kcenter` | k-center greedy coreset | *Active Learning for CNNs: A Core-Set Approach* | k-center in image/text/joint spaces (current impl joint default) -> pair indices | `faithful_but_practical` (`kcenter`), `project_specific_variant` (`ccs-kcenter`) | CCS wrapper is project-defined |
| `forget` / `ccs-forget` | Forgetting-events baseline | *An Empirical Study of Example Forgetting...* | branch forgetting-event counts -> pair forgetting score | `faithful_but_practical` (`forget`), `project_specific_variant` (`ccs-forget`) | surrogate correctness proxy used |
| `dq` | Dataset Quantization adaptation | *Dataset Quantization* (2023) | pair-level representative quantization over multimodal features, prototype assignment | `assumed_version` | **adopted as practical DQ variant**, full original optimization not fully reproduced |
| `dfool` | DeepFool/DFAL-style boundary proximity baseline | DeepFool; DFAL literature | branch boundary-proximity proxy -> fused pair score | `surrogate` | practical surrogate of DFAL/DeepFool (margin proxy instead of full perturbation loop) |
| `nms` | Near-Memory Sampling on Manifolds (not non-max suppression) | *NMS: Efficient Edge DNN Training via Near-Memory Sampling on Manifolds* (2025) | manifold embedding on multimodal features + representative sampling -> pair subset | `assumed_version` | algorithmic NMS baseline, hardware-specific parts omitted |
| `adap_sne` / `adapsne` | AdapSNE | *AdapSNE: Adaptive Fireworks-Optimized and Entropy-Guided Dataset Sampling...* (2025) | manifold embedding + entropy/density-aware representative sampling -> pair subset | `assumed_version` | practical surrogate inspired by AdapSNE; fireworks/hardware details omitted |
| `presel` | PreSel image-first counterexample baseline | *Filter Images First, Generate Instructions Later...* (CVPR 2025) | image-side scoring first, then recover pair-level sample_idx | `adapted_counterexample` | adapted from PreSel; not a faithful reproduction of original visual instruction tuning data formation |
| `visa` | ViSA visual-centric counterexample baseline | *Picking the Cream of the Crop: Visual-Centric Data Selection with Collaborative Agents* (2025) | visual-centric image-first score + lightweight image-text relevance correction -> pair subset | `adapted_counterexample` | adopted interpretation = visual-centric image-first selection |
| `dataprophet` | DataProphet-inspired sample-level surrogate | *DataProphet: Demystifying Supervision Data Generalization in Multimodal LLMs* (2026) | training-free pair-level score from perplexity proxy + relevance + diversity | `surrogate_sample_level` | original method is dataset-level transfer ranking; this repo uses sample-level surrogate |
| `dynamic_pruning` / `infobatch` | InfoBatch-style dynamic pruning baseline | *InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning* | dynamic training-time keep/drop proxy + budget-aligned compatibility subset export | `adapted_dynamic_baseline` | dynamic method by nature; static selected_indices is compatibility export for unified pipeline |

## 5) CCS clarification (must-read)
- In this repository, **CCS is treated as a coverage-aware wrapper/framework concept**.
- `ccs_rand`, `ccs_herd`, `ccs_kcenter`, `ccs_forget` are **project-specific CCS variants**:
  - CCS-rand = coverage-centric wrapper + random base selector
  - CCS-herd = coverage-centric wrapper + herding base selector
  - CCS-kcenter = coverage-centric wrapper + k-center base selector
  - CCS-forget = coverage-centric wrapper + forgetting-events base selector
- These names are **not claimed** as official canonical sub-algorithms from a single CCS paper.

## 6) New method adaptation notes (PreSel / ViSA / DataProphet / dynamic_pruning)
- `presel`:
  - original setting: pre-instruction image-first filtering for visual instruction tuning.
  - current adaptation: image-first counterexample baseline in pair-level retrieval subset selection.
  - status: `adapted_counterexample`.
- `visa`:
  - original setting: visual-centric collaborative-agent data selection.
  - current adaptation: visual-centric image-first scoring with lightweight image-text relevance correction.
  - status: `adapted_counterexample`.
- `dataprophet`:
  - original setting: dataset/source-level transfer generalization ranking.
  - current adaptation: practical sample-level surrogate with training-free multimodal score.
  - status: `surrogate_sample_level`.
- `dynamic_pruning` (`infobatch` alias):
  - original setting: dynamic training-time unbiased pruning (not static subset selection).
  - current adaptation: InfoBatch-style dynamic proxy with budget-aligned compatibility subset export.
  - status: `adapted_dynamic_baseline`.

## 7) Public-paper reference pointers
- Data Diet (EL2N/GraNd): *Deep Learning on a Data Diet: Finding Important Examples Early in Training*
- GradMatch: *GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training*
- GLISTER: *GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning*
- Forgetting: *An Empirical Study of Example Forgetting during Deep Neural Network Learning*
- Core-set k-center: *Active Learning for Convolutional Neural Networks: A Core-Set Approach*
- DeepFool: *A Simple and Accurate Method to Fool Deep Neural Networks*
- DQ: *Dataset Quantization* (2023)
- NMS (edge sampling): *NMS: Efficient Edge DNN Training via Near-Memory Sampling on Manifolds* (2025)
- AdapSNE: *AdapSNE: Adaptive Fireworks-Optimized and Entropy-Guided Dataset Sampling for Edge DNN Training* (2025)
