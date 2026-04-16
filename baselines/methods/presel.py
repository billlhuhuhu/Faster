"""
PreSel baseline (adapted counterexample).

Source paper:
- "Filter Images First, Generate Instructions Later: Pre-Instruction Data Selection
  for Visual Instruction Tuning" (CVPR 2025).

Adopted interpretation in this project:
- Image-first counterexample baseline.
- Selection score is built from image-side signals first, then mapped back to
  pair-level sample_idx.

Important note:
- This is an adapted baseline for multimodal retrieval subset selection, not a
  faithful reproduction of the original instruction tuning setting.

reproduction_status: adapted_counterexample
"""

from typing import Any, Dict

import numpy as np

from baselines.common.selection_utils import resolve_subset_size, topk_indices
from baselines.common.training_free_multimodal_scoring import (
    knn_indices_distances,
    local_diversity_score,
    local_entropy_from_distances,
)
from baselines.registry import register_method


@register_method("presel")
def select_subset(
    dataset: Dict[str, Any],
    ratio: float,
    model=None,
    image_features=None,
    text_features=None,
    labels=None,
    config=None,
) -> Dict[str, Any]:
    cfg = dict(config or {})
    img = np.asarray(image_features if image_features is not None else dataset["image_features"], dtype=np.float32)
    n = int(img.shape[0])
    k = resolve_subset_size(n, ratio)

    nn_k = int(cfg.get("presel_k", 32))
    temp = float(cfg.get("presel_temperature", 0.07))
    w_entropy = float(cfg.get("presel_w_entropy", 0.7))
    w_diversity = float(cfg.get("presel_w_diversity", 0.3))

    img_neighbors, img_dist = knn_indices_distances(img, k=nn_k, metric="cosine", n_jobs=int(cfg.get("presel_n_jobs", 1)))
    score_img_entropy = local_entropy_from_distances(img_dist, temperature=temp)
    score_img_div = local_diversity_score(img_dist)
    score_img = (w_entropy * score_img_entropy + w_diversity * score_img_div).astype(np.float32)

    # Image-first counterexample: text/joint scores are compatibility placeholders.
    score_txt = np.zeros(n, dtype=np.float32)
    score_joint = score_img.copy()
    score_pair = score_img.copy()
    selected = topk_indices(score_pair, k, largest=True)

    return {
        "method": "presel",
        "ratio": float(ratio),
        "selected_local_indices": selected,
        "scores": {
            "score_img": score_img,
            "score_txt": score_txt,
            "score_joint": score_joint,
            "score_pair": score_pair,
        },
        "meta": {
            "subset_size": int(k),
            "num_samples": int(n),
            "notes": (
                "Image-first counterexample baseline adapted from PreSel; "
                "not a faithful reproduction of the original VIT data formation setting."
            ),
            "config": cfg,
        },
    }

