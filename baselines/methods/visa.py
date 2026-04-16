"""
ViSA baseline (visual-centric adapted counterexample).

Source paper:
- "Picking the Cream of the Crop: Visual-Centric Data Selection with Collaborative
  Agents" (2025).

Adopted interpretation in this project:
- Visual-centric image-first selection.
- Image information score is primary; image-text relevance is a lightweight
  correction term.

Important note:
- Practical adaptation for pair-level multimodal retrieval subset selection,
  not a strict reproduction of the original setup.

reproduction_status: adapted_counterexample
"""

from typing import Any, Dict

import numpy as np

from baselines.common.selection_utils import resolve_subset_size, topk_indices
from baselines.common.training_free_multimodal_scoring import (
    knn_indices_distances,
    local_diversity_score,
    local_entropy_from_distances,
    neighborhood_overlap_score,
)
from baselines.registry import register_method


@register_method("visa")
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
    txt = np.asarray(text_features if text_features is not None else dataset["text_features"], dtype=np.float32)
    n = int(img.shape[0])
    k = resolve_subset_size(n, ratio)

    nn_k = int(cfg.get("visa_k", 32))
    temp = float(cfg.get("visa_temperature", 0.07))
    w_visual = float(cfg.get("visa_w_visual", 0.8))
    w_relevance = float(cfg.get("visa_w_relevance", 0.2))

    img_neighbors, img_dist = knn_indices_distances(img, k=nn_k, metric="cosine", n_jobs=int(cfg.get("visa_n_jobs", 1)))
    txt_neighbors, _ = knn_indices_distances(txt, k=nn_k, metric="cosine", n_jobs=int(cfg.get("visa_n_jobs", 1)))

    visual_score = (
        0.6 * local_entropy_from_distances(img_dist, temperature=temp)
        + 0.4 * local_diversity_score(img_dist)
    ).astype(np.float32)
    relevance_score = neighborhood_overlap_score(img_neighbors, txt_neighbors).astype(np.float32)

    score_img = visual_score
    score_txt = relevance_score
    score_joint = (0.5 * visual_score + 0.5 * relevance_score).astype(np.float32)
    score_pair = (w_visual * visual_score + w_relevance * relevance_score).astype(np.float32)
    selected = topk_indices(score_pair, k, largest=True)

    return {
        "method": "visa",
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
                "Adopted interpretation: visual-centric image-first selection with "
                "lightweight image-text relevance correction (practical adaptation)."
            ),
            "config": cfg,
        },
    }

