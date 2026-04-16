"""
DataProphet-inspired baseline (sample-level surrogate).

Source paper:
- "DataProphet: Demystifying Supervision Data Generalization in Multimodal LLMs" (2026).

Original setting:
- Primarily dataset-level/source-level transfer ranking.

Adopted interpretation in this project:
- Practical sample-level training-free surrogate for pair-level subset selection.
- Combines perplexity proxy, image-text relevance proxy, and local diversity.

reproduction_status: surrogate_sample_level
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


@register_method("dataprophet")
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
    joint = np.asarray(dataset["joint_features"], dtype=np.float32)
    n = int(joint.shape[0])
    k = resolve_subset_size(n, ratio)

    nn_k = int(cfg.get("dataprophet_k", 32))
    temp = float(cfg.get("dataprophet_temperature", 0.07))
    w_perplex = float(cfg.get("dataprophet_w_perplexity", 0.45))
    w_relevance = float(cfg.get("dataprophet_w_relevance", 0.30))
    w_diversity = float(cfg.get("dataprophet_w_diversity", 0.25))
    n_jobs = int(cfg.get("dataprophet_n_jobs", 1))

    img_neighbors, img_dist = knn_indices_distances(img, k=nn_k, metric="cosine", n_jobs=n_jobs)
    txt_neighbors, txt_dist = knn_indices_distances(txt, k=nn_k, metric="cosine", n_jobs=n_jobs)
    _, joint_dist = knn_indices_distances(joint, k=nn_k, metric="cosine", n_jobs=n_jobs)

    perplex_img = local_entropy_from_distances(img_dist, temperature=temp)
    perplex_txt = local_entropy_from_distances(txt_dist, temperature=temp)
    perplex_joint = local_entropy_from_distances(joint_dist, temperature=temp)
    perplexity_proxy = (0.4 * perplex_img + 0.4 * perplex_txt + 0.2 * perplex_joint).astype(np.float32)

    relevance_proxy = neighborhood_overlap_score(img_neighbors, txt_neighbors).astype(np.float32)
    diversity_proxy = local_diversity_score(joint_dist).astype(np.float32)

    score_pair = (
        w_perplex * perplexity_proxy
        + w_relevance * relevance_proxy
        + w_diversity * diversity_proxy
    ).astype(np.float32)
    selected = topk_indices(score_pair, k, largest=True)

    return {
        "method": "dataprophet",
        "ratio": float(ratio),
        "selected_local_indices": selected,
        "scores": {
            "score_img": perplex_img.astype(np.float32),
            "score_txt": perplex_txt.astype(np.float32),
            "score_joint": perplex_joint.astype(np.float32),
            "score_pair": score_pair,
        },
        "meta": {
            "subset_size": int(k),
            "num_samples": int(n),
            "notes": (
                "Adapted from DataProphet: original dataset-level transfer method; "
                "current implementation is a practical sample-level surrogate."
            ),
            "config": cfg,
        },
    }

