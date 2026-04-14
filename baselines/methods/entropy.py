"""
Entropy baseline (uncertainty sampling family).

Paper/definition context:
- Shannon entropy of model predictive distribution is a standard uncertainty score
  in active learning and uncertainty sampling literature.

Multimodal pair-level adaptation in this project:
- Compute entropy-style uncertainty for image->text and text->image retrieval neighborhoods.
- Optional joint score is built from branch scores.
- Fuse to pair-level score for final pair `sample_idx` selection.

reproduction_status: faithful_but_practical
"""

from typing import Any, Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from baselines.registry import register_method
from ._utils import make_result


def _local_entropy(query_feat: np.ndarray, key_feat: np.ndarray, top_k: int = 64, temperature: float = 0.07) -> np.ndarray:
    top_k = max(2, min(int(top_k), int(key_feat.shape[0])))
    q = query_feat / np.maximum(np.linalg.norm(query_feat, axis=1, keepdims=True), 1e-8)
    k = key_feat / np.maximum(np.linalg.norm(key_feat, axis=1, keepdims=True), 1e-8)
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    nn.fit(k)
    distances, ids = nn.kneighbors(q)
    sims = 1.0 - distances
    logits = sims / max(float(temperature), 1e-6)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    prob = np.exp(logits)
    prob = prob / np.maximum(np.sum(prob, axis=1, keepdims=True), 1e-8)
    entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-8)), axis=1)
    return entropy.astype(np.float32), ids


@register_method("entropy")
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
    k = int(cfg.get("entropy_top_k", 64))
    temperature = float(cfg.get("entropy_temperature", 0.07))

    score_img, _ = _local_entropy(img, txt, top_k=k, temperature=temperature)
    score_txt, _ = _local_entropy(txt, img, top_k=k, temperature=temperature)
    score_joint = np.mean(np.stack([score_img, score_txt], axis=1), axis=1)
    return make_result(
        method="entropy",
        ratio=ratio,
        n=img.shape[0],
        score_img=score_img,
        score_txt=score_txt,
        score_joint=score_joint,
        config=cfg,
        notes="Uncertainty from local cross-modal neighborhood entropy.",
    )
