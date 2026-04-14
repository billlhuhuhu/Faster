"""
AdapSNE baseline (NMS successor interpretation).

Paper target (adopted interpretation):
- "AdapSNE: Adaptive Fireworks-Optimized and Entropy-Guided Dataset Sampling for Edge DNN Training" (2025).

Definition guidance:
- Treated as an NMS-enhanced manifold sampler with adaptive/entropy guidance.

Implementation note:
- Practical surrogate inspired by AdapSNE:
  manifold embedding + entropy/density-aware representative sampling.
- Fireworks optimization and hardware-specific edge details are omitted.

Multimodal pair-level adaptation:
- Build manifold embeddings for pair-level multimodal samples and select pair indices.

reproduction_status: assumed_version (practical surrogate)
"""

from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from baselines.common.selection_utils import resolve_subset_size
from baselines.registry import register_method


def _density_uncertainty_score(x2d: np.ndarray, k: int = 16) -> np.ndarray:
    k = max(2, min(int(k), int(x2d.shape[0])))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(x2d)
    distances, _ = nn.kneighbors(x2d)
    local_density = 1.0 / np.maximum(np.mean(distances[:, 1:], axis=1), 1e-6)
    density_norm = (local_density - np.min(local_density)) / max(float(np.max(local_density) - np.min(local_density)), 1e-8)
    uncertainty = 1.0 - density_norm
    return uncertainty.astype(np.float32)


@register_method("adapsne")
@register_method("adap_sne")
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
    joint = np.asarray(dataset["joint_features"], dtype=np.float32)
    n = int(joint.shape[0])
    k = resolve_subset_size(n, ratio)

    tsne = TSNE(
        n_components=2,
        perplexity=float(cfg.get("adapsne_perplexity", 30.0)),
        learning_rate=float(cfg.get("adapsne_learning_rate", 200.0)),
        init="pca",
        random_state=int(cfg.get("seed", 0)),
    )
    z2d = tsne.fit_transform(joint)
    uncertainty = _density_uncertainty_score(z2d, k=int(cfg.get("adapsne_density_k", 16)))

    # Adaptive spatial coverage in t-SNE space.
    kmeans = KMeans(n_clusters=int(k), random_state=int(cfg.get("seed", 0)), n_init=10)
    cluster_id = kmeans.fit_predict(z2d)
    centers = kmeans.cluster_centers_
    selected = []
    for c in range(int(k)):
        members = np.where(cluster_id == c)[0]
        if members.size == 0:
            continue
        c_points = z2d[members]
        center = centers[c : c + 1]
        dist = np.linalg.norm(c_points - center, axis=1)
        local_quality = uncertainty[members] - float(cfg.get("adapsne_center_dist_weight", 0.3)) * dist
        selected.append(int(members[np.argmax(local_quality)]))

    if len(selected) < int(k):
        extra = np.argsort(-uncertainty)
        for idx in extra:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= int(k):
                break

    return {
        "method": "adap_sne",
        "ratio": float(ratio),
        "selected_local_indices": selected[: int(k)],
        "scores": {
            "score_img": uncertainty,
            "score_txt": uncertainty,
            "score_joint": uncertainty,
            "score_pair": uncertainty,
        },
        "meta": {
            "subset_size": int(k),
            "num_samples": int(n),
            "notes": "Assumed AdapSNE version: t-SNE density-adaptive coverage sampling.",
            "config": cfg,
        },
    }
