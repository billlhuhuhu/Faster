"""
NMS baseline (Near-Memory Sampling on Manifolds interpretation).

Paper target (adopted interpretation):
- "NMS: Efficient Edge DNN Training via Near-Memory Sampling on Manifolds" (2025).

Important naming clarification:
- Here NMS means Near-Memory Sampling, not generic non-maximum suppression.

Practical multimodal adaptation in this project:
- Build pair-level manifold representation from image/text/joint features.
- Perform low-dimensional embedding + representative sampling in manifold space.
- Output pair-level `sample_idx` subset.

Implementation note:
- Algorithmic NMS baseline; hardware-specific near-memory parts are omitted.

reproduction_status: assumed_version (algorithmic approximation)
"""

from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from baselines.common.selection_utils import resolve_subset_size
from baselines.registry import register_method


def _manifold_embed(x: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    pca_dim = int(cfg.get("nms_pca_dim", min(64, x.shape[1])))
    pca_dim = max(2, min(pca_dim, x.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=int(cfg.get("seed", 0)))
    x_pca = pca.fit_transform(x)

    method = str(cfg.get("nms_manifold_method", "pca")).lower()
    if method == "pca":
        return x_pca.astype(np.float32)

    if method == "tsne":
        tsne = TSNE(
            n_components=int(cfg.get("nms_manifold_dim", 2)),
            perplexity=float(cfg.get("nms_perplexity", 30.0)),
            learning_rate=float(cfg.get("nms_learning_rate", 200.0)),
            init="pca",
            random_state=int(cfg.get("seed", 0)),
        )
        return tsne.fit_transform(x_pca).astype(np.float32)

    raise ValueError(f"Unsupported nms_manifold_method={method}")


@register_method("nms")
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

    z_joint = _manifold_embed(joint, cfg)
    z_img = _manifold_embed(img, cfg)
    z_txt = _manifold_embed(txt, cfg)

    km = KMeans(n_clusters=int(k), random_state=int(cfg.get("seed", 0)), n_init=10)
    cid = km.fit_predict(z_joint)
    centers = km.cluster_centers_

    selected = []
    dist_joint = np.zeros(n, dtype=np.float32)
    for c in range(int(k)):
        members = np.where(cid == c)[0]
        if members.size == 0:
            continue
        c_points = z_joint[members]
        d = np.linalg.norm(c_points - centers[c : c + 1], axis=1)
        dist_joint[members] = d.astype(np.float32)
        selected.append(int(members[np.argmin(d)]))

    if len(selected) < int(k):
        order = np.argsort(dist_joint)
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= int(k):
                break

    score_joint = (-dist_joint).astype(np.float32)
    score_img = (-np.linalg.norm(z_img - np.mean(z_img, axis=0, keepdims=True), axis=1)).astype(np.float32)
    score_txt = (-np.linalg.norm(z_txt - np.mean(z_txt, axis=0, keepdims=True), axis=1)).astype(np.float32)
    score_pair = (
        float(cfg.get("nms_w_joint", 0.6)) * score_joint
        + float(cfg.get("nms_w_img", 0.2)) * score_img
        + float(cfg.get("nms_w_txt", 0.2)) * score_txt
    ).astype(np.float32)

    return {
        "method": "nms",
        "ratio": float(ratio),
        "selected_local_indices": selected[: int(k)],
        "scores": {
            "score_img": score_img,
            "score_txt": score_txt,
            "score_joint": score_joint,
            "score_pair": score_pair,
        },
        "meta": {
            "subset_size": int(k),
            "num_samples": int(n),
            "notes": "Algorithmic NMS baseline (Near-Memory Sampling interpretation), hardware-specific parts omitted.",
            "config": cfg,
        },
    }

