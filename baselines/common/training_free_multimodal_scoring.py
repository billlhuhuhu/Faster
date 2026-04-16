from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def knn_indices_distances(features: np.ndarray, k: int = 32, metric: str = "cosine", n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    x = l2_normalize(features)
    k = max(2, min(int(k), int(x.shape[0])))
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=max(1, int(n_jobs)))
    nn.fit(x)
    distances, indices = nn.kneighbors(x)
    return indices, distances.astype(np.float32)


def local_entropy_from_distances(distances: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    sims = 1.0 - np.asarray(distances, dtype=np.float32)
    logits = sims / max(float(temperature), 1e-6)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    prob = np.exp(logits)
    prob = prob / np.maximum(np.sum(prob, axis=1, keepdims=True), 1e-8)
    entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-8)), axis=1)
    return entropy.astype(np.float32)


def local_diversity_score(distances: np.ndarray) -> np.ndarray:
    dist = np.asarray(distances, dtype=np.float32)
    if dist.shape[1] > 1:
        return np.mean(dist[:, 1:], axis=1).astype(np.float32)
    return dist[:, 0].astype(np.float32)


def neighborhood_overlap_score(img_neighbors: np.ndarray, txt_neighbors: np.ndarray) -> np.ndarray:
    img_n = np.asarray(img_neighbors)
    txt_n = np.asarray(txt_neighbors)
    if img_n.shape != txt_n.shape:
        raise ValueError(f"shape mismatch: img={img_n.shape}, txt={txt_n.shape}")
    n, k = img_n.shape
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = set(int(x) for x in img_n[i].tolist())
        b = set(int(x) for x in txt_n[i].tolist())
        inter = len(a.intersection(b))
        out[i] = float(inter) / max(float(k), 1.0)
    return out

