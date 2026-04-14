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
import hashlib
import os
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _local_entropy(
    query_feat: np.ndarray,
    key_feat: np.ndarray,
    top_k: int = 64,
    temperature: float = 0.07,
    batch_size: int = 2048,
    show_progress: bool = False,
    progress_desc: str = "entropy-knn",
    n_jobs: int = -1,
) -> np.ndarray:
    top_k = max(2, min(int(top_k), int(key_feat.shape[0])))
    q = query_feat / np.maximum(np.linalg.norm(query_feat, axis=1, keepdims=True), 1e-8)
    k = key_feat / np.maximum(np.linalg.norm(key_feat, axis=1, keepdims=True), 1e-8)
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine", n_jobs=int(n_jobs))
    nn.fit(k)
    n = q.shape[0]
    batch_size = max(1, int(batch_size))
    entropy_chunks = []
    ids_chunks = []
    ranges = range(0, n, batch_size)
    iterator = ranges
    if show_progress and tqdm is not None:
        iterator = tqdm(ranges, desc=progress_desc, unit="batch", leave=False, dynamic_ncols=True)
    for start in iterator:
        end = min(start + batch_size, n)
        distances, ids = nn.kneighbors(q[start:end])
        sims = 1.0 - distances
        logits = sims / max(float(temperature), 1e-6)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        prob = np.exp(logits)
        prob = prob / np.maximum(np.sum(prob, axis=1, keepdims=True), 1e-8)
        entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-8)), axis=1)
        entropy_chunks.append(entropy.astype(np.float32))
        ids_chunks.append(ids)
    return np.concatenate(entropy_chunks, axis=0), np.concatenate(ids_chunks, axis=0)


def _weights_tag(config: Dict[str, Any]) -> str:
    weights = config.get("pair_score_weights", [0.5, 0.5, 0.0])
    try:
        return "_".join([f"{float(x):.4f}" for x in weights])
    except Exception:
        return str(weights)


def _build_cache_path(cfg: Dict[str, Any], dataset: Dict[str, Any], k: int, temperature: float) -> str:
    cache_root = str(cfg.get("cache_root", "artifacts/baselines/cache"))
    cache_dir = os.path.join(cache_root, "entropy")
    os.makedirs(cache_dir, exist_ok=True)
    feature_dir = str(dataset.get("feature_dir", "unknown_feature_dir"))
    feature_hash = hashlib.md5(feature_dir.encode("utf-8")).hexdigest()[:10]
    dataset_name = str(cfg.get("dataset_name", dataset.get("dataset_name", "dataset")))
    split = str(cfg.get("split", dataset.get("split", "train")))
    image_encoder = str(cfg.get("image_encoder", dataset.get("image_encoder", "img")))
    text_encoder = str(cfg.get("text_encoder", dataset.get("text_encoder", "txt")))
    seed = int(cfg.get("seed", 0))
    normalization = str(cfg.get("score_normalization", "zscore"))
    fusion = str(cfg.get("pair_score_fusion", "weighted_sum"))
    weights_tag = _weights_tag(cfg)
    file_name = (
        f"{dataset_name}_{split}_{image_encoder}_{text_encoder}_"
        f"s{seed}_k{k}_t{temperature:.4f}_{normalization}_{fusion}_{weights_tag}_{feature_hash}.npz"
    )
    return os.path.join(cache_dir, file_name)


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
    batch_size = int(cfg.get("entropy_batch_size", 2048))
    show_progress = bool(cfg.get("entropy_show_progress", True))
    knn_n_jobs = int(cfg.get("entropy_knn_n_jobs", -1))
    use_cache = bool(cfg.get("entropy_use_cache", True))

    cache_path = _build_cache_path(cfg, dataset, k=k, temperature=temperature)
    cached = None
    if use_cache and os.path.exists(cache_path):
        try:
            payload = np.load(cache_path)
            score_img = payload["score_img"].astype(np.float32)
            score_txt = payload["score_txt"].astype(np.float32)
            score_joint = payload["score_joint"].astype(np.float32)
            if score_img.shape[0] == img.shape[0]:
                cached = True
        except Exception:
            cached = None

    if cached:
        if show_progress:
            print(f"[entropy-cache-hit] {cache_path}")
    else:
        # NOTE:
        # Image/Text encoders can have different feature dimensions (e.g., nfnet vs bert),
        # so branch entropy is computed within each branch space to keep NN lookup valid.
        score_img, _ = _local_entropy(
            img,
            img,
            top_k=k,
            temperature=temperature,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="entropy-img",
            n_jobs=knn_n_jobs,
        )
        score_txt, _ = _local_entropy(
            txt,
            txt,
            top_k=k,
            temperature=temperature,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="entropy-txt",
            n_jobs=knn_n_jobs,
        )
        score_joint, _ = _local_entropy(
            joint,
            joint,
            top_k=k,
            temperature=temperature,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="entropy-joint",
            n_jobs=knn_n_jobs,
        )
        if use_cache:
            np.savez(
                cache_path,
                score_img=score_img.astype(np.float32),
                score_txt=score_txt.astype(np.float32),
                score_joint=score_joint.astype(np.float32),
            )
            if show_progress:
                print(f"[entropy-cache-save] {cache_path}")
    return make_result(
        method="entropy",
        ratio=ratio,
        n=img.shape[0],
        score_img=score_img,
        score_txt=score_txt,
        score_joint=score_joint,
        config=cfg,
        notes="Uncertainty from local in-branch neighborhood entropy with pair-level fusion.",
    )
