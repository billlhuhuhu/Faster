from typing import Dict, Optional, Tuple

import numpy as np


def normalize_scores(scores: np.ndarray, mode: str = "zscore", eps: float = 1e-8) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if mode == "none":
        return scores
    if mode == "zscore":
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        return (scores - mean) / max(std, eps)
    if mode == "minmax":
        lo = float(np.min(scores))
        hi = float(np.max(scores))
        return (scores - lo) / max(hi - lo, eps)
    if mode == "rank":
        order = np.argsort(np.argsort(scores))
        return order.astype(np.float32) / max(float(len(scores) - 1), 1.0)
    raise ValueError(f"Unknown score normalization mode: {mode}")


def fuse_pair_scores(
    score_img: np.ndarray,
    score_txt: np.ndarray,
    score_joint: Optional[np.ndarray] = None,
    fusion: str = "weighted_sum",
    weights: Tuple[float, float, float] = (0.5, 0.5, 0.0),
    normalize_mode: str = "zscore",
) -> Dict[str, np.ndarray]:
    img = normalize_scores(score_img, mode=normalize_mode)
    txt = normalize_scores(score_txt, mode=normalize_mode)
    if score_joint is None:
        joint = np.zeros_like(img)
    else:
        joint = normalize_scores(score_joint, mode=normalize_mode)

    if fusion == "average":
        pair = (img + txt) / 2.0
    elif fusion == "max":
        pair = np.maximum(img, txt)
    elif fusion == "geometric_mean":
        img_shift = img - np.min(img) + 1e-6
        txt_shift = txt - np.min(txt) + 1e-6
        pair = np.sqrt(img_shift * txt_shift)
    elif fusion == "normalized_sum":
        pair = normalize_scores(img + txt + joint, mode="minmax")
    elif fusion == "weighted_sum":
        w_img, w_txt, w_joint = weights
        pair = w_img * img + w_txt * txt + w_joint * joint
    else:
        raise ValueError(f"Unknown pair fusion mode: {fusion}")

    return {
        "score_img": img.astype(np.float32),
        "score_txt": txt.astype(np.float32),
        "score_joint": joint.astype(np.float32),
        "score_pair": pair.astype(np.float32),
    }

