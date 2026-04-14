from typing import Any, Dict, Optional

import numpy as np

from baselines.common.multimodal_scoring import fuse_pair_scores
from baselines.common.selection_utils import resolve_subset_size, topk_indices


def make_result(
    method: str,
    ratio: float,
    n: int,
    score_img: np.ndarray,
    score_txt: np.ndarray,
    score_joint: Optional[np.ndarray],
    config: Dict[str, Any],
    notes: str = "",
) -> Dict[str, Any]:
    k = resolve_subset_size(n, ratio)
    fused = fuse_pair_scores(
        score_img=score_img,
        score_txt=score_txt,
        score_joint=score_joint,
        fusion=str(config.get("pair_score_fusion", "weighted_sum")),
        weights=tuple(config.get("pair_score_weights", [0.5, 0.5, 0.0])),
        normalize_mode=str(config.get("score_normalization", "zscore")),
    )
    selected_local = topk_indices(fused["score_pair"], k, largest=True)
    return {
        "method": method,
        "ratio": float(ratio),
        "selected_local_indices": selected_local,
        "scores": fused,
        "meta": {
            "subset_size": int(k),
            "num_samples": int(n),
            "notes": notes,
        },
    }

