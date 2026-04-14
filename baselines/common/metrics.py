from typing import Any, Dict

import numpy as np


def score_stats(scores: np.ndarray) -> Dict[str, float]:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_method_meta(
    method: str,
    ratio: float,
    num_samples: int,
    subset_size: int,
    config: Dict[str, Any],
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "method": str(method),
        "ratio": float(ratio),
        "num_samples": int(num_samples),
        "subset_size": int(subset_size),
        "notes": str(notes),
        "config": config,
    }

