"""
CCS-rand / rand baseline.

Definition:
- Random sampling baseline on pair-level sample indices.

CCS note:
- In this project, CCS is used as a coverage-centric wrapper notion.
- `ccs-rand` is a project-specific CCS variant with random base selector.

reproduction_status: faithful (rand), project_specific_variant (ccs-rand)
"""

from typing import Any, Dict

import numpy as np

from baselines.common.selection_utils import random_select_indices, resolve_subset_size
from baselines.registry import register_method


@register_method("ccs-rand")
@register_method("rand")
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
    n = int(dataset["num_samples"])
    k = resolve_subset_size(n, ratio)
    selected = random_select_indices(n, k, seed=int(cfg.get("seed", 0)))
    zero = np.zeros(n, dtype=np.float32)
    return {
        "method": "ccs-rand",
        "ratio": float(ratio),
        "selected_local_indices": selected,
        "scores": {
            "score_img": zero,
            "score_txt": zero,
            "score_joint": zero,
            "score_pair": zero,
        },
        "meta": {
            "subset_size": int(k),
            "num_samples": int(n),
            "notes": "CCS group baseline: random pair-level sampling.",
        },
    }
