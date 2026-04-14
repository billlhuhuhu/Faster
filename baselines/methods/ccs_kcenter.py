"""
CCS-kcenter / kcenter baseline.

Paper context:
- "Active Learning for Convolutional Neural Networks: A Core-Set Approach".

Definition:
- k-center greedy covering in feature space.

Multimodal pair-level adaptation:
- Run on joint pair features (can be extended to image/text/joint spaces).

CCS note:
- `ccs-kcenter` is a project-specific CCS variant: coverage wrapper + k-center base selector.

reproduction_status: faithful_but_practical (kcenter), project_specific_variant (ccs-kcenter)
"""

from typing import Any, Dict

import numpy as np

from baselines.common.selection_utils import kcenter_greedy_select, resolve_subset_size
from baselines.registry import register_method


@register_method("ccs-kcenter")
@register_method("kcenter")
@register_method("k-center")
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
    selected = kcenter_greedy_select(joint, k, seed=int(cfg.get("seed", 0)))
    centroid = np.mean(joint, axis=0, keepdims=True)
    score = np.linalg.norm(joint - centroid, axis=1).astype(np.float32)
    return {
        "method": "ccs-kcenter",
        "ratio": float(ratio),
        "selected_local_indices": selected,
        "scores": {
            "score_img": score,
            "score_txt": score,
            "score_joint": score,
            "score_pair": score,
        },
        "meta": {
            "subset_size": int(k),
            "num_samples": int(n),
            "notes": "CCS group baseline: pair-level k-center greedy on joint features.",
            "config": cfg,
        },
    }
