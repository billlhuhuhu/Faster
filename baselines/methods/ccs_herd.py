"""
CCS-herd / herd baseline.

Definition context:
- Herding-style exemplar mean approximation (e.g., exemplar selection in iCaRL-style literature).

Multimodal pair-level adaptation:
- Perform herding in joint feature space for pair samples.

CCS note:
- `ccs-herd` is a project-specific CCS variant: coverage wrapper + herding base selector.

reproduction_status: faithful_but_practical (herd), project_specific_variant (ccs-herd)
"""

from typing import Any, Dict

import numpy as np

from baselines.common.selection_utils import herding_select, resolve_subset_size
from baselines.registry import register_method


@register_method("ccs-herd")
@register_method("herd")
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
    selected = herding_select(joint, k)
    score = -np.linalg.norm(joint - np.mean(joint, axis=0, keepdims=True), axis=1).astype(np.float32)
    return {
        "method": "ccs-herd",
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
            "notes": "CCS group baseline: pair-level herding on joint features.",
            "config": cfg,
        },
    }
