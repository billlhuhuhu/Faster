"""
CCS-forget / forget baseline.

Paper:
- "An Empirical Study of Example Forgetting during Deep Neural Network Learning".

Definition:
- Count forgetting events: transitions from learned/correct to unlearned/incorrect.

Multimodal pair-level adaptation:
- Track surrogate forgetting dynamics for image/text branches and aggregate to pair-level.

CCS note:
- `ccs-forget` is a project-specific CCS variant: coverage wrapper + forgetting base selector.

reproduction_status: faithful_but_practical (forget), project_specific_variant (ccs-forget)
"""

from typing import Any, Dict

import numpy as np

from baselines.common.train_utils import SurrogateConfig, compute_forgetting_counts, run_surrogate_training
from baselines.registry import register_method
from ._utils import make_result


@register_method("ccs-forget")
@register_method("forget")
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
    train_cfg = SurrogateConfig(
        epochs=int(cfg.get("surrogate_epochs", 8)),
        batch_size=int(cfg.get("surrogate_batch_size", 256)),
        proj_dim=int(cfg.get("surrogate_proj_dim", 128)),
        lr=float(cfg.get("surrogate_lr", 1e-2)),
        temperature=float(cfg.get("surrogate_temperature", 0.07)),
        seed=int(cfg.get("seed", 0)),
        device=str(cfg.get("device", "cpu")),
    )
    outputs = run_surrogate_training(img, txt, train_cfg)
    conf_hist = outputs["history"]["confidence"]
    forget_score = compute_forgetting_counts(conf_hist, threshold=float(cfg.get("forget_threshold", 0.5)))
    return make_result(
        method="ccs-forget",
        ratio=ratio,
        n=img.shape[0],
        score_img=forget_score,
        score_txt=forget_score,
        score_joint=forget_score,
        config=cfg,
        notes="CCS group baseline: forgetting events from surrogate pair training dynamics.",
    )
