"""
EL2N baseline.

Paper:
- "Deep Learning on a Data Diet: Finding Important Examples Early in Training".

Definition:
- EL2N score = L2 norm of early-training prediction error vector.

Multimodal pair-level adaptation:
- Compute surrogate early-training EL2N on image/text pair branches.
- Aggregate branch scores into pair-level importance.

reproduction_status: faithful_but_practical (surrogate early-training pipeline)
"""

from typing import Any, Dict

import numpy as np

from baselines.common.train_utils import SurrogateConfig, run_surrogate_training
from baselines.registry import register_method
from ._utils import make_result


@register_method("el2n")
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
        epochs=int(cfg.get("surrogate_epochs", 5)),
        batch_size=int(cfg.get("surrogate_batch_size", 256)),
        proj_dim=int(cfg.get("surrogate_proj_dim", 128)),
        lr=float(cfg.get("surrogate_lr", 1e-2)),
        temperature=float(cfg.get("surrogate_temperature", 0.07)),
        seed=int(cfg.get("seed", 0)),
        device=str(cfg.get("device", "cpu")),
    )
    outputs = run_surrogate_training(img, txt, train_cfg)
    hist_img = outputs["history"]["el2n_img"]
    hist_txt = outputs["history"]["el2n_txt"]
    stacked_img = np.stack(hist_img, axis=0)
    stacked_txt = np.stack(hist_txt, axis=0)
    window = int(cfg.get("el2n_epoch_window", min(3, stacked_img.shape[0], stacked_txt.shape[0])))
    score_img = np.mean(stacked_img[:window], axis=0)
    score_txt = np.mean(stacked_txt[:window], axis=0)
    score_joint = (score_img + score_txt) / 2.0
    return make_result(
        method="el2n",
        ratio=ratio,
        n=img.shape[0],
        score_img=score_img,
        score_txt=score_txt,
        score_joint=score_joint,
        config=cfg,
        notes="Early-epoch L2 error norm using branch-specific surrogate dynamics.",
    )
