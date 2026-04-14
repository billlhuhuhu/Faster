"""
GradMatch baseline.

Paper:
- "GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training".

Definition:
- Select a subset whose gradients match full-data (or target) gradients.

Multimodal pair-level adaptation:
- Build pair-level gradient representation by combining image/text branch gradients.
- Use greedy gradient matching on pair samples.

reproduction_status: faithful_but_practical (surrogate gradient representation)
"""

from typing import Any, Dict

import numpy as np

from baselines.common.selection_utils import gradmatch_greedy, resolve_subset_size
from baselines.common.train_utils import (
    SurrogateConfig,
    build_sample_gradients,
    run_surrogate_training,
)
from baselines.registry import register_method


@register_method("gradmatch")
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
    grads = build_sample_gradients(outputs["img_embed"], outputs["txt_embed"])
    target_grad = np.mean(grads, axis=0)
    k = resolve_subset_size(grads.shape[0], ratio)
    selected, residual = gradmatch_greedy(grads, target_grad, k)
    score = (grads @ target_grad).astype(np.float32)
    return {
        "method": "gradmatch",
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
            "num_samples": int(grads.shape[0]),
            "residual_norm": float(np.linalg.norm(residual)),
            "notes": "Practical GradMatch approximation on surrogate pair gradients.",
            "config": cfg,
        },
    }
