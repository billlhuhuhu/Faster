"""
GLISTER baseline.

Paper:
- "GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning".

Definition:
- Validation-driven (bi-level style) subset selection maximizing generalization gain.

Multimodal pair-level adaptation:
- Pair sample is the selection unit.
- Image/text branch information contributes to pair-level gradient/gain proxy.

reproduction_status: faithful_but_practical (online/taylor-style surrogate approximation)
"""

from typing import Any, Dict

import numpy as np

from baselines.common.selection_utils import glister_greedy, resolve_subset_size
from baselines.common.train_utils import (
    SurrogateConfig,
    build_sample_gradients,
    run_surrogate_training,
    split_train_val,
)
from baselines.registry import register_method


@register_method("glister")
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
    split = split_train_val(grads.shape[0], val_ratio=float(cfg.get("glister_val_ratio", 0.1)), seed=int(cfg.get("seed", 0)))
    train_idx = split["train"]
    val_idx = split["val"]
    train_grads = grads[train_idx]
    val_grad = np.mean(grads[val_idx], axis=0)

    target_k = resolve_subset_size(grads.shape[0], ratio)
    k_train = min(int(target_k), int(train_grads.shape[0]))
    selected_train_local, residual = glister_greedy(train_grads, val_grad, k_train)
    selected = [int(train_idx[idx]) for idx in selected_train_local]
    score = (grads @ val_grad).astype(np.float32)

    # Budget-equality fix:
    # GLISTER greedy runs on train split; if train subset is smaller than target budget,
    # fill the remainder from non-selected samples by global validation-gain proxy.
    if len(selected) < int(target_k):
        picked = set(selected)
        order = np.argsort(-score)
        for idx in order:
            idx = int(idx)
            if idx in picked:
                continue
            selected.append(idx)
            picked.add(idx)
            if len(selected) >= int(target_k):
                break
    selected = selected[: int(target_k)]
    return {
        "method": "glister",
        "ratio": float(ratio),
        "selected_local_indices": selected,
        "scores": {
            "score_img": score,
            "score_txt": score,
            "score_joint": score,
        },
        "meta": {
            "subset_size": int(target_k),
            "num_samples": int(grads.shape[0]),
            "train_size": int(train_idx.shape[0]),
            "val_size": int(val_idx.shape[0]),
            "residual_norm": float(np.linalg.norm(residual)),
            "budget_equalized": True,
            "notes": "Practical GLISTER approximation using held-out surrogate validation gradient.",
            "config": cfg,
        },
    }
