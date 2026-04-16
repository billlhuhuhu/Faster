"""
Dynamic pruning baseline (InfoBatch-style adaptation).

Source method:
- InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning.

Adopted interpretation in this project:
- Dynamic training-time pruning baseline adapted to pair-level multimodal retrieval.
- Uses surrogate training dynamics to emulate epoch-wise keep/drop behavior.
- Exports a static compatibility subset for unified downstream evaluation tables.

Important note:
- Not a strict static subset selector in the original sense.
- Current implementation is an adapted dynamic baseline with budget-aligned export.

reproduction_status: adapted_dynamic_baseline
"""

from typing import Any, Dict

import numpy as np

from baselines.common.selection_utils import resolve_subset_size, topk_indices
from baselines.common.train_utils import SurrogateConfig, run_surrogate_training
from baselines.registry import register_method


def _dynamic_keep_counts(loss_history: np.ndarray, keep_ratio: float) -> np.ndarray:
    """
    loss_history: [E, N], larger loss => more informative => higher keep priority.
    """
    epochs, n = loss_history.shape
    keep_k = max(1, min(n, int(round(float(keep_ratio) * float(n)))))
    keep_counts = np.zeros(n, dtype=np.float32)
    for ep in range(epochs):
        order = np.argsort(-loss_history[ep], kind="stable")
        keep = order[:keep_k]
        keep_counts[keep] += 1.0
    return keep_counts


@register_method("dynamic_pruning")
@register_method("infobatch")
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
    n = int(img.shape[0])
    k = resolve_subset_size(n, ratio)
    keep_ratio = float(k) / max(float(n), 1.0)

    train_cfg = SurrogateConfig(
        epochs=int(cfg.get("dynamic_pruning_epochs", cfg.get("surrogate_epochs", 6))),
        batch_size=int(cfg.get("dynamic_pruning_batch_size", cfg.get("surrogate_batch_size", 256))),
        proj_dim=int(cfg.get("dynamic_pruning_proj_dim", cfg.get("surrogate_proj_dim", 128))),
        lr=float(cfg.get("dynamic_pruning_lr", cfg.get("surrogate_lr", 1e-2))),
        temperature=float(cfg.get("dynamic_pruning_temperature", cfg.get("surrogate_temperature", 0.07))),
        seed=int(cfg.get("seed", 0)),
        device=str(cfg.get("device", "cpu")),
    )
    outputs = run_surrogate_training(img, txt, train_cfg)
    loss_hist = np.stack(outputs["history"]["loss"], axis=0).astype(np.float32)
    keep_counts = _dynamic_keep_counts(loss_hist, keep_ratio=keep_ratio)
    keep_freq = keep_counts / max(float(loss_hist.shape[0]), 1.0)
    mean_loss = np.mean(loss_hist, axis=0).astype(np.float32)

    # InfoBatch-style proxy score: frequently kept + still hard samples.
    score_pair = (0.7 * keep_freq + 0.3 * (mean_loss / max(float(np.max(mean_loss)), 1e-8))).astype(np.float32)
    selected = topk_indices(score_pair, k, largest=True)
    selected_mask = np.zeros(n, dtype=np.float32)
    selected_mask[selected] = 1.0

    return {
        "method": "dynamic_pruning",
        "ratio": float(ratio),
        "selected_local_indices": selected,
        "scores": {
            "score_img": keep_freq.astype(np.float32),
            "score_txt": mean_loss.astype(np.float32),
            "score_joint": keep_counts.astype(np.float32),
            "score_pair": score_pair.astype(np.float32),
            "dynamic_selected_mask": selected_mask.astype(np.float32),
        },
        "meta": {
            "subset_size": int(k),
            "num_samples": int(n),
            "dynamic_keep_ratio": keep_ratio,
            "dynamic_epochs": int(loss_hist.shape[0]),
            "effective_sample_coverage": float(np.mean((keep_counts > 0).astype(np.float32))),
            "notes": (
                "InfoBatch-style dynamic pruning adaptation. "
                "Static selected_indices export is a compatibility view for unified baseline pipeline."
            ),
            "config": cfg,
        },
    }

