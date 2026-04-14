"""
Dfool baseline (DeepFool/DFAL interpretation).

Paper context:
- DeepFool: "A Simple and Accurate Method to Fool Deep Neural Networks".
- DFAL-style active learning uses minimal adversarial perturbation as boundary proxy.

Adopted interpretation:
- DeepFool-based boundary proximity baseline.

Implementation note:
- Practical surrogate of DFAL/DeepFool score: uses retrieval margin inverse as a
  boundary-distance proxy rather than full iterative DeepFool perturbation.

Multimodal pair-level adaptation:
- Compute image/text branch boundary-proximity proxies and fuse to pair-level score.

reproduction_status: surrogate
"""

from typing import Any, Dict

import numpy as np

from baselines.common.train_utils import SurrogateConfig, cosine_similarity_matrix, run_surrogate_training
from baselines.registry import register_method
from ._utils import make_result


def _margin_score(sim: np.ndarray) -> np.ndarray:
    diag = np.diag(sim)
    masked = sim.copy()
    np.fill_diagonal(masked, -np.inf)
    hardest = np.max(masked, axis=1)
    margin = diag - hardest
    return margin


@register_method("dfool")
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
    sim = cosine_similarity_matrix(outputs["img_embed"], outputs["txt_embed"])
    margin = _margin_score(sim)
    # Assumed Dfool: smaller margin => easier to fool => higher selection score.
    score = (1.0 / np.maximum(margin, 1e-4)).astype(np.float32)
    return make_result(
        method="dfool",
        ratio=ratio,
        n=img.shape[0],
        score_img=score,
        score_txt=score,
        score_joint=score,
        config=cfg,
        notes="Assumed Dfool version: pair-level fooling score via inverse retrieval margin.",
    )
