"""
DQ baseline (Dataset Quantization adaptation).

Paper:
- "Dataset Quantization" (2023).

Adopted interpretation in this project:
- DQ refers to Dataset Quantization, not generic "data quality".
- Practical pair-level adaptation: quantize multimodal feature space into prototype
  representatives and select nearest pair samples.

Implementation note:
- This is an assumed practical DQ variant (representative quantization / clustering),
  not a strict full reproduction of all original optimization details.

reproduction_status: assumed_version
"""

from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans

from baselines.common.selection_utils import resolve_subset_size
from baselines.registry import register_method


@register_method("dq")
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
    joint = np.asarray(dataset["joint_features"], dtype=np.float32)
    n = int(joint.shape[0])
    k = resolve_subset_size(n, ratio)

    clusterer = KMeans(
        n_clusters=int(k),
        random_state=int(cfg.get("seed", 0)),
        n_init=10,
    )
    assign = clusterer.fit_predict(joint)
    centers = clusterer.cluster_centers_.astype(np.float32)

    center_per_sample = centers[assign]
    dist_joint = np.linalg.norm(joint - center_per_sample, axis=1).astype(np.float32)
    score_joint = -dist_joint

    img_centers = np.zeros((k, img.shape[1]), dtype=np.float32)
    txt_centers = np.zeros((k, txt.shape[1]), dtype=np.float32)
    for cid in range(int(k)):
        members = np.where(assign == cid)[0]
        if members.size == 0:
            continue
        img_centers[cid] = np.mean(img[members], axis=0)
        txt_centers[cid] = np.mean(txt[members], axis=0)
    score_img = -np.linalg.norm(img - img_centers[assign], axis=1).astype(np.float32)
    score_txt = -np.linalg.norm(txt - txt_centers[assign], axis=1).astype(np.float32)

    selected = []
    for cid in range(int(k)):
        members = np.where(assign == cid)[0]
        if members.size == 0:
            continue
        best_local = int(members[np.argmin(dist_joint[members])])
        selected.append(best_local)

    if len(selected) < int(k):
        order = np.argsort(-score_joint)
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= int(k):
                break

    score_pair = (
        float(cfg.get("dq_w_joint", 0.5)) * score_joint
        + float(cfg.get("dq_w_img", 0.25)) * score_img
        + float(cfg.get("dq_w_txt", 0.25)) * score_txt
    ).astype(np.float32)

    return {
        "method": "dq",
        "ratio": float(ratio),
        "selected_local_indices": selected[: int(k)],
        "scores": {
            "score_img": score_img.astype(np.float32),
            "score_txt": score_txt.astype(np.float32),
            "score_joint": score_joint.astype(np.float32),
            "score_pair": score_pair.astype(np.float32),
        },
        "meta": {
            "subset_size": int(k),
            "num_samples": int(n),
            "notes": "Assumed practical DQ variant: pair-level feature-space quantization with representative assignment.",
            "config": cfg,
        },
    }

