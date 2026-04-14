import json
import os
from typing import Dict

import numpy as np
import torch

from .io import sanitize_name


def resolve_feature_dir(
    feature_cache_root: str,
    dataset_name: str,
    split: str,
    image_encoder: str,
    text_encoder: str,
) -> str:
    model_tag = f"{sanitize_name(image_encoder)}_{sanitize_name(text_encoder)}"
    return os.path.join(feature_cache_root, dataset_name, split, model_tag)


def _resolve_feature_paths(feature_dir: str) -> Dict[str, str]:
    img_selection = os.path.join(feature_dir, "img_features_selection.pt")
    txt_selection = os.path.join(feature_dir, "txt_features_selection.pt")
    if os.path.exists(img_selection) and os.path.exists(txt_selection):
        img_path = img_selection
        txt_path = txt_selection
    else:
        img_path = os.path.join(feature_dir, "img_features.pt")
        txt_path = os.path.join(feature_dir, "txt_features.pt")
    return {
        "img": img_path,
        "txt": txt_path,
        "meta": os.path.join(feature_dir, "sample_meta.json"),
        "info": os.path.join(feature_dir, "feature_info.json"),
    }


def load_feature_bundle(feature_dir: str) -> Dict[str, object]:
    paths = _resolve_feature_paths(feature_dir)
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing feature cache files: {missing}")

    img_features = torch.load(paths["img"], map_location="cpu").float().numpy().astype(np.float32)
    txt_features = torch.load(paths["txt"], map_location="cpu").float().numpy().astype(np.float32)
    with open(paths["meta"], "r", encoding="utf-8") as handle:
        sample_meta = json.load(handle)
    with open(paths["info"], "r", encoding="utf-8") as handle:
        feature_info = json.load(handle)

    if img_features.shape[0] != txt_features.shape[0] or img_features.shape[0] != len(sample_meta):
        raise ValueError(
            "Feature cache shape mismatch: "
            f"img={img_features.shape}, txt={txt_features.shape}, meta={len(sample_meta)}"
        )

    return {
        "image_features": img_features,
        "text_features": txt_features,
        "sample_meta": sample_meta,
        "feature_info": feature_info,
        "num_samples": int(img_features.shape[0]),
        "feature_dir": feature_dir,
    }

