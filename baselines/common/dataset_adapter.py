from typing import Any, Dict

import numpy as np

from .feature_cache import load_feature_bundle, resolve_feature_dir


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def build_joint_features(image_features: np.ndarray, text_features: np.ndarray, mode: str = "concat") -> np.ndarray:
    if mode != "concat":
        raise ValueError(f"Unsupported joint feature mode: {mode}")
    return np.concatenate([l2_normalize(image_features), l2_normalize(text_features)], axis=1).astype(np.float32)


def load_multimodal_dataset_bundle(
    feature_cache_root: str,
    dataset_name: str,
    split: str,
    image_encoder: str,
    text_encoder: str,
    joint_mode: str = "concat",
) -> Dict[str, Any]:
    feature_dir = resolve_feature_dir(feature_cache_root, dataset_name, split, image_encoder, text_encoder)
    bundle = load_feature_bundle(feature_dir)
    image_features = bundle["image_features"]
    text_features = bundle["text_features"]
    joint_features = build_joint_features(image_features, text_features, mode=joint_mode)
    sample_indices = [int(item["sample_idx"]) for item in bundle["sample_meta"]]
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample_idx in sample_meta must be unique for pair-level selection.")
    return {
        **bundle,
        "joint_features": joint_features,
        "sample_indices": sample_indices,
        "sample_unit": "pair_level_sample_idx",
        "dataset_name": dataset_name,
        "split": split,
        "image_encoder": image_encoder,
        "text_encoder": text_encoder,
    }
