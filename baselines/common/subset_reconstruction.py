import os
from typing import Any, Dict, List, Optional

from baselines.common.feature_cache import load_feature_bundle, resolve_feature_dir
from baselines.common.io import load_json, save_json


def _read_selected_indices(selected_indices_path: str) -> List[int]:
    payload = load_json(selected_indices_path)
    values = payload.get("selected_indices", [])
    if not isinstance(values, list):
        raise ValueError(f"Invalid selected_indices format: {selected_indices_path}")
    out = [int(v) for v in values]
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate sample_idx found in {selected_indices_path}")
    return out


def build_subset_spec(
    *,
    baseline_result_dir: str,
    selected_indices_path: str,
    feature_cache_root: str,
    dataset_name: str,
    split: str,
    image_encoder: str,
    text_encoder: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    feature_dir = resolve_feature_dir(
        feature_cache_root=feature_cache_root,
        dataset_name=dataset_name,
        split=split,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    )
    bundle = load_feature_bundle(feature_dir)
    selected_indices = _read_selected_indices(selected_indices_path)
    sample_meta = bundle["sample_meta"]

    sample_idx_to_local = {}
    for local_idx, item in enumerate(sample_meta):
        sample_idx = int(item["sample_idx"])
        sample_idx_to_local[sample_idx] = int(local_idx)

    missing = [idx for idx in selected_indices if idx not in sample_idx_to_local]
    if missing:
        raise ValueError(
            "selected_indices contains sample_idx not found in feature cache sample_meta: "
            f"{missing[:10]} (total_missing={len(missing)})"
        )

    selected_local_indices = [sample_idx_to_local[idx] for idx in selected_indices]
    selected_meta = [sample_meta[local] for local in selected_local_indices]

    subset_spec = {
        "baseline_result_dir": baseline_result_dir,
        "selected_indices_path": selected_indices_path,
        "dataset_name": dataset_name,
        "split": split,
        "image_encoder": image_encoder,
        "text_encoder": text_encoder,
        "sample_unit": "pair_level_sample_idx",
        "feature_cache_root": feature_cache_root,
        "feature_dir": feature_dir,
        "subset_size": int(len(selected_indices)),
        "selected_indices": selected_indices,
        "selected_local_indices": selected_local_indices,
        "selected_meta_preview": selected_meta[:10],
        "all_indices_validated": True,
    }
    if output_path is None:
        output_path = os.path.join(baseline_result_dir, "subset_spec.json")
    save_json(output_path, subset_spec)
    subset_spec["subset_spec_path"] = output_path
    return subset_spec

