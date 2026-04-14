import json
import os
import time
from typing import Any, Dict, Iterable, List


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_name(name: str) -> str:
    return str(name).replace("\\", "-").replace("/", "-").replace(" ", "_")


def ratio_tag(ratio: float) -> str:
    return f"ratio_{int(round(float(ratio) * 100)):02d}"


def as_int_list(values: Iterable[int]) -> List[int]:
    return [int(v) for v in values]


def save_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_selection_outputs(
    output_dir: str,
    method: str,
    ratio: float,
    budget: int,
    total_train_size: int,
    selected_indices: Iterable[int],
    scores: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, str]:
    import numpy as np

    ensure_dir(output_dir)
    selected_indices = as_int_list(selected_indices)
    selected_path = os.path.join(output_dir, "selected_indices.json")
    score_path = os.path.join(output_dir, "selection_scores.npz")
    summary_path = os.path.join(output_dir, "baseline_summary.json")

    save_json(selected_path, {"selected_indices": selected_indices})
    np.savez(score_path, **{k: np.asarray(v) for k, v in scores.items()})
    summary_payload = {
        "method": str(method),
        "ratio": float(ratio),
        "budget": int(budget),
        "total_train_size": int(total_train_size),
        "sample_unit": "pair_level_sample_idx",
        "subset_size": int(len(selected_indices)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
    }
    save_json(summary_path, summary_payload)
    return {
        "selected_indices": selected_path,
        "selection_scores": score_path,
        "baseline_summary": summary_path,
    }
