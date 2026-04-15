import json
import time
from pathlib import Path

import numpy as np

from data.subset_dataset import save_selected_indices


def sanitize_name(name):
    return str(name).replace("\\", "-").replace("/", "-").replace(" ", "_")


def build_feature_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    return Path(args.feature_cache_root) / args.dataset / args.split / model_tag


def resolve_subset_size(num_samples, budget_ratio=None, budget_size=None):
    if budget_size is not None:
        subset_size = int(budget_size)
    elif budget_ratio is not None:
        subset_size = int(round(num_samples * float(budget_ratio)))
    else:
        raise ValueError("Either budget_ratio or budget_size must be provided.")
    subset_size = max(1, subset_size)
    subset_size = min(int(num_samples), subset_size)
    return subset_size


def build_budget_tag(args):
    if getattr(args, "budget_size", None) is not None:
        return f"size_{int(args.budget_size):04d}"
    if getattr(args, "budget_ratio", None) is not None:
        return f"ratio_{int(round(float(args.budget_ratio) * 100)):02d}"
    raise ValueError("Either budget_ratio or budget_size must be provided.")


def build_output_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    budget_tag = build_budget_tag(args)
    method_tag = sanitize_name(getattr(args, "selection_method", "random"))
    seed_tag = f"seed_{int(getattr(args, 'random_state', 0))}"
    return Path(args.output_root) / args.dataset / args.split / model_tag / budget_tag / method_tag / seed_tag


def load_sample_meta(feature_dir):
    feature_dir = Path(feature_dir)
    sample_meta_path = feature_dir / "sample_meta.json"
    if not sample_meta_path.exists():
        raise FileNotFoundError(f"sample_meta.json not found at {sample_meta_path}")
    with open(sample_meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_random_indices(num_samples, subset_size, random_state=0):
    rng = np.random.default_rng(int(random_state))
    selected = rng.choice(int(num_samples), size=int(subset_size), replace=False)
    selected = np.sort(selected.astype(np.int64))
    return selected.tolist()


def build_selected_meta(sample_meta, selected_indices):
    return [sample_meta[int(idx)] for idx in selected_indices]


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_random_subset_selection(args):
    feature_dir = build_feature_dir(args)
    sample_meta = load_sample_meta(feature_dir)
    num_samples = int(len(sample_meta))
    subset_size = resolve_subset_size(
        num_samples,
        budget_ratio=getattr(args, "budget_ratio", None),
        budget_size=getattr(args, "budget_size", None),
    )
    selected_indices = sample_random_indices(
        num_samples,
        subset_size=subset_size,
        random_state=getattr(args, "random_state", 0),
    )
    selected_meta = build_selected_meta(sample_meta, selected_indices)

    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_indices_path = output_dir / "selected_indices.json"
    selected_meta_path = output_dir / "selected_meta.json"
    summary_path = output_dir / "summary.json"

    save_selected_indices(selected_indices_path, selected_indices)
    save_json(selected_meta_path, selected_meta)
    save_json(
        summary_path,
        {
            "dataset": args.dataset,
            "split": args.split,
            "image_encoder": args.image_encoder,
            "text_encoder": args.text_encoder,
            "selection_method": getattr(args, "selection_method", "random"),
            "selection_seed": int(getattr(args, "random_state", 0)),
            "budget_ratio": float(args.budget_ratio) if getattr(args, "budget_ratio", None) is not None else None,
            "budget_size": int(subset_size),
            "requested_budget_size": None if getattr(args, "budget_size", None) is None else int(args.budget_size),
            "requested_budget_ratio": None if getattr(args, "budget_ratio", None) is None else float(args.budget_ratio),
            "num_samples": int(num_samples),
            "feature_dir": str(feature_dir),
            "selected_indices_path": str(selected_indices_path),
            "selected_meta_path": str(selected_meta_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    return {
        "output_dir": str(output_dir),
        "selected_indices_path": str(selected_indices_path),
        "selected_meta_path": str(selected_meta_path),
        "summary_path": str(summary_path),
        "subset_size": int(subset_size),
        "selected_indices": selected_indices,
    }
