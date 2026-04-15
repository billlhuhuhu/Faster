import argparse
import os
import time
from typing import Any, Dict, Optional

from baselines.common.io import ratio_tag, sanitize_name, save_selection_outputs
from baselines.common.multimodal_scoring import fuse_pair_scores
from baselines.common.selection_utils import resolve_budget_and_ratio
from baselines.registry import get_method, list_methods


def _load_yaml_with_fallback(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return dict(payload)
    except Exception:
        payload: Dict[str, Any] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    items = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                    parsed = []
                    for item in items:
                        if item.lower() in {"on", "off", "true", "false"}:
                            parsed.append(item.lower() in {"on", "true"})
                            continue
                        try:
                            parsed.append(int(item))
                            continue
                        except ValueError:
                            pass
                        try:
                            parsed.append(float(item))
                            continue
                        except ValueError:
                            pass
                        parsed.append(item)
                    payload[key] = parsed
                    continue
                if value.lower() in {"on", "off", "true", "false"}:
                    payload[key] = value.lower() in {"on", "true"}
                    continue
                try:
                    payload[key] = int(value)
                    continue
                except ValueError:
                    pass
                try:
                    payload[key] = float(value)
                    continue
                except ValueError:
                    pass
                payload[key] = value
        return payload


def _resolve_config_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Config file not found: {path}")


def load_config_chain(config_path: Optional[str], method: Optional[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    visited = set()

    def load_one(path: str):
        resolved = _resolve_config_path(path)
        if resolved in visited:
            return
        visited.add(resolved)
        payload = _load_yaml_with_fallback(resolved)
        parent = payload.pop("base_config", None)
        if parent:
            load_one(str(parent))
        cfg.update(payload)

    default_cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "default.yaml")
    if os.path.exists(default_cfg):
        load_one(default_cfg)
    if config_path:
        load_one(config_path)
    if method:
        method_cfg = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            f"{sanitize_name(method).replace('-', '_')}.yaml",
        )
        if os.path.exists(method_cfg):
            load_one(method_cfg)
    return cfg


def parse_pair_weights(value: Any) -> list:
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(x) for x in str(value).split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent multimodal baseline subset selection.")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--image_encoder", type=str, default="nfnet")
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--feature_source", type=str, default="artifacts/feature_cache")
    parser.add_argument("--output_dir", type=str, default="artifacts/baselines")
    parser.add_argument("--pair_score_fusion", type=str, default="weighted_sum", choices=["average", "weighted_sum", "max", "geometric_mean", "normalized_sum"])
    parser.add_argument("--score_normalization", type=str, default="zscore", choices=["none", "zscore", "minmax", "rank"])
    parser.add_argument("--pair_score_weights", type=str, default="0.5,0.5,0.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--list_methods", action="store_true")
    parser.add_argument("--output_layout", type=str, default="ratio", choices=["ratio", "budget"])
    parser.add_argument("--candidate_pool_size", type=int, default=None)
    parser.add_argument("--candidate_pool_mode", type=str, default="head", choices=["head", "random"])
    return parser


def _merge_runtime_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(cfg)
    out.update(
        {
            "seed": int(args.seed),
            "device": str(args.device),
            "pair_score_fusion": str(args.pair_score_fusion),
            "score_normalization": str(args.score_normalization),
            "pair_score_weights": parse_pair_weights(args.pair_score_weights),
            "dataset_name": str(args.dataset_name),
            "image_encoder": str(args.image_encoder),
            "text_encoder": str(args.text_encoder),
            "split": str(args.split),
            "feature_source": str(args.feature_source),
            "output_root": str(args.output_dir),
            "candidate_pool_size": args.candidate_pool_size,
            "candidate_pool_mode": str(args.candidate_pool_mode),
        }
    )
    return out


def _apply_candidate_pool(dataset: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    pool_size = cfg.get("candidate_pool_size")
    if pool_size is None:
        return dataset
    n = int(dataset["num_samples"])
    k = max(1, min(int(pool_size), n))
    if k >= n:
        return dataset

    mode = str(cfg.get("candidate_pool_mode", "head")).lower()
    if mode == "head":
        local_idx = list(range(k))
    elif mode == "random":
        import numpy as np
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        local_idx = [int(x) for x in np.sort(rng.choice(n, size=k, replace=False))]
    else:
        raise ValueError(f"Unsupported candidate_pool_mode={mode}")

    def _slice(value):
        try:
            return value[local_idx]
        except Exception:
            return value

    narrowed = dict(dataset)
    for key in ("image_features", "text_features", "joint_features"):
        narrowed[key] = _slice(dataset[key])
    narrowed["sample_indices"] = [int(dataset["sample_indices"][i]) for i in local_idx]
    if "sample_meta" in dataset and isinstance(dataset["sample_meta"], list):
        narrowed["sample_meta"] = [dataset["sample_meta"][i] for i in local_idx]
    narrowed["num_samples"] = int(k)
    narrowed["candidate_pool_size"] = int(k)
    narrowed["candidate_pool_mode"] = mode
    return narrowed


def _enforce_fusion_contract(outputs: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    scores = dict(outputs.get("scores", {}))
    if "score_img" not in scores or "score_txt" not in scores:
        raise ValueError(
            f"Method {outputs.get('method')} must provide score_img and score_txt. "
            "score_pair is forbidden as primary output and will be recomputed externally."
        )
    score_img = scores["score_img"]
    score_txt = scores["score_txt"]
    score_joint = scores.get("score_joint")
    fused = fuse_pair_scores(
        score_img=score_img,
        score_txt=score_txt,
        score_joint=score_joint,
        fusion=str(cfg.get("pair_score_fusion", "weighted_sum")),
        weights=tuple(cfg.get("pair_score_weights", [0.5, 0.5, 0.0])),
        normalize_mode=str(cfg.get("score_normalization", "zscore")),
    )
    outputs = dict(outputs)
    outputs["scores"] = fused
    meta = dict(outputs.get("meta", {}))
    meta["fusion_enforced"] = True
    meta["fusion_mode"] = str(cfg.get("pair_score_fusion", "weighted_sum"))
    meta["score_normalization"] = str(cfg.get("score_normalization", "zscore"))
    outputs["meta"] = meta
    return outputs


def run_baseline_selection_once(
    *,
    method: str,
    dataset_name: str,
    split: str,
    image_encoder: str,
    text_encoder: str,
    feature_source: str,
    output_root: str,
    seed: int,
    device: str,
    ratio: Optional[float] = None,
    budget: Optional[int] = None,
    config_path: Optional[str] = None,
    runtime_config_overrides: Optional[Dict[str, Any]] = None,
    output_layout: str = "ratio",
    candidate_pool_size: Optional[int] = None,
    candidate_pool_mode: str = "head",
) -> Dict[str, Any]:
    import baselines.methods  # noqa: F401
    from baselines.common.dataset_adapter import load_multimodal_dataset_bundle

    base_cfg = load_config_chain(config_path, method=method)
    cfg = dict(base_cfg)
    if runtime_config_overrides:
        cfg.update(runtime_config_overrides)
    cfg.update(
        {
            "seed": int(seed),
            "device": str(device),
            "dataset_name": str(dataset_name),
            "split": str(split),
            "image_encoder": str(image_encoder),
            "text_encoder": str(text_encoder),
            "feature_source": str(feature_source),
            "output_root": str(output_root),
            "candidate_pool_size": candidate_pool_size,
            "candidate_pool_mode": candidate_pool_mode,
        }
    )
    cfg["pair_score_weights"] = parse_pair_weights(cfg.get("pair_score_weights", [0.5, 0.5, 0.0]))

    dataset = load_multimodal_dataset_bundle(
        feature_cache_root=feature_source,
        dataset_name=dataset_name,
        split=split,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    )
    dataset = _apply_candidate_pool(dataset, cfg)

    total_train_size = int(dataset["num_samples"])
    resolved_budget, resolved_ratio = resolve_budget_and_ratio(total_train_size, ratio=ratio, budget=budget)
    cfg["budget"] = int(resolved_budget)
    cfg["ratio"] = float(resolved_ratio)

    method_fn = get_method(method)
    select_started = time.time()
    outputs = method_fn(
        dataset=dataset,
        ratio=resolved_ratio,
        image_features=dataset["image_features"],
        text_features=dataset["text_features"],
        config=cfg,
    )
    outputs = _enforce_fusion_contract(outputs, cfg)
    selection_time = float(time.time() - select_started)
    selected_local = [int(x) for x in outputs["selected_local_indices"]]
    selected_local = selected_local[:resolved_budget]
    if len(set(selected_local)) != len(selected_local):
        raise ValueError(f"Method {method} returned duplicate local indices.")
    if any(x < 0 or x >= total_train_size for x in selected_local):
        raise ValueError(f"Method {method} returned out-of-range local index.")

    sample_idx = [int(dataset["sample_indices"][x]) for x in selected_local]
    if len(set(sample_idx)) != len(sample_idx):
        raise ValueError(f"Method {method} produced duplicate pair-level sample_idx.")

    model_tag = f"{sanitize_name(image_encoder)}_{sanitize_name(text_encoder)}"
    if output_layout == "budget":
        run_dir = os.path.join(
            output_root,
            dataset_name,
            model_tag,
            sanitize_name(outputs["method"]),
            f"budget_{int(resolved_budget):04d}",
            f"seed_{int(seed)}",
        )
    else:
        run_dir = os.path.join(
            output_root,
            dataset_name,
            split,
            model_tag,
            ratio_tag(resolved_ratio),
            sanitize_name(outputs["method"]),
            f"seed_{int(seed)}",
        )

    saved = save_selection_outputs(
        output_dir=run_dir,
        method=outputs["method"],
        ratio=resolved_ratio,
        budget=resolved_budget,
        total_train_size=total_train_size,
        selected_indices=sample_idx,
        scores=outputs["scores"],
        meta={
            "method_meta": outputs.get("meta", {}),
            "dataset_name": dataset_name,
            "split": split,
            "image_encoder": image_encoder,
            "text_encoder": text_encoder,
            "feature_source": dataset["feature_dir"],
            "pair_score_fusion": cfg.get("pair_score_fusion", "weighted_sum"),
            "pair_score_weights": cfg["pair_score_weights"],
            "score_normalization": cfg.get("score_normalization", "zscore"),
            "selected_local_indices": selected_local,
            "sample_unit": "pair_level_sample_idx",
            "evaluation_protocol": cfg.get("evaluation_protocol", "main_aligned_pair_selection"),
            "joint_feature_mode": cfg.get("joint_feature_mode", "concat"),
            "seed": int(seed),
            "selection_time": selection_time,
            "candidate_pool_size": cfg.get("candidate_pool_size"),
            "candidate_pool_mode": cfg.get("candidate_pool_mode", "head"),
        },
    )
    return {
        "method": outputs["method"],
        "ratio": float(resolved_ratio),
        "budget": int(resolved_budget),
        "total_train_size": total_train_size,
        "subset_size": len(sample_idx),
        "output_dir": run_dir,
        "paths": saved,
        "selected_indices": sample_idx,
        "config": cfg,
        "selection_time": selection_time,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
    os.environ.setdefault("BLIS_NUM_THREADS", "8")
    if args.list_methods:
        import baselines.methods  # noqa: F401

        print("Available methods:")
        for item in list_methods():
            print(f"  - {item}")
        return

    cfg = load_config_chain(args.config, method=args.method)
    cfg = _merge_runtime_overrides(cfg, args)
    runtime_overrides = {
        "pair_score_fusion": cfg["pair_score_fusion"],
        "score_normalization": cfg["score_normalization"],
        "pair_score_weights": cfg["pair_score_weights"],
    }
    out = run_baseline_selection_once(
        method=args.method,
        dataset_name=args.dataset_name,
        split=args.split,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        feature_source=args.feature_source,
        output_root=args.output_dir,
        seed=args.seed,
        device=args.device,
        ratio=args.ratio,
        budget=args.budget,
        config_path=args.config,
        runtime_config_overrides=runtime_overrides,
        output_layout=args.output_layout,
        candidate_pool_size=args.candidate_pool_size,
        candidate_pool_mode=args.candidate_pool_mode,
    )
    print("Baseline selection finished:")
    print(f"  method: {out['method']}")
    print(f"  budget: {out['budget']}")
    print(f"  ratio: {out['ratio']:.6f}")
    print(f"  total_train_size: {out['total_train_size']}")
    print(f"  subset_size: {out['subset_size']}")
    print(f"  selection_time: {out['selection_time']:.3f}s")
    print(f"  output_dir: {out['output_dir']}")
    print(f"  selected_indices: {out['paths']['selected_indices']}")
    print(f"  selection_scores: {out['paths']['selection_scores']}")
    print(f"  baseline_summary: {out['paths']['baseline_summary']}")


if __name__ == "__main__":
    main()
