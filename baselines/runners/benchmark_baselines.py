import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import baselines.methods  # noqa: F401
from baselines.common.io import ensure_dir
from baselines.registry import list_methods
from baselines.runners.run_baseline_selection import load_config_chain, run_baseline_selection_once
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - graceful fallback when tqdm is unavailable
    tqdm = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark baselines with budget-aligned protocol.")
    parser.add_argument(
        "--config",
        type=str,
        default="baselines/configs/main_aligned_flickr_nfnet_bert.yaml",
    )
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--budgets", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--no_progress", action="store_true")
    return parser


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main():
    args = build_parser().parse_args()
    cfg = load_config_chain(args.config, method=None)
    methods = args.methods if args.methods else list_methods()
    budgets = args.budgets if args.budgets else list(cfg.get("budgets", [100, 200, 500]))
    seeds = args.seeds if args.seeds else list(cfg.get("default_seeds", [cfg.get("default_seed", 0)]))
    device = args.device if args.device else str(cfg.get("default_device", "cpu"))
    output_root = args.output_root if args.output_root else str(cfg.get("output_root", "artifacts/baselines"))
    ensure_dir(output_root)

    run_rows: List[Dict[str, Any]] = []
    grouped = defaultdict(list)

    allowed_methods = set(list_methods())
    jobs: List[Dict[str, int | str]] = []
    for method in methods:
        method_name = str(method).lower()
        if method_name not in allowed_methods:
            raise ValueError(f"Unknown method '{method_name}'. Available methods: {list_methods()}")
        for budget in budgets:
            for seed in seeds:
                jobs.append({"method": method_name, "budget": int(budget), "seed": int(seed)})

    use_progress = (not args.no_progress) and (tqdm is not None)
    iterator = tqdm(jobs, desc="benchmark", unit="job", dynamic_ncols=True) if use_progress else jobs

    for job in iterator:
        method_name = str(job["method"])
        budget = int(job["budget"])
        seed = int(job["seed"])
        if use_progress:
            iterator.set_postfix_str(f"{method_name}|b{budget}|s{seed}")  # type: ignore[attr-defined]
        record = run_baseline_selection_once(
            method=method_name,
            dataset_name=str(cfg.get("dataset_name", "flickr")),
            split=str(cfg.get("split", "train")),
            image_encoder=str(cfg.get("image_encoder", "nfnet")),
            text_encoder=str(cfg.get("text_encoder", "bert")),
            feature_source=str(cfg.get("feature_source", "artifacts/feature_cache")),
            output_root=output_root,
            seed=seed,
            device=device,
            budget=budget,
            config_path=args.config,
            runtime_config_overrides={
                "pair_score_fusion": cfg.get("pair_score_fusion", "weighted_sum"),
                "pair_score_weights": cfg.get("pair_score_weights", [0.5, 0.5, 0.0]),
                "score_normalization": cfg.get("score_normalization", "zscore"),
                "joint_feature_mode": cfg.get("joint_feature_mode", "concat"),
                "evaluation_protocol": cfg.get("evaluation_protocol", "main_aligned_pair_selection_v1"),
            },
            output_layout="budget",
        )
        row = {
            "dataset_name": str(cfg.get("dataset_name", "flickr")),
            "split": str(cfg.get("split", "train")),
            "image_encoder": str(cfg.get("image_encoder", "nfnet")),
            "text_encoder": str(cfg.get("text_encoder", "bert")),
            "method": record["method"],
            "budget": int(record["budget"]),
            "ratio": float(record["ratio"]),
            "total_train_size": int(record["total_train_size"]),
            "subset_size": int(record["subset_size"]),
            "seed": seed,
            "sample_unit": "pair_level_sample_idx",
            "feature_source": str(cfg.get("feature_source", "artifacts/feature_cache")),
            "selected_indices_path": record["paths"]["selected_indices"],
            "baseline_summary_path": record["paths"]["baseline_summary"],
        }
        run_rows.append(row)
        grouped[(row["method"], row["budget"])].append(row)
        print(
            f"[benchmark] method={row['method']} budget={row['budget']} "
            f"seed={row['seed']} ratio={row['ratio']:.6f}"
        )

    benchmark_summary_json = os.path.join(output_root, "benchmark_summary.json")
    benchmark_summary_csv = os.path.join(output_root, "benchmark_summary.csv")
    main_table_aligned_csv = os.path.join(output_root, "main_table_aligned.csv")

    agg_rows = []
    for (method, budget), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        agg_rows.append(
            {
                "dataset_name": str(cfg.get("dataset_name", "flickr")),
                "image_encoder": str(cfg.get("image_encoder", "nfnet")),
                "text_encoder": str(cfg.get("text_encoder", "bert")),
                "method": method,
                "budget": int(budget),
                "num_seeds": len(rows),
                "mean_ratio": _mean([float(r["ratio"]) for r in rows]),
                "mean_subset_size": _mean([float(r["subset_size"]) for r in rows]),
                "mean_total_train_size": _mean([float(r["total_train_size"]) for r in rows]),
                "feature_source": str(cfg.get("feature_source", "artifacts/feature_cache")),
                "sample_unit": "pair_level_sample_idx",
            }
        )

    payload = {
        "config": args.config,
        "dataset_name": str(cfg.get("dataset_name", "flickr")),
        "image_encoder": str(cfg.get("image_encoder", "nfnet")),
        "text_encoder": str(cfg.get("text_encoder", "bert")),
        "budgets": [int(x) for x in budgets],
        "seeds": [int(x) for x in seeds],
        "feature_source": str(cfg.get("feature_source", "artifacts/feature_cache")),
        "sample_unit": "pair_level_sample_idx",
        "runs": run_rows,
        "aggregated": agg_rows,
    }
    with open(benchmark_summary_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if run_rows:
        with open(benchmark_summary_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
    if agg_rows:
        with open(main_table_aligned_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)

    print("Benchmark finished:")
    print(f"  benchmark_summary_json: {benchmark_summary_json}")
    print(f"  benchmark_summary_csv: {benchmark_summary_csv}")
    print(f"  main_table_aligned_csv: {main_table_aligned_csv}")


if __name__ == "__main__":
    main()
