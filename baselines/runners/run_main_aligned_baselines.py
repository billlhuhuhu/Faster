import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List

import baselines.methods  # noqa: F401
from baselines.common.io import ensure_dir
from baselines.registry import list_methods
from baselines.runners.run_baseline_selection import load_config_chain, run_baseline_selection_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baselines with main-experiment-aligned protocol.")
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
    return parser


def _as_list(value: Any, fallback: List[Any]) -> List[Any]:
    if value is None:
        return list(fallback)
    if isinstance(value, list):
        return list(value)
    return [value]


def main():
    args = build_parser().parse_args()
    cfg = load_config_chain(args.config, method=None)

    methods = args.methods if args.methods else _as_list(cfg.get("methods"), list_methods())
    budgets = args.budgets if args.budgets else _as_list(cfg.get("budgets"), [100, 200, 500])
    seeds = args.seeds if args.seeds else _as_list(cfg.get("default_seeds"), [cfg.get("default_seed", 0)])
    device = args.device if args.device else str(cfg.get("default_device", "cpu"))
    output_root = args.output_root if args.output_root else str(cfg.get("output_root", "artifacts/baselines"))
    log_root = str(cfg.get("log_root", os.path.join(output_root, "logs")))
    ensure_dir(output_root)
    ensure_dir(log_root)

    run_records: List[Dict[str, Any]] = []
    for method in methods:
        method_name = str(method).lower()
        if method_name not in set(list_methods()):
            raise ValueError(f"Unknown method '{method_name}'. Available methods: {list_methods()}")
        for budget in budgets:
            for seed in seeds:
                record = run_baseline_selection_once(
                    method=method_name,
                    dataset_name=str(cfg.get("dataset_name", "flickr")),
                    split=str(cfg.get("split", "train")),
                    image_encoder=str(cfg.get("image_encoder", "nfnet")),
                    text_encoder=str(cfg.get("text_encoder", "bert")),
                    feature_source=str(cfg.get("feature_source", "artifacts/feature_cache")),
                    output_root=output_root,
                    seed=int(seed),
                    device=device,
                    budget=int(budget),
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
                run_records.append(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "method": record["method"],
                        "budget": record["budget"],
                        "ratio": record["ratio"],
                        "subset_size": record["subset_size"],
                        "total_train_size": record["total_train_size"],
                        "seed": int(seed),
                        "dataset_name": str(cfg.get("dataset_name", "flickr")),
                        "image_encoder": str(cfg.get("image_encoder", "nfnet")),
                        "text_encoder": str(cfg.get("text_encoder", "bert")),
                        "feature_source": str(cfg.get("feature_source", "artifacts/feature_cache")),
                        "sample_unit": "pair_level_sample_idx",
                        "output_dir": record["output_dir"],
                        "selected_indices_path": record["paths"]["selected_indices"],
                        "baseline_summary_path": record["paths"]["baseline_summary"],
                    }
                )
                print(
                    f"[main-aligned] method={record['method']} budget={record['budget']} "
                    f"seed={seed} -> {record['paths']['selected_indices']}"
                )

    summary_json = os.path.join(output_root, "main_aligned_run_records.json")
    summary_csv = os.path.join(output_root, "main_aligned_run_records.csv")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(run_records, handle, ensure_ascii=False, indent=2)
    if run_records:
        with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_records[0].keys()))
            writer.writeheader()
            writer.writerows(run_records)

    print("Main-aligned baseline run finished:")
    print(f"  records_json: {summary_json}")
    print(f"  records_csv: {summary_csv}")


if __name__ == "__main__":
    main()

