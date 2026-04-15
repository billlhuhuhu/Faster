import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from baselines.common.io import ensure_dir
from baselines.common.io import sanitize_name
from baselines.common.result_aggregation import export_all_tables
from baselines.registry import list_methods
from baselines.runners.evaluate_baseline_subsets import run_downstream_eval
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
    parser.add_argument("--run_selection_only", action="store_true")
    parser.add_argument("--run_eval_only", action="store_true")
    parser.add_argument("--run_full_pipeline", action="store_true")
    return parser


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mean_opt(values: List[Any]) -> float:
    numeric = []
    for v in values:
        if v is None:
            continue
        try:
            numeric.append(float(v))
        except Exception:
            continue
    return _mean(numeric)


def _expected_run_dir(output_root: str, dataset_name: str, image_encoder: str, text_encoder: str, method: str, budget: int, seed: int) -> str:
    model_tag = f"{sanitize_name(image_encoder)}_{sanitize_name(text_encoder)}"
    return os.path.join(
        output_root,
        dataset_name,
        model_tag,
        sanitize_name(method),
        f"budget_{int(budget):04d}",
        f"seed_{int(seed)}",
    )


def _configure_thread_env(cfg: Dict[str, Any]) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cfg.get("openblas_num_threads", 8)))
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.get("omp_num_threads", 8)))
    os.environ.setdefault("MKL_NUM_THREADS", str(cfg.get("mkl_num_threads", 8)))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cfg.get("numexpr_num_threads", 8)))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(cfg.get("veclib_maximum_threads", 8)))
    os.environ.setdefault("BLIS_NUM_THREADS", str(cfg.get("blis_num_threads", 8)))


def main():
    args = build_parser().parse_args()
    cfg = load_config_chain(args.config, method=None)
    _configure_thread_env(cfg)
    import baselines.methods  # noqa: F401
    methods = args.methods if args.methods else list_methods()
    budgets = args.budgets if args.budgets else list(cfg.get("budgets", [100, 200, 500]))
    seeds = args.seeds if args.seeds else list(cfg.get("default_seeds", [cfg.get("default_seed", 0)]))
    device = args.device if args.device else str(cfg.get("default_device", "cpu"))
    output_root = args.output_root if args.output_root else str(cfg.get("output_root", "artifacts/baselines"))
    ensure_dir(output_root)
    eval_output_root = str(cfg.get("eval_output_root", output_root))
    ensure_dir(eval_output_root)

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
    run_mode = "full_pipeline"
    if args.run_selection_only:
        run_mode = "selection_only"
    if args.run_eval_only:
        run_mode = "eval_only"
    if args.run_full_pipeline:
        run_mode = "full_pipeline"

    for job in iterator:
        method_name = str(job["method"])
        budget = int(job["budget"])
        seed = int(job["seed"])
        if use_progress:
            iterator.set_postfix_str(f"{method_name}|b{budget}|s{seed}")  # type: ignore[attr-defined]
        baseline_run_dir = _expected_run_dir(
            output_root=output_root,
            dataset_name=str(cfg.get("dataset_name", "flickr")),
            image_encoder=str(cfg.get("image_encoder", "nfnet")),
            text_encoder=str(cfg.get("text_encoder", "bert")),
            method=method_name,
            budget=budget,
            seed=seed,
        )
        record = {}
        if run_mode in {"selection_only", "full_pipeline"}:
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
                candidate_pool_size=cfg.get("candidate_pool_size"),
                candidate_pool_mode=str(cfg.get("candidate_pool_mode", "head")),
            )
            baseline_run_dir = record["output_dir"]
        elif not os.path.exists(os.path.join(baseline_run_dir, "selected_indices.json")):
            raise FileNotFoundError(
                f"run_eval_only requires existing selection outputs, but not found: {baseline_run_dir}"
            )

        downstream_metrics = None
        if run_mode in {"eval_only", "full_pipeline"}:
            downstream_metrics = run_downstream_eval(
                baseline_result_dir=baseline_run_dir,
                selected_indices_path=os.path.join(baseline_run_dir, "selected_indices.json"),
                dataset_name=str(cfg.get("dataset_name", "flickr")),
                split=str(cfg.get("split", "train")),
                image_encoder=str(cfg.get("image_encoder", "nfnet")),
                text_encoder=str(cfg.get("text_encoder", "bert")),
                budget=budget,
                seed=seed,
                device=device,
                feature_source=str(cfg.get("feature_source", "artifacts/feature_cache")),
                image_root=str(cfg.get("image_root", "data/flickr30k")),
                ann_root=str(cfg.get("ann_root", "data/Flickr30k_ann")),
                train_entry=str(cfg.get("retrieval_train_entry", "run_subset_train.py")),
                output_dir=eval_output_root,
                epochs=int(cfg.get("train_epochs", 20)),
                batch_size_train=int(cfg.get("train_batch_size", 64)),
                batch_size_test=int(cfg.get("test_batch_size", 128)),
                text_batch_size=int(cfg.get("text_batch_size", 1024)),
                num_workers=int(cfg.get("num_workers", cfg.get("default_num_workers", 4))),
                eval_interval=int(cfg.get("eval_interval", 1)),
                subset_tag=method_name,
                subset_restore_mode=str(cfg.get("subset_restore_mode", "pair_level_indices")),
                no_aug=bool(cfg.get("train_no_aug", True)),
            )

        row = {
            "dataset_name": str(cfg.get("dataset_name", "flickr")),
            "split": str(cfg.get("split", "train")),
            "image_encoder": str(cfg.get("image_encoder", "nfnet")),
            "text_encoder": str(cfg.get("text_encoder", "bert")),
            "method": method_name,
            "budget": int(budget),
            "ratio": record.get("ratio"),
            "total_train_size": record.get("total_train_size"),
            "subset_size": record.get("subset_size"),
            "seed": seed,
            "sample_unit": "pair_level_sample_idx",
            "feature_source": str(cfg.get("feature_source", "artifacts/feature_cache")),
            "selected_indices_path": os.path.join(baseline_run_dir, "selected_indices.json"),
            "baseline_summary_path": os.path.join(baseline_run_dir, "baseline_summary.json"),
            "downstream_metrics_path": os.path.join(baseline_run_dir, "downstream_metrics.json")
            if os.path.exists(os.path.join(baseline_run_dir, "downstream_metrics.json"))
            else None,
            "I2T_R1": None if downstream_metrics is None else downstream_metrics.get("I2T_R1"),
            "I2T_R5": None if downstream_metrics is None else downstream_metrics.get("I2T_R5"),
            "I2T_R10": None if downstream_metrics is None else downstream_metrics.get("I2T_R10"),
            "T2I_R1": None if downstream_metrics is None else downstream_metrics.get("T2I_R1"),
            "T2I_R5": None if downstream_metrics is None else downstream_metrics.get("T2I_R5"),
            "T2I_R10": None if downstream_metrics is None else downstream_metrics.get("T2I_R10"),
            "MeanRecall": None if downstream_metrics is None else downstream_metrics.get("MeanRecall"),
            "selection_time": record.get("selection_time"),
            "train_time": None if downstream_metrics is None else downstream_metrics.get("train_time"),
            "eval_time": None if downstream_metrics is None else downstream_metrics.get("eval_time"),
        }
        run_rows.append(row)
        grouped[(row["method"], row["budget"])].append(row)
        print(
            f"[benchmark] method={row['method']} budget={row['budget']} "
            f"seed={row['seed']} mean_recall={row['MeanRecall']}"
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
                "mean_ratio": _mean_opt([r.get("ratio") for r in rows]),
                "mean_subset_size": _mean_opt([r.get("subset_size") for r in rows]),
                "mean_total_train_size": _mean_opt([r.get("total_train_size") for r in rows]),
                "I2T_R1_mean": _mean_opt([r.get("I2T_R1") for r in rows]),
                "I2T_R5_mean": _mean_opt([r.get("I2T_R5") for r in rows]),
                "I2T_R10_mean": _mean_opt([r.get("I2T_R10") for r in rows]),
                "T2I_R1_mean": _mean_opt([r.get("T2I_R1") for r in rows]),
                "T2I_R5_mean": _mean_opt([r.get("T2I_R5") for r in rows]),
                "T2I_R10_mean": _mean_opt([r.get("T2I_R10") for r in rows]),
                "MeanRecall_mean": _mean_opt([r.get("MeanRecall") for r in rows]),
                "selection_time_mean": _mean_opt([r.get("selection_time") for r in rows]),
                "train_time_mean": _mean_opt([r.get("train_time") for r in rows]),
                "eval_time_mean": _mean_opt([r.get("eval_time") for r in rows]),
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

    exported = export_all_tables(
        root=output_root,
        output_dir=output_root,
        budgets=[int(x) for x in budgets],
        methods=[str(x).lower() for x in methods],
        mapping_doc_path=os.path.join("baselines", "docs", "method_mapping.md"),
        expected_dataset=str(cfg.get("dataset_name", "flickr")),
        expected_image_encoder=str(cfg.get("image_encoder", "nfnet")),
        expected_text_encoder=str(cfg.get("text_encoder", "bert")),
    )
    print("Unified tables exported:")
    print(f"  main_table_aligned_json: {exported['main_table_json']}")
    print(f"  main_table_aligned_agg_csv: {exported['main_table_agg_csv']}")
    print(f"  main_table_wide_csv: {exported['main_table_wide_csv']}")
    print(f"  baseline_method_status_csv: {exported['method_status_csv']}")
    print(f"  baseline_protocol_alignment_csv: {exported['protocol_alignment_csv']}")


if __name__ == "__main__":
    main()
