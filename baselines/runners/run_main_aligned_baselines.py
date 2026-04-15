import argparse
import csv
import json
import os
import time
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
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--run_selection_only", action="store_true")
    parser.add_argument("--run_eval_only", action="store_true")
    parser.add_argument("--run_full_pipeline", action="store_true")
    return parser


def _as_list(value: Any, fallback: List[Any]) -> List[Any]:
    if value is None:
        return list(fallback)
    if isinstance(value, list):
        return list(value)
    return [value]


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

    methods = args.methods if args.methods else _as_list(cfg.get("methods"), list_methods())
    budgets = args.budgets if args.budgets else _as_list(cfg.get("budgets"), [100, 200, 500])
    seeds = args.seeds if args.seeds else _as_list(cfg.get("default_seeds"), [cfg.get("default_seed", 0)])
    device = args.device if args.device else str(cfg.get("default_device", "cpu"))
    output_root = args.output_root if args.output_root else str(cfg.get("output_root", "artifacts/baselines"))
    log_root = str(cfg.get("log_root", os.path.join(output_root, "logs")))
    ensure_dir(output_root)
    ensure_dir(log_root)
    eval_output_root = str(cfg.get("eval_output_root", output_root))
    ensure_dir(eval_output_root)

    run_records: List[Dict[str, Any]] = []
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
    iterator = tqdm(jobs, desc="main-aligned", unit="job", dynamic_ncols=True) if use_progress else jobs

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
        record: Dict[str, Any] = {}
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
            print(
                f"[main-aligned-select] method={record['method']} budget={record['budget']} "
                f"seed={seed} -> {record['paths']['selected_indices']}"
            )
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
            print(
                f"[main-aligned-eval] method={method_name} budget={budget} seed={seed} "
                f"MeanRecall={downstream_metrics['MeanRecall']:.2f}"
            )

        run_records.append(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": method_name,
                "budget": int(budget),
                "ratio": record.get("ratio"),
                "subset_size": record.get("subset_size"),
                "total_train_size": record.get("total_train_size"),
                "seed": seed,
                "dataset_name": str(cfg.get("dataset_name", "flickr")),
                "image_encoder": str(cfg.get("image_encoder", "nfnet")),
                "text_encoder": str(cfg.get("text_encoder", "bert")),
                "feature_source": str(cfg.get("feature_source", "artifacts/feature_cache")),
                "sample_unit": "pair_level_sample_idx",
                "output_dir": baseline_run_dir,
                "selected_indices_path": os.path.join(baseline_run_dir, "selected_indices.json"),
                "baseline_summary_path": os.path.join(baseline_run_dir, "baseline_summary.json"),
                "downstream_metrics_path": (
                    os.path.join(baseline_run_dir, "downstream_metrics.json")
                    if os.path.exists(os.path.join(baseline_run_dir, "downstream_metrics.json"))
                    else None
                ),
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
    print(f"  main_table_aligned_csv: {exported['main_table_csv']}")
    print(f"  main_table_aligned_agg_csv: {exported['main_table_agg_csv']}")
    print(f"  main_table_wide_csv: {exported['main_table_wide_csv']}")


if __name__ == "__main__":
    main()
