import argparse
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from baselines.common.io import load_json, save_json
from baselines.common.subset_reconstruction import build_subset_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a baseline-selected subset via mainline retrieval train/eval.")
    parser.add_argument("--baseline_result_dir", type=str, required=True)
    parser.add_argument("--selected_indices_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None, choices=["flickr", "coco"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image_encoder", type=str, default=None)
    parser.add_argument("--text_encoder", type=str, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--feature_source", type=str, default="artifacts/feature_cache")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--ann_root", type=str, default=None)
    parser.add_argument("--train_entry", type=str, default="run_subset_train.py")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size_train", type=int, default=64)
    parser.add_argument("--batch_size_test", type=int, default=128)
    parser.add_argument("--text_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--subset_tag", type=str, default=None)
    parser.add_argument("--subset_restore_mode", type=str, default="pair_level_indices")
    parser.add_argument("--no_aug", action="store_true", default=True)
    return parser


def _infer_defaults_from_summary(baseline_result_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(baseline_result_dir, "baseline_summary.json")
    summary = load_json(summary_path) if os.path.exists(summary_path) else {}
    return summary


def _default_paths_for_dataset(dataset_name: str) -> Dict[str, str]:
    if dataset_name == "flickr":
        return {"image_root": "data/flickr30k", "ann_root": "data/Flickr30k_ann"}
    if dataset_name == "coco":
        return {"image_root": "data/coco", "ann_root": "data/COCO"}
    raise ValueError(f"Unsupported dataset_name={dataset_name}")


def _extract_metrics_path(stdout_text: str) -> Optional[str]:
    m = re.search(r"metrics_path:\s*(.+)", stdout_text)
    if m:
        return m.group(1).strip()
    return None


def _extract_output_dir(stdout_text: str) -> Optional[str]:
    m = re.search(r"output_dir:\s*(.+)", stdout_text)
    if m:
        return m.group(1).strip()
    return None


def run_downstream_eval(
    *,
    baseline_result_dir: str,
    selected_indices_path: Optional[str],
    dataset_name: Optional[str],
    split: str,
    image_encoder: Optional[str],
    text_encoder: Optional[str],
    budget: Optional[int],
    seed: Optional[int],
    device: str,
    feature_source: str,
    image_root: Optional[str],
    ann_root: Optional[str],
    train_entry: str,
    output_dir: Optional[str],
    epochs: int,
    batch_size_train: int,
    batch_size_test: int,
    text_batch_size: int,
    num_workers: int,
    eval_interval: int,
    subset_tag: Optional[str],
    subset_restore_mode: str,
    no_aug: bool = True,
) -> Dict[str, Any]:
    summary = _infer_defaults_from_summary(baseline_result_dir)
    dataset_name = dataset_name or summary.get("dataset_name") or summary.get("dataset")
    image_encoder = image_encoder or summary.get("image_encoder")
    text_encoder = text_encoder or summary.get("text_encoder", "bert")
    seed = int(seed if seed is not None else summary.get("seed", 0))
    budget = int(budget if budget is not None else summary.get("budget", 0))
    if not dataset_name or not image_encoder or not text_encoder:
        raise ValueError("dataset_name/image_encoder/text_encoder are required and could not be inferred.")

    if selected_indices_path is None:
        selected_indices_path = os.path.join(baseline_result_dir, "selected_indices.json")
    if not os.path.exists(selected_indices_path):
        raise FileNotFoundError(f"selected_indices.json not found: {selected_indices_path}")

    defaults = _default_paths_for_dataset(str(dataset_name))
    image_root = image_root or defaults["image_root"]
    ann_root = ann_root or defaults["ann_root"]
    output_dir = output_dir or baseline_result_dir
    subset_tag = subset_tag or str(summary.get("method") or os.path.basename(baseline_result_dir))

    subset_spec = build_subset_spec(
        baseline_result_dir=baseline_result_dir,
        selected_indices_path=selected_indices_path,
        feature_cache_root=feature_source,
        dataset_name=str(dataset_name),
        split=str(split),
        image_encoder=str(image_encoder),
        text_encoder=str(text_encoder),
        output_path=os.path.join(baseline_result_dir, "subset_spec.json"),
    )

    cmd = [
        sys.executable,
        train_entry,
        "--dataset",
        str(dataset_name),
        "--image_root",
        str(image_root),
        "--ann_root",
        str(ann_root),
        "--selected_indices_path",
        str(selected_indices_path),
        "--subset_size",
        str(subset_spec["subset_size"]),
        "--subset_tag",
        str(subset_tag),
        "--image_encoder",
        str(image_encoder),
        "--text_encoder",
        str(text_encoder),
        "--output_root",
        str(output_dir),
        "--epochs",
        str(int(epochs)),
        "--batch_size_train",
        str(int(batch_size_train)),
        "--batch_size_test",
        str(int(batch_size_test)),
        "--text_batch_size",
        str(int(text_batch_size)),
        "--num_workers",
        str(int(num_workers)),
        "--eval_interval",
        str(int(eval_interval)),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
    ]
    if no_aug:
        cmd.append("--no_aug")

    started = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = float(time.time() - started)
    log_path = os.path.join(baseline_result_dir, "train_eval_log.txt")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Downstream eval failed (exit={proc.returncode}). "
            f"See log: {log_path}"
        )

    inferred_metrics_path = _extract_metrics_path(proc.stdout)
    inferred_output_dir = _extract_output_dir(proc.stdout)
    metrics_path = inferred_metrics_path
    if metrics_path is None and inferred_output_dir is not None:
        candidate = os.path.join(inferred_output_dir, "metrics.json")
        if os.path.exists(candidate):
            metrics_path = candidate
    if metrics_path is None or not os.path.exists(metrics_path):
        raise FileNotFoundError(
            "Could not locate mainline metrics.json from run_subset_train output. "
            f"log={log_path}"
        )

    raw = load_json(metrics_path)
    downstream = {
        "method": summary.get("method"),
        "dataset": dataset_name,
        "image_encoder": image_encoder,
        "text_encoder": text_encoder,
        "sample_unit": "pair_level_sample_idx",
        "subset_restore_mode": subset_restore_mode,
        "budget": budget,
        "seed": int(seed),
        "subset_size": int(subset_spec["subset_size"]),
        "I2T_R1": float(raw.get("i2t_r1")),
        "I2T_R5": float(raw.get("i2t_r5")),
        "I2T_R10": float(raw.get("i2t_r10")),
        "T2I_R1": float(raw.get("t2i_r1")),
        "T2I_R5": float(raw.get("t2i_r5")),
        "T2I_R10": float(raw.get("t2i_r10")),
        "MeanRecall": float(raw.get("mean_recall")),
        "train_time": elapsed,
        "eval_time": raw.get("eval_time"),
        "selection_time": summary.get("selection_time"),
        "mainline_metrics_path": metrics_path,
        "mainline_output_dir": inferred_output_dir,
        "train_entry": train_entry,
        "train_eval_log": log_path,
        "subset_spec_path": subset_spec["subset_spec_path"],
    }
    save_json(os.path.join(baseline_result_dir, "downstream_metrics.json"), downstream)
    return downstream


def main():
    args = build_parser().parse_args()
    out = run_downstream_eval(
        baseline_result_dir=args.baseline_result_dir,
        selected_indices_path=args.selected_indices_path,
        dataset_name=args.dataset_name,
        split=args.split,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        budget=args.budget,
        seed=args.seed,
        device=args.device,
        feature_source=args.feature_source,
        image_root=args.image_root,
        ann_root=args.ann_root,
        train_entry=args.train_entry,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        text_batch_size=args.text_batch_size,
        num_workers=args.num_workers,
        eval_interval=args.eval_interval,
        subset_tag=args.subset_tag,
        subset_restore_mode=args.subset_restore_mode,
        no_aug=bool(args.no_aug),
    )
    print("Downstream eval finished:")
    print(f"  method={out['method']} budget={out['budget']} seed={out['seed']}")
    print(f"  I2T_R1={out['I2T_R1']:.2f} T2I_R1={out['T2I_R1']:.2f} MeanRecall={out['MeanRecall']:.2f}")
    print(f"  downstream_metrics={os.path.join(args.baseline_result_dir, 'downstream_metrics.json')}")


if __name__ == "__main__":
    main()
