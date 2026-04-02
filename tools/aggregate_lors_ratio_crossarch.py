import argparse
import csv
import json
import re
from pathlib import Path


TABLE_LINE_PATTERN = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def parse_eval_metrics(evaluate_log_path):
    lines = Path(evaluate_log_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    fallback_metrics = None
    for line in reversed(lines):
        if "|" not in line:
            continue
        numbers = TABLE_LINE_PATTERN.findall(line)
        if len(numbers) < 7:
            continue
        values = [float(x) for x in numbers[-7:]]
        metrics = {
            "img_r1": values[0],
            "img_r5": values[1],
            "img_r10": values[2],
            "txt_r1": values[3],
            "txt_r5": values[4],
            "txt_r10": values[5],
            "r_mean": values[6],
        }
        if fallback_metrics is None:
            fallback_metrics = metrics
        if any(value != 0.0 for value in values):
            return metrics
    return fallback_metrics


def main():
    parser = argparse.ArgumentParser(description="Aggregate LoRS ratio cross-architecture evaluation logs into a CSV table.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []

    for item in payload.get("runs", []):
        evaluate_log_path = item.get("evaluate_log")
        metrics = parse_eval_metrics(evaluate_log_path) if evaluate_log_path and Path(evaluate_log_path).exists() else None
        row = {
            "dataset": payload.get("dataset"),
            "ratio": item.get("ratio"),
            "budget_size": item.get("budget_size"),
            "distill_backbone": item.get("distill_backbone"),
            "eval_backbone": item.get("eval_backbone"),
            "run_name": item.get("run_name"),
            "baseline_log_dir": item.get("baseline_log_dir"),
            "checkpoint_path": item.get("checkpoint_path"),
            "evaluate_log": evaluate_log_path,
            "img_r1": None,
            "img_r5": None,
            "img_r10": None,
            "txt_r1": None,
            "txt_r5": None,
            "txt_r10": None,
            "r_mean": None,
        }
        if metrics is not None:
            row.update(metrics)
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "dataset", "ratio", "budget_size", "distill_backbone", "eval_backbone", "run_name",
            "baseline_log_dir", "checkpoint_path", "evaluate_log",
            "img_r1", "img_r5", "img_r10", "txt_r1", "txt_r5", "txt_r10", "r_mean",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {output_csv}")


if __name__ == "__main__":
    main()
