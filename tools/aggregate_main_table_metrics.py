import argparse
import csv
import json
import statistics
import time
from pathlib import Path


RAW_FIELDS = [
    "dataset",
    "backbone",
    "method",
    "budget_tag",
    "budget_size",
    "seed",
    "i2t_r1",
    "i2t_r5",
    "i2t_r10",
    "t2i_r1",
    "t2i_r5",
    "t2i_r10",
    "mean_recall",
    "metrics_path",
]

SUMMARY_FIELDS = [
    "dataset",
    "backbone",
    "method",
    "budget_tag",
    "budget_size",
    "num_runs",
    "i2t_r1_mean",
    "i2t_r1_std",
    "i2t_r5_mean",
    "i2t_r5_std",
    "i2t_r10_mean",
    "i2t_r10_std",
    "t2i_r1_mean",
    "t2i_r1_std",
    "t2i_r5_mean",
    "t2i_r5_std",
    "t2i_r10_mean",
    "t2i_r10_std",
    "mean_recall_mean",
    "mean_recall_std",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate main-table metrics.json files into CSV.")
    parser.add_argument("--subset_train_root", type=str, default="artifacts/subset_train")
    parser.add_argument("--output_root", type=str, default="artifacts/reports")
    parser.add_argument("--report_name", type=str, default="main_table_abs")
    parser.add_argument("--datasets", type=str, nargs="*", default=["flickr", "coco"])
    parser.add_argument("--backbone", type=str, default="nfnet")
    parser.add_argument("--methods", type=str, nargs="*", default=["ours_baseline", "ours_full"])
    parser.add_argument("--budget_sizes", type=int, nargs="*", default=[100, 200, 500])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    return parser.parse_args()


def format_budget_tag(budget_size):
    return f"size_{int(budget_size):04d}"


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def read_metrics(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_rows(args):
    subset_train_root = Path(args.subset_train_root)
    rows = []
    missing = []

    for dataset in args.datasets:
        for method in args.methods:
            for budget_size in args.budget_sizes:
                budget_tag = format_budget_tag(budget_size)
                for seed in args.seeds:
                    metrics_path = (
                        subset_train_root
                        / dataset
                        / f"{args.backbone}_bert"
                        / budget_tag
                        / method
                        / f"seed_{int(seed)}"
                        / "metrics.json"
                    )
                    if not metrics_path.exists():
                        missing.append(str(metrics_path))
                        continue

                    payload = read_metrics(metrics_path)
                    rows.append(
                        {
                            "dataset": dataset,
                            "backbone": args.backbone,
                            "method": method,
                            "budget_tag": budget_tag,
                            "budget_size": int(payload.get("subset_size", budget_size)),
                            "seed": int(payload.get("seed", seed)),
                            "i2t_r1": float(payload["i2t_r1"]),
                            "i2t_r5": float(payload["i2t_r5"]),
                            "i2t_r10": float(payload["i2t_r10"]),
                            "t2i_r1": float(payload["t2i_r1"]),
                            "t2i_r5": float(payload["t2i_r5"]),
                            "t2i_r10": float(payload["t2i_r10"]),
                            "mean_recall": float(payload["mean_recall"]),
                            "metrics_path": str(metrics_path),
                        }
                    )
    return rows, missing


def build_summary_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["backbone"], row["method"], row["budget_tag"], row["budget_size"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    metric_keys = ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall"]
    for key in sorted(grouped.keys()):
        dataset, backbone, method, budget_tag, budget_size = key
        group_rows = grouped[key]
        summary = {
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "budget_tag": budget_tag,
            "budget_size": int(budget_size),
            "num_runs": int(len(group_rows)),
        }
        for metric in metric_keys:
            values = [float(item[metric]) for item in group_rows]
            summary[f"{metric}_mean"] = float(sum(values) / max(len(values), 1))
            summary[f"{metric}_std"] = safe_std(values)
        summary_rows.append(summary)
    return summary_rows


def write_csv(path, rows, fieldnames, encoding="utf-8", excel_sep_hint=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as handle:
        if excel_sep_hint:
            handle.write("sep=,\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.output_root) / f"{args.report_name}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = collect_rows(args)
    summary_rows = build_summary_rows(rows)

    raw_csv = report_dir / "main_table_raw.csv"
    summary_csv = report_dir / "main_table_summary.csv"
    raw_excel_csv = report_dir / "main_table_raw_excel.csv"
    summary_excel_csv = report_dir / "main_table_summary_excel.csv"
    missing_txt = report_dir / "missing_metrics.txt"

    write_csv(raw_csv, rows, RAW_FIELDS)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    write_csv(raw_excel_csv, rows, RAW_FIELDS, encoding="utf-8-sig", excel_sep_hint=True)
    write_csv(summary_excel_csv, summary_rows, SUMMARY_FIELDS, encoding="utf-8-sig", excel_sep_hint=True)
    with open(missing_txt, "w", encoding="utf-8") as handle:
        for item in missing:
            handle.write(item + "\n")

    print("Main-table aggregation finished:")
    print(f"  report_dir: {report_dir}")
    print(f"  raw_csv: {raw_csv}")
    print(f"  summary_csv: {summary_csv}")
    print(f"  raw_excel_csv: {raw_excel_csv}")
    print(f"  summary_excel_csv: {summary_excel_csv}")
    print(f"  missing_metrics: {missing_txt}")
    print(f"  collected_runs: {len(rows)}")
    print(f"  grouped_entries: {len(summary_rows)}")


if __name__ == "__main__":
    main()
