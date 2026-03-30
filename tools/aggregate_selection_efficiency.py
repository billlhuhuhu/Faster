import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_summary_paths(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("selection_efficiency_summary.json"))


def main():
    parser = argparse.ArgumentParser(description="Aggregate selection-only efficiency summaries into a CSV table.")
    parser.add_argument("--summary_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    rows = []
    for summary_path in collect_summary_paths(args.summary_root):
        summary = load_json(summary_path)
        rows.append(
            {
                "variant": summary.get("variant_name"),
                "dataset": summary.get("dataset"),
                "budget_size": summary.get("budget_size"),
                "selection_time_s": summary.get("selection_time_s"),
                "selection_total_energy_wh": summary.get("selection_total_energy_wh"),
                "mean_recall": summary.get("mean_recall"),
                "mean_recall_per_wh": summary.get("mean_recall_per_wh"),
                "mean_recall_per_second": summary.get("mean_recall_per_second"),
                "speedup_vs_baseline": summary.get("speedup_vs_baseline"),
                "energy_reduction_vs_baseline": summary.get("energy_reduction_vs_baseline"),
                "summary_path": str(summary_path),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "dataset",
        "budget_size",
        "selection_time_s",
        "selection_total_energy_wh",
        "mean_recall",
        "mean_recall_per_wh",
        "mean_recall_per_second",
        "speedup_vs_baseline",
        "energy_reduction_vs_baseline",
        "summary_path",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {output_csv}")


if __name__ == "__main__":
    main()
