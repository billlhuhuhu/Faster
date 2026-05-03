import argparse
import csv
import json
import statistics
from pathlib import Path


METRIC_KEYS = ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall"]


RAW_FIELDS = [
    "dataset",
    "model_tag",
    "scale_group",
    "scale_type",
    "scales",
    "variant",
    "source_train_root",
    "budget_type",
    "budget_tag",
    "budget_value",
    "seed",
    *METRIC_KEYS,
    "metrics_path",
]


SUMMARY_FIELDS = [
    "dataset",
    "model_tag",
    "scale_group",
    "scale_type",
    "scales",
    "variant",
    "source_train_root",
    "budget_type",
    "budget_tag",
    "budget_value",
    "num_runs",
    *[f"{metric}_{suffix}" for metric in METRIC_KEYS for suffix in ("mean", "std")],
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate wavelet scale ablation retrieval metrics.")
    parser.add_argument("--subset_train_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="flickr")
    parser.add_argument("--model_tag", default="nfnet_bert")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config entries: label|type|scales|variant or label|type|scales|variant|source_train_root",
    )
    parser.add_argument("--budgets", nargs="*", default=["100", "200", "500"])
    parser.add_argument("--ratios", nargs="*", default=["0.01", "0.02", "0.03"])
    parser.add_argument("--seeds", nargs="*", default=["0"])
    return parser.parse_args()


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def ratio_tag(ratio):
    return f"ratio_{int(round(float(ratio) * 100)):02d}"


def parse_config(raw):
    parts = raw.split("|")
    if len(parts) not in {4, 5}:
        raise ValueError(f"Invalid config entry {raw!r}; expected label|type|scales|variant[|source_train_root]")
    label, scale_type, scales, variant = parts[:4]
    source_train_root = parts[4] if len(parts) == 5 and parts[4] else None
    return {
        "scale_group": label,
        "scale_type": scale_type,
        "scales": scales,
        "variant": variant,
        "source_train_root": source_train_root,
    }


def load_metric_row(path, cfg, dataset, model_tag, budget_type, budget_tag, budget_value, seed):
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "dataset": dataset,
        "model_tag": model_tag,
        **cfg,
        "budget_type": budget_type,
        "budget_tag": budget_tag,
        "budget_value": budget_value,
        "seed": int(seed),
        "metrics_path": str(path),
    }
    for metric in METRIC_KEYS:
        row[metric] = float(payload[metric])
    return row


def collect_rows(args, configs):
    default_root = Path(args.subset_train_root)
    rows = []
    missing = []
    targets = []
    for budget in args.budgets:
        if str(budget).strip():
            budget_int = int(float(budget))
            targets.append(("abs", f"size_{budget_int:04d}", str(budget_int)))
    for ratio in args.ratios:
        if str(ratio).strip():
            targets.append(("ratio", ratio_tag(ratio), f"{float(ratio):.6f}"))

    for cfg in configs:
        root = Path(cfg["source_train_root"]) if cfg.get("source_train_root") else default_root
        cfg_for_row = dict(cfg)
        cfg_for_row["source_train_root"] = str(root)
        for budget_type, budget_tag, budget_value in targets:
            for seed in args.seeds:
                metrics_path = (
                    root
                    / args.dataset
                    / args.model_tag
                    / budget_tag
                    / cfg["variant"]
                    / f"seed_{int(seed)}"
                    / "metrics.json"
                )
                if not metrics_path.exists():
                    missing.append(str(metrics_path))
                    continue
                rows.append(
                    load_metric_row(
                        metrics_path,
                        cfg_for_row,
                        args.dataset,
                        args.model_tag,
                        budget_type,
                        budget_tag,
                        budget_value,
                        seed,
                    )
                )
    return rows, missing


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (
            row["dataset"],
            row["model_tag"],
            row["scale_group"],
            row["scale_type"],
            row["scales"],
            row["variant"],
            row["source_train_root"],
            row["budget_type"],
            row["budget_tag"],
            row["budget_value"],
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key in sorted(grouped):
        (
            dataset,
            model_tag,
            scale_group,
            scale_type,
            scales,
            variant,
            source_train_root,
            budget_type,
            budget_tag,
            budget_value,
        ) = key
        group_rows = grouped[key]
        summary = {
            "dataset": dataset,
            "model_tag": model_tag,
            "scale_group": scale_group,
            "scale_type": scale_type,
            "scales": scales,
            "variant": variant,
            "source_train_root": source_train_root,
            "budget_type": budget_type,
            "budget_tag": budget_tag,
            "budget_value": budget_value,
            "num_runs": len(group_rows),
        }
        for metric in METRIC_KEYS:
            values = [float(row[metric]) for row in group_rows]
            summary[f"{metric}_mean"] = float(sum(values) / len(values))
            summary[f"{metric}_std"] = safe_std(values)
        summaries.append(summary)
    return summaries


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    fields = [
        "scale_group",
        "scale_type",
        "scales",
        "budget_tag",
        "budget_value",
        "num_runs",
        "i2t_r1_mean",
        "i2t_r5_mean",
        "i2t_r10_mean",
        "t2i_r1_mean",
        "t2i_r5_mean",
        "t2i_r10_mean",
        "mean_recall_mean",
        "mean_recall_std",
    ]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    configs = [parse_config(raw) for raw in args.configs]
    rows, missing = collect_rows(args, configs)
    summary_rows = summarize(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "wavelet_scale_ablation_raw.csv"
    summary_csv = output_dir / "wavelet_scale_ablation_summary.csv"
    summary_md = output_dir / "wavelet_scale_ablation_summary.md"
    missing_txt = output_dir / "missing_metrics.txt"

    write_csv(raw_csv, rows, RAW_FIELDS)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    write_markdown(summary_md, summary_rows)
    missing_txt.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    print(f"saved raw csv: {raw_csv}")
    print(f"saved summary csv: {summary_csv}")
    print(f"saved summary md: {summary_md}")
    print(f"saved missing list: {missing_txt}")
    print(f"collected runs: {len(rows)}")
    print(f"grouped entries: {len(summary_rows)}")


if __name__ == "__main__":
    main()
