import argparse
import csv
import json
from pathlib import Path


VARIANTS = [
    {
        "name": "dir1_bidir_only",
        "correction_mode": "bidirectional",
        "fusion_mode": "intersection",
        "enable_lsrc": False,
    },
    {
        "name": "dir2_conf_only",
        "correction_mode": "directional",
        "fusion_mode": "confidence_aware",
        "enable_lsrc": False,
    },
    {
        "name": "dir3_lsrc_only",
        "correction_mode": "directional",
        "fusion_mode": "intersection",
        "enable_lsrc": True,
    },
    {
        "name": "all_enabled",
        "correction_mode": "bidirectional",
        "fusion_mode": "confidence_aware",
        "enable_lsrc": True,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate direction-ablation experiment metrics.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="nfnet")
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_train_root", type=str, default="artifacts/subset_train")
    parser.add_argument("--ablation_train_root", type=str, default="artifacts/subset_train_ablation")
    parser.add_argument("--report_root", type=str, default="artifacts/reports")
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 200, 500])
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_metrics_path(root, dataset, model_tag, budget, variant, seed):
    budget_tag = f"size_{int(budget):04d}"
    candidates = [
        Path(root) / dataset / model_tag / budget_tag / variant / f"seed_{int(seed)}" / "metrics.json",
        Path(root) / variant / dataset / model_tag / budget_tag / variant / f"seed_{int(seed)}" / "metrics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def main():
    args = parse_args()
    model_tag = f"{args.backbone}_{args.text_encoder}"
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []

    for budget in args.budgets:
        base_metrics_path = find_metrics_path(
            args.base_train_root,
            args.dataset,
            model_tag,
            budget,
            "ours_full",
            args.seed,
        )
        base_metrics = read_json(base_metrics_path) if base_metrics_path.exists() else None
        if base_metrics is None:
            missing.append(str(base_metrics_path))

        for variant in VARIANTS:
            metrics_path = find_metrics_path(
                args.ablation_train_root,
                args.dataset,
                model_tag,
                budget,
                variant["name"],
                args.seed,
            )
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                row = {
                    "dataset": args.dataset,
                    "budget_size": int(budget),
                    "variant": variant["name"],
                    "correction_mode": variant["correction_mode"],
                    "fusion_mode": variant["fusion_mode"],
                    "enable_lsrc": int(variant["enable_lsrc"]),
                    "i2t_r1": None,
                    "i2t_r5": None,
                    "i2t_r10": None,
                    "t2i_r1": None,
                    "t2i_r5": None,
                    "t2i_r10": None,
                    "mean_recall": None,
                    "base_mean_recall": base_metrics.get("mean_recall") if base_metrics else None,
                    "delta_mean_recall_vs_base": None,
                    "metrics_path": str(metrics_path),
                }
                rows.append(row)
                continue

            metrics = read_json(metrics_path)
            row = {
                "dataset": args.dataset,
                "budget_size": int(budget),
                "variant": variant["name"],
                "correction_mode": variant["correction_mode"],
                "fusion_mode": variant["fusion_mode"],
                "enable_lsrc": int(variant["enable_lsrc"]),
                "i2t_r1": metrics.get("i2t_r1"),
                "i2t_r5": metrics.get("i2t_r5"),
                "i2t_r10": metrics.get("i2t_r10"),
                "t2i_r1": metrics.get("t2i_r1"),
                "t2i_r5": metrics.get("t2i_r5"),
                "t2i_r10": metrics.get("t2i_r10"),
                "mean_recall": metrics.get("mean_recall"),
                "base_mean_recall": base_metrics.get("mean_recall") if base_metrics else None,
                "delta_mean_recall_vs_base": (
                    metrics.get("mean_recall") - base_metrics.get("mean_recall")
                    if base_metrics is not None and metrics.get("mean_recall") is not None
                    else None
                ),
                "metrics_path": str(metrics_path),
            }
            rows.append(row)

    csv_path = report_root / f"direction_ablation_{args.dataset}_seed{args.seed}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    missing_path = report_root / f"direction_ablation_{args.dataset}_seed{args.seed}_missing.txt"
    with open(missing_path, "w", encoding="utf-8") as handle:
        for item in missing:
            handle.write(f"{item}\n")

    print(f"saved_csv: {csv_path}")
    print(f"saved_missing: {missing_path}")


if __name__ == "__main__":
    main()
