import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional


TABLE_LINE_PATTERN = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def to_float(value: Any) -> Optional[float]:
    try:
        if value in {"", None, "-"}:
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_lors_eval_log(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    fallback = None
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "|" not in line:
            continue
        numbers = TABLE_LINE_PATTERN.findall(line)
        if len(numbers) < 7:
            continue
        values = [float(item) for item in numbers[-7:]]
        metrics = {
            "t2i_r1": values[0],
            "t2i_r5": values[1],
            "t2i_r10": values[2],
            "i2t_r1": values[3],
            "i2t_r5": values[4],
            "i2t_r10": values[5],
            "mean_recall": values[6],
        }
        if fallback is None:
            fallback = metrics
        if any(value != 0.0 for value in values):
            return metrics
    return fallback or {}


def parse_metrics_json(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    payload = load_json(path)
    out = {}
    for key in ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall"]:
        value = to_float(payload.get(key))
        if value is not None:
            out[key] = value
    return out


def parse_metric_source(row: Dict[str, Any]) -> Dict[str, float]:
    metrics_path = row.get("metrics_path")
    evaluate_log = row.get("evaluate_log")
    if metrics_path:
        metrics = parse_metrics_json(Path(metrics_path))
        if metrics:
            return metrics
    if evaluate_log:
        return parse_lors_eval_log(Path(evaluate_log))
    out = {}
    for key in ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall", "r_mean"]:
        value = to_float(row.get(key))
        if value is not None:
            out["mean_recall" if key == "r_mean" else key] = value
    return out


def parse_measurement_json(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    key_map = {
        "wall_seconds": "seconds",
        "gpu_energy_Wh": "gpu_energy_Wh",
        "cpu_energy_Wh": "cpu_energy_Wh",
        "total_energy_Wh": "energy_Wh",
    }
    for src, dst in key_map.items():
        value = to_float(payload.get(src))
        if value is not None:
            out[dst] = value
    return out


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_external_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row.get("method") or row.get("subset_mode") or row.get("name")
            if not method:
                continue
            normalized = {
                "method": method,
                "dataset": row.get("dataset", ""),
                "budget_type": row.get("budget_type", ""),
                "budget_value": row.get("budget_value", row.get("budget_size", row.get("ratio", ""))),
                "budget_tag": row.get("budget_tag", ""),
                "eval_backbone": row.get("eval_backbone", row.get("backbone", "")),
                "stage": row.get("stage", "external"),
                "seconds": row.get("seconds", ""),
                "gpu_count": row.get("gpu_count", ""),
                "gpu_hours": row.get("gpu_hours", row.get("total_gpu_hours", "")),
                "source": str(path),
            }
            for key in ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall", "r_mean"]:
                if row.get(key) not in {"", None}:
                    normalized[key] = row.get(key)
            rows.append(normalized)
    return rows


def normalize_result_rows(manifest_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in manifest_rows:
        metrics = parse_metric_source(row)
        item = dict(row)
        measurement_path = item.get("measurement_path")
        if measurement_path:
            item.update(parse_measurement_json(Path(measurement_path)))
        item.update(metrics)
        if "mean_recall" not in item and "r_mean" in item:
            item["mean_recall"] = item["r_mean"]
        has_measurement = any(to_float(item.get(key)) is not None for key in ["seconds", "energy_Wh", "gpu_energy_Wh", "cpu_energy_Wh"])
        if to_float(item.get("mean_recall")) is None and not has_measurement:
            continue
        if to_float(item.get("mean_recall")) is not None:
            item["mean_recall"] = float(item["mean_recall"])
            item["test_accuracy"] = float(item["mean_recall"])
        item["seconds"] = to_float(item.get("seconds")) or 0.0
        item["gpu_count"] = int(to_float(item.get("gpu_count")) or 1)
        gpu_hours = to_float(item.get("gpu_hours"))
        if gpu_hours is None:
            gpu_hours = float(item["seconds"]) * int(item["gpu_count"]) / 3600.0
        item["gpu_hours"] = gpu_hours
        item["gpu_energy_Wh"] = to_float(item.get("gpu_energy_Wh")) or 0.0
        item["cpu_energy_Wh"] = to_float(item.get("cpu_energy_Wh")) or 0.0
        item["energy_Wh"] = to_float(item.get("energy_Wh")) or to_float(item.get("total_energy_Wh")) or (
            float(item["gpu_energy_Wh"]) + float(item["cpu_energy_Wh"])
        )
        item.setdefault("budget_tag", "")
        item.setdefault("budget_type", "")
        item.setdefault("budget_value", "")
        item.setdefault("eval_backbone", "")
        item.setdefault("method", "")
        out.append(item)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_arch_bias(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        if not row.get("eval_backbone"):
            continue
        if to_float(row.get("mean_recall")) is None:
            continue
        key = (row.get("method"), row.get("dataset"), row.get("budget_tag"), row.get("budget_type"), str(row.get("budget_value")))
        groups.setdefault(key, []).append(row)
    out = []
    for key, items in sorted(groups.items()):
        values = [float(item["mean_recall"]) for item in items]
        backbones = sorted({str(item.get("eval_backbone")) for item in items if item.get("eval_backbone")})
        row = {
            "method": key[0],
            "dataset": key[1],
            "budget_tag": key[2],
            "budget_type": key[3],
            "budget_value": key[4],
            "num_architectures": len(backbones),
            "eval_backbones": " ".join(backbones),
            "mean_recall_mean": mean(values),
            "test_accuracy_mean": mean(values),
            "arch_std": pstdev(values) if len(values) > 1 else 0.0,
            "arch_max_drop": max(values) - min(values) if values else 0.0,
        }
        for item in items:
            if item.get("eval_backbone"):
                row[f"mr_{item['eval_backbone']}"] = float(item["mean_recall"])
        out.append(row)
    return out


def build_energy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("method"), row.get("dataset"), row.get("budget_tag"), row.get("budget_type"), str(row.get("budget_value")))
        groups.setdefault(key, []).append(row)
    out = []
    for key, items in sorted(groups.items()):
        selection_items = [item for item in items if str(item.get("stage", "")).startswith(("selection", "distill"))]
        training_all = [item for item in items if str(item.get("stage", "")).startswith(("training", "train_eval", "evaluate"))]
        eval_backbones = sorted({str(item.get("eval_backbone")) for item in training_all if item.get("eval_backbone")})
        if not eval_backbones:
            eval_backbones = [""]
        for eval_backbone in eval_backbones:
            training_items = [item for item in training_all if str(item.get("eval_backbone", "")) == eval_backbone]
            row_items = selection_items + training_items
            gpu_hours = sum(float(item.get("gpu_hours") or 0.0) for item in row_items)
            seconds = sum(float(item.get("seconds") or 0.0) for item in row_items)
            selection_seconds = sum(float(item.get("seconds") or 0.0) for item in selection_items)
            training_seconds = sum(float(item.get("seconds") or 0.0) for item in training_items)
            selection_energy = sum(float(item.get("energy_Wh") or 0.0) for item in selection_items)
            training_energy = sum(float(item.get("energy_Wh") or 0.0) for item in training_items)
            perf_values = [float(item["mean_recall"]) for item in training_items if to_float(item.get("mean_recall")) is not None]
            mean_recall = mean(perf_values) if perf_values else None
            out.append(
                {
                    "method": key[0],
                    "dataset": key[1],
                    "budget_tag": key[2],
                    "budget_type": key[3],
                    "budget_value": key[4],
                    "eval_backbone": eval_backbone,
                    "selection_time_seconds": selection_seconds,
                    "training_time_seconds": training_seconds,
                    "total_time_seconds": selection_seconds + training_seconds,
                    "selection_energy_Wh": selection_energy,
                    "training_energy_Wh": training_energy,
                    "total_energy_Wh": selection_energy + training_energy,
                    "total_seconds": seconds,
                    "total_gpu_hours_proxy": gpu_hours,
                    "mean_recall": "" if mean_recall is None else mean_recall,
                    "test_accuracy": "" if mean_recall is None else mean_recall,
                    "mr_per_gpu_hour": "" if not gpu_hours or mean_recall is None else mean_recall / gpu_hours,
                    "test_accuracy_per_Wh": "" if not (selection_energy + training_energy) or mean_recall is None else mean_recall / (selection_energy + training_energy),
                    "num_result_rows": len(row_items),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supplemental architecture-bias and energy-efficiency tables.")
    parser.add_argument("--manifest_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--external_csv", action="append", default=[])
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.manifest_jsonl)))
    for csv_path in args.external_csv:
        if csv_path:
            rows.extend(read_external_csv(Path(csv_path)))
    detail = normalize_result_rows(rows)

    output_dir = Path(args.output_dir)
    detail_fields = [
        "method", "dataset", "budget_tag", "budget_type", "budget_value", "eval_backbone",
        "mean_recall", "i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10",
        "test_accuracy", "stage", "seconds", "gpu_count", "gpu_hours",
        "gpu_energy_Wh", "cpu_energy_Wh", "energy_Wh", "measurement_path",
        "metrics_path", "evaluate_log", "source",
    ]
    write_csv(output_dir / "supplemental_detail.csv", detail, detail_fields)
    write_csv(output_dir / "architecture_bias.csv", build_arch_bias(detail))
    write_csv(output_dir / "energy_efficiency.csv", build_energy(detail))
    print(f"saved detail: {output_dir / 'supplemental_detail.csv'}")
    print(f"saved architecture bias: {output_dir / 'architecture_bias.csv'}")
    print(f"saved energy efficiency: {output_dir / 'energy_efficiency.csv'}")


if __name__ == "__main__":
    main()
