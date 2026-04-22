import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_numeric(prefix: str, payload: Any, out: Dict[str, float]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            flatten_numeric(next_prefix, value, out)
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            next_prefix = f"{prefix}_{idx}" if prefix else str(idx)
            flatten_numeric(next_prefix, value, out)
    else:
        try:
            if isinstance(payload, bool):
                return
            out[prefix] = float(payload)
        except (TypeError, ValueError):
            return


def read_csv_numeric(path: Path) -> Dict[str, float]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    out: Dict[str, float] = {}
    for row_idx, row in enumerate(rows):
        for key, value in row.items():
            if key is None:
                continue
            name = key.strip()
            if not name:
                continue
            try:
                out[f"{path.stem}_{row_idx}_{name}"] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def read_excel_numeric(path: Path) -> Dict[str, float]:
    try:
        import pandas as pd
    except ImportError:
        return {}
    out: Dict[str, float] = {}
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception:
        return out
    for sheet_name, frame in sheets.items():
        numeric = frame.select_dtypes(include="number")
        for column in numeric.columns:
            values = numeric[column].dropna()
            if len(values) == 0:
                continue
            out[f"{path.stem}_{sheet_name}_{column}_mean"] = float(values.mean())
            if len(values) == 1:
                out[f"{path.stem}_{sheet_name}_{column}"] = float(values.iloc[0])
    return out


def collect_numeric_results(output_dir: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not output_dir.exists():
        return metrics
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                local: Dict[str, float] = {}
                flatten_numeric(path.stem, load_json(path), local)
                metrics.update(local)
            except Exception:
                continue
        elif suffix == ".jsonl":
            try:
                for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    local = {}
                    flatten_numeric(f"{path.stem}_{idx}", json.loads(line), local)
                    metrics.update(local)
            except Exception:
                continue
        elif suffix == ".csv":
            try:
                metrics.update(read_csv_numeric(path))
            except Exception:
                continue
        elif suffix in {".xlsx", ".xls"}:
            metrics.update(read_excel_numeric(path))
    return metrics


def mean_selected_metrics(row: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    values = []
    for key in keys:
        try:
            value = row.get(key)
            if value not in {"", None}:
                values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect VLMEvalKit/lmms-eval numeric outputs for Qwen2-VL subset runs.")
    parser.add_argument("--plan_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    plan_root = Path(args.plan_root)
    rows: List[Dict[str, Any]] = []
    for plan_path in sorted(plan_root.rglob("benchmark_eval_plan.json")):
        plan = load_json(plan_path)
        subset_info_path = plan_path.parent / "subset_info.json"
        train_metrics_path = plan_path.parent / "metrics.json"
        subset_info = load_json(subset_info_path) if subset_info_path.exists() else {}
        train_metrics = load_json(train_metrics_path) if train_metrics_path.exists() else {}
        result_metrics = collect_numeric_results(Path(plan.get("vlmevalkit_output_dir", "")))
        row: Dict[str, Any] = {
            "run_dir": str(plan_path.parent),
            "subset_mode": plan.get("subset_mode", subset_info.get("subset_mode", "")),
            "subset_ratio": plan.get("subset_ratio", subset_info.get("subset_ratio", "")),
            "seed": plan.get("seed", subset_info.get("seed", "")),
            "num_selected_records": subset_info.get("num_selected_records", train_metrics.get("num_selected_records", "")),
            "base_model_path": plan.get("base_model_path", ""),
            "adapter_path": plan.get("adapter_path", ""),
            "merged_model_path": plan.get("merged_model_path", ""),
            "recommended_model_path": plan.get("recommended_model_path", ""),
            "train_loss": train_metrics.get("train_loss", ""),
            "final_eval_loss": train_metrics.get("final_eval_loss", ""),
        }
        row.update(result_metrics)
        numeric_keys = [
            key for key in result_metrics
            if any(token in key.lower() for token in ["acc", "accuracy", "score", "f1", "em"])
        ]
        mean_score = mean_selected_metrics(row, numeric_keys)
        row["mean_benchmark_score"] = "" if mean_score is None else mean_score
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "run_dir",
        "subset_mode",
        "subset_ratio",
        "seed",
        "num_selected_records",
        "train_loss",
        "final_eval_loss",
        "mean_benchmark_score",
        "recommended_model_path",
        "adapter_path",
        "merged_model_path",
    ]
    fields = preferred + [key for key in fields if key not in preferred]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved csv: {output_csv}")
    print(f"saved json: {output_json}")
    print(f"collected plans: {len(rows)}")


if __name__ == "__main__":
    main()
