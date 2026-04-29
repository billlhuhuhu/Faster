import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BENCHMARK_ORDER = ["GQA", "ScienceQA", "MMBench", "TextVQA", "POPE"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_benchmark_name(name: str) -> Optional[str]:
    lower = str(name).lower()
    if "gqa" in lower:
        return "GQA"
    if "science" in lower:
        return "ScienceQA"
    if "mmbench" in lower:
        return "MMBench"
    if "textvqa" in lower:
        return "TextVQA"
    if "pope" in lower:
        return "POPE"
    return None


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


def as_percent(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def normalize_ratio(value: Any, run_dir: Path) -> Optional[int]:
    raw = str(value if value not in {None, ""} else "").strip()
    if raw:
        numeric = to_float(raw)
        if numeric is not None:
            if 0.0 < numeric < 1.0:
                return int(round(numeric * 100))
            return int(round(numeric))
    text = str(run_dir).lower()
    for ratio in (1, 5, 10):
        if f"_{ratio}" in text or f"ratio_{ratio:02d}" in text:
            return ratio
    return None


def normalize_mode(value: Any, run_dir: Path) -> str:
    raw = str(value if value not in {None, ""} else "").strip().lower()
    if raw in {"ours", "random"}:
        return raw
    text = str(run_dir).lower()
    if "random" in text:
        return "random"
    if "ours" in text:
        return "ours"
    return raw


def iter_dicts(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from iter_dicts(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from iter_dicts(value)


def pick_numeric_metric_from_row(row: Dict[str, Any]) -> Optional[float]:
    preferred = [
        "primary_metric_value",
        "Overall",
        "overall",
        "Accuracy",
        "accuracy",
        "Acc",
        "acc",
        "Score",
        "score",
        "F1",
        "f1",
    ]
    for key in preferred:
        if key in row:
            value = as_percent(to_float(row.get(key)))
            if value is not None:
                return value
    numeric = []
    for key, value in row.items():
        if key is None:
            continue
        name = str(key).lower()
        if any(skip in name for skip in ["fail", "index", "idx", "id", "rank"]):
            continue
        parsed = as_percent(to_float(value))
        if parsed is not None:
            numeric.append(parsed)
    if not numeric:
        return None
    return float(numeric[0])


def looks_like_aggregate_file(path: Path) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    aggregate_tokens = ["summary", "score", "acc", "result", "eval", "rating", "overall"]
    raw_tokens = ["prediction", "infer", "detail", "answer", "submission", "tmp", "cache"]
    if any(token in name for token in raw_tokens):
        return False
    if any(token in name for token in aggregate_tokens):
        return True
    # VLMEvalKit sometimes writes benchmark-level Excel files directly as <benchmark>.xlsx.
    return path.suffix.lower() in {".xlsx", ".xls"} and normalize_benchmark_name(stem) is not None


def collect_from_summary_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    path_benchmark = normalize_benchmark_name(str(path))
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                benchmark = row.get("benchmark") or row.get("dataset") or row.get("data") or row.get("name")
                canonical = normalize_benchmark_name(str(benchmark)) or path_benchmark
                if "primary_metric_value" in row:
                    value = as_percent(to_float(row.get("primary_metric_value")))
                else:
                    value = pick_numeric_metric_from_row(row)
                if canonical and value is not None:
                    out[canonical] = value
    except Exception:
        return {}
    return out


def collect_from_xlsx(path: Path) -> Dict[str, float]:
    try:
        import pandas as pd
    except ImportError:
        return {}
    canonical = normalize_benchmark_name(str(path))
    if canonical is None:
        return {}
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception:
        return {}
    for _, frame in sheets.items():
        if frame.empty:
            continue
        columns_lower = {str(column).lower(): column for column in frame.columns}
        for preferred in ["overall", "accuracy", "acc", "score"]:
            if preferred in columns_lower:
                values = frame[columns_lower[preferred]].dropna()
                values = [as_percent(to_float(value)) for value in values]
                values = [value for value in values if value is not None]
                if values:
                    # For aggregate sheets this is usually a single value. For item-level
                    # sheets, averaging avoids accidentally taking the first sample's score.
                    return {canonical: float(sum(values) / len(values))}
        rows = frame.to_dict(orient="records")
        for row in rows:
            row_text = " ".join(str(value).lower() for value in row.values())
            if not any(token in row_text for token in ["overall", "total", "avg", "average", "accuracy", "score"]):
                continue
            value = pick_numeric_metric_from_row(row)
            if value is not None:
                return {canonical: value}
        numeric = frame.select_dtypes(include="number")
        if not numeric.empty:
            # Prefer an Overall-like column when present, otherwise use the first numeric value.
            for column in numeric.columns:
                if "overall" in str(column).lower():
                    values = numeric[column].dropna()
                    if len(values):
                        return {canonical: as_percent(float(values.iloc[0]))}
            values = numeric.stack().dropna()
            if len(values):
                parsed_values = [as_percent(to_float(value)) for value in values.tolist()]
                parsed_values = [value for value in parsed_values if value is not None]
                if parsed_values:
                    return {canonical: float(sum(parsed_values) / len(parsed_values))}
    return {}


def collect_from_json(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        payload = load_json(path)
    except Exception:
        return out
    for item in iter_dicts(payload):
        benchmark = item.get("benchmark") or item.get("dataset") or item.get("name")
        canonical = normalize_benchmark_name(str(benchmark))
        if canonical is None:
            continue
        value = (
            item.get("primary_metric_value")
            or item.get("score")
            or item.get("accuracy")
            or item.get("acc")
            or item.get("Overall")
            or item.get("overall")
        )
        parsed = as_percent(to_float(value))
        if parsed is not None:
            out[canonical] = parsed
    return out


def collect_benchmark_scores(output_dir: Path, debug: bool = False) -> Dict[str, float]:
    if not output_dir.exists():
        return {}
    scores: Dict[str, float] = {}
    candidate_files = sorted(
        [
            path for path in output_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".csv", ".json", ".xlsx", ".xls"}
            and looks_like_aggregate_file(path)
        ],
        key=lambda path: (0 if "summary" in path.name.lower() else 1, len(str(path))),
    )
    for path in candidate_files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            local = collect_from_summary_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            local = collect_from_xlsx(path)
        else:
            local = collect_from_json(path)
        if debug and local:
            print(f"[source] {path}: {local}")
        scores.update(local)
    return scores


def collect_global_summary_scores(plan_root: Path) -> Dict[tuple, Dict[str, float]]:
    """Read wrapper-level summaries when per-run VLMEvalKit files are not easy to locate."""
    out: Dict[tuple, Dict[str, float]] = {}
    for path in sorted((plan_root / "reports").glob("*summary*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            continue
        for row in rows:
            mode = normalize_mode(row.get("subset_mode", ""), Path(row.get("run_dir", "")))
            ratio = normalize_ratio(row.get("subset_ratio", ""), Path(row.get("run_dir", "")))
            seed = str(row.get("seed", ""))
            benchmark = normalize_benchmark_name(str(row.get("benchmark") or row.get("dataset") or row.get("name") or ""))
            value = as_percent(
                to_float(
                    row.get("primary_metric_value")
                    or row.get("score")
                    or row.get("accuracy")
                    or row.get("acc")
                    or row.get("Overall")
                    or row.get("overall")
                )
            )
            if mode and ratio is not None and benchmark and value is not None:
                out.setdefault((mode, int(ratio), seed), {})[benchmark] = value
    return out


def resolve_output_dirs(plan: Dict[str, Any], plan_path: Path, plan_root: Path) -> List[Path]:
    raw_dirs = [
        plan.get("vlmevalkit_output_dir"),
        plan.get("output_dir"),
        plan_path.parent / "vlmevalkit_outputs",
        plan_path.parent,
        plan_root / "reports",
    ]
    out = []
    for item in raw_dirs:
        if not item:
            continue
        path = Path(item)
        if path.exists() and path not in out:
            out.append(path)
    return out


def mean_available(row: Dict[str, Any]) -> str:
    values = [to_float(row.get(name)) for name in BENCHMARK_ORDER]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def format_value(value: Any) -> str:
    parsed = to_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.4f}"


def write_markdown(rows: List[Dict[str, Any]], output_md: Path) -> None:
    headers = ["subset_mode", "subset_ratio", "seed", "num_selected_records"] + BENCHMARK_ORDER + ["mean_score"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [str(row.get(key, "")) for key in headers]
        lines.append("| " + " | ".join(values) + " |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final VLM benchmark table for ours/random 1/5/10 runs.")
    parser.add_argument("--plan_root", type=str, default="artifacts/vlm_finetune/qwen2vl_llava_subset")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--ratios", type=str, default="1,5,10")
    parser.add_argument("--modes", type=str, default="ours,random")
    parser.add_argument("--debug_sources", action="store_true")
    args = parser.parse_args()

    plan_root = Path(args.plan_root)
    output_csv = Path(args.output_csv) if args.output_csv else plan_root / "reports" / "vlm_final_ours_random_1_5_10.csv"
    output_md = Path(args.output_md) if args.output_md else plan_root / "reports" / "vlm_final_ours_random_1_5_10.md"
    keep_ratios = {int(item.strip()) for item in args.ratios.replace(",", " ").split() if item.strip()}
    keep_modes = {item.strip().lower() for item in args.modes.replace(",", " ").split() if item.strip()}
    global_summary_scores = collect_global_summary_scores(plan_root)

    rows: List[Dict[str, Any]] = []
    for plan_path in sorted(plan_root.rglob("benchmark_eval_plan.json")):
        plan = load_json(plan_path)
        subset_info_path = plan_path.parent / "subset_info.json"
        train_metrics_path = plan_path.parent / "metrics.json"
        subset_info = load_json(subset_info_path) if subset_info_path.exists() else {}
        train_metrics = load_json(train_metrics_path) if train_metrics_path.exists() else {}

        mode = normalize_mode(plan.get("subset_mode", subset_info.get("subset_mode", "")), plan_path.parent)
        ratio = normalize_ratio(plan.get("subset_ratio", subset_info.get("subset_ratio", "")), plan_path.parent)
        if mode not in keep_modes or ratio not in keep_ratios:
            continue

        scores: Dict[str, float] = {}
        output_dirs = resolve_output_dirs(plan, plan_path, plan_root)
        for output_dir in output_dirs:
            scores.update(collect_benchmark_scores(output_dir, debug=bool(args.debug_sources)))
        scores.update(global_summary_scores.get((mode, int(ratio), str(plan.get("seed", subset_info.get("seed", "")))), {}))
        row: Dict[str, Any] = {
            "subset_mode": mode,
            "subset_ratio": ratio,
            "seed": plan.get("seed", subset_info.get("seed", "")),
            "num_selected_records": subset_info.get("num_selected_records", train_metrics.get("num_selected_records", "")),
            "train_loss": format_value(train_metrics.get("train_loss", "")),
            "run_dir": str(plan_path.parent),
            "vlmevalkit_output_dir": ";".join(str(path) for path in output_dirs),
        }
        for benchmark in BENCHMARK_ORDER:
            row[benchmark] = format_value(scores.get(benchmark, ""))
        row["mean_score"] = mean_available(row)
        rows.append(row)

    mode_order = {"ours": 0, "random": 1}
    rows.sort(key=lambda row: (int(row["subset_ratio"]), mode_order.get(str(row["subset_mode"]), 99), str(row["seed"])))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subset_mode",
        "subset_ratio",
        "seed",
        "num_selected_records",
        *BENCHMARK_ORDER,
        "mean_score",
        "train_loss",
        "run_dir",
        "vlmevalkit_output_dir",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, output_md)
    print(f"saved csv: {output_csv}")
    print(f"saved markdown: {output_md}")
    print(f"collected rows: {len(rows)}")


if __name__ == "__main__":
    main()
