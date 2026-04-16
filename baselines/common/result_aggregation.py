import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


METRIC_COLUMNS = [
    "I2T_R1",
    "I2T_R5",
    "I2T_R10",
    "T2I_R1",
    "T2I_R5",
    "T2I_R10",
    "MeanRecall",
]


_METRIC_CANDIDATES = {
    "I2T_R1": ["I2T_R1", "i2t_r1", "i2t_recall@1", "i2t_r_at_1"],
    "I2T_R5": ["I2T_R5", "i2t_r5", "i2t_recall@5", "i2t_r_at_5"],
    "I2T_R10": ["I2T_R10", "i2t_r10", "i2t_recall@10", "i2t_r_at_10"],
    "T2I_R1": ["T2I_R1", "t2i_r1", "t2i_recall@1", "t2i_r_at_1"],
    "T2I_R5": ["T2I_R5", "t2i_r5", "t2i_recall@5", "t2i_r_at_5"],
    "T2I_R10": ["T2I_R10", "t2i_r10", "t2i_recall@10", "t2i_r_at_10"],
    "MeanRecall": ["MeanRecall", "mean_recall", "mR", "mr"],
}


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _pick_metric(payloads: Iterable[Dict[str, Any]], canonical_key: str) -> Optional[float]:
    for payload in payloads:
        for key in _METRIC_CANDIDATES[canonical_key]:
            if key in payload:
                val = _to_float(payload.get(key))
                if val is not None:
                    return val
    return None


def _compute_mean_recall(metrics: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [metrics[k] for k in METRIC_COLUMNS if k != "MeanRecall" and metrics.get(k) is not None]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float32)))


def _parse_seed_from_dir(path: str) -> Optional[int]:
    m = re.search(r"seed_(\d+)", path)
    return int(m.group(1)) if m else None


def _parse_budget_from_dir(path: str) -> Optional[int]:
    m = re.search(r"budget_(\d+)", path)
    return int(m.group(1)) if m else None


def _method_config_candidates(method: str) -> List[str]:
    method_slug = str(method).replace("-", "_")
    cands = [f"{method_slug}.yaml"]
    if method == "adap_sne":
        cands.append("adap_sne.yaml")
    return cands


def scan_baseline_runs(root: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for current_root, _, files in os.walk(root):
        if "baseline_summary.json" not in files:
            continue

        summary_path = os.path.join(current_root, "baseline_summary.json")
        selected_path = os.path.join(current_root, "selected_indices.json")
        metrics_path = os.path.join(current_root, "downstream_metrics.json")
        summary = _read_json(summary_path)
        metrics_payload = _read_json(metrics_path) if os.path.exists(metrics_path) else {}

        metrics = {}
        for key in METRIC_COLUMNS:
            metrics[key] = _pick_metric([metrics_payload, summary], key)
        if metrics["MeanRecall"] is None:
            metrics["MeanRecall"] = _compute_mean_recall(metrics)

        method = str(summary.get("method", "")).strip().lower()
        budget = _to_int(summary.get("budget"))
        if budget is None:
            budget = _parse_budget_from_dir(current_root)
        seed = _to_int(summary.get("seed"))
        if seed is None:
            seed = _parse_seed_from_dir(current_root)

        selection_time = None
        # explicit time fallbacks
        for payload in (summary, metrics_payload):
            for key in ("selection_time", "selection_time_sec", "selection_seconds"):
                if key in payload:
                    selection_time = _to_float(payload.get(key))
                    break
            if selection_time is not None:
                break
        eval_time = None
        for payload in (metrics_payload, summary):
            for key in ("eval_time", "eval_time_sec", "evaluation_time", "downstream_eval_time"):
                if key in payload:
                    eval_time = _to_float(payload.get(key))
                    break
            if eval_time is not None:
                break
        train_time = None
        for payload in (metrics_payload, summary):
            for key in ("train_time", "train_time_sec", "training_time", "train_eval_time"):
                if key in payload:
                    train_time = _to_float(payload.get(key))
                    break
            if train_time is not None:
                break

        selected_count = None
        if os.path.exists(selected_path):
            try:
                selected_payload = _read_json(selected_path)
                arr = selected_payload.get("selected_indices", [])
                if isinstance(arr, list):
                    selected_count = len(arr)
            except Exception:
                selected_count = None

        record = {
            "method": method or None,
            "budget": budget,
            "seed": seed,
            "dataset": summary.get("dataset_name"),
            "image_encoder": summary.get("image_encoder"),
            "text_encoder": summary.get("text_encoder"),
            "sample_unit": summary.get("sample_unit", "pair_level_sample_idx"),
            "ratio": _to_float(summary.get("ratio")),
            "total_train_size": _to_int(summary.get("total_train_size")),
            "subset_size": _to_int(summary.get("subset_size")) or selected_count,
            "feature_source": summary.get("feature_source"),
            "selection_time": selection_time,
            "train_time": train_time,
            "eval_time": eval_time,
            "output_dir": current_root,
            "baseline_summary_path": summary_path,
            "selected_indices_path": selected_path if os.path.exists(selected_path) else None,
            "downstream_metrics_path": metrics_path if os.path.exists(metrics_path) else None,
            **metrics,
        }
        records.append(record)
    return records


def _float_stats(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return None, None, 0
    np_arr = np.asarray(arr, dtype=np.float32)
    return float(np.mean(np_arr)), float(np.std(np_arr)), int(np_arr.shape[0])


def aggregate_main_table(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        method = r.get("method")
        budget = r.get("budget")
        if method is None or budget is None:
            continue
        grouped[(str(method), int(budget))].append(r)

    out: List[Dict[str, Any]] = []
    for (method, budget), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        agg: Dict[str, Any] = {
            "method": method,
            "budget": int(budget),
            "seed_count": int(len(rows)),
            "dataset": rows[0].get("dataset"),
            "image_encoder": rows[0].get("image_encoder"),
            "text_encoder": rows[0].get("text_encoder"),
            "sample_unit": rows[0].get("sample_unit"),
        }
        for metric in METRIC_COLUMNS + ["selection_time", "train_time", "eval_time"]:
            m, s, c = _float_stats([row.get(metric) for row in rows])
            agg[f"{metric}_mean"] = m
            agg[f"{metric}_std"] = s
            agg[f"{metric}_count"] = c
        out.append(agg)
    return out


def build_main_table_wide(agg_records: List[Dict[str, Any]], budgets: List[int]) -> List[Dict[str, Any]]:
    by_method: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in agg_records:
        method = str(row.get("method"))
        budget = _to_int(row.get("budget"))
        if budget is None:
            continue
        by_method[method][budget] = row

    out: List[Dict[str, Any]] = []
    for method in sorted(by_method.keys()):
        row: Dict[str, Any] = {"method": method}
        for budget in budgets:
            source = by_method[method].get(int(budget), {})
            for metric in METRIC_COLUMNS:
                row[f"R{int(budget)}_{metric}"] = source.get(f"{metric}_mean")
        out.append(row)
    return out


def _split_method_aliases(cell: str) -> List[str]:
    text = cell.replace("`", "").strip()
    parts = [p.strip() for p in text.split("/") if p.strip()]
    return [p.lower() for p in parts]


def parse_method_mapping_table(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    lines = []
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    rows = [line for line in lines if line.strip().startswith("|")]
    mapping: Dict[str, Dict[str, Any]] = {}
    for line in rows:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        if cells[0].lower() in {"method", "---"} or set(cells[0]) == {"-"}:
            continue
        aliases = _split_method_aliases(cells[0])
        payload = {
            "full_name": cells[1],
            "paper_or_source": cells[2],
            "multimodal_adaptation": cells[3],
            "reproduction_status": cells[4],
            "ambiguity_note": cells[5],
        }
        for alias in aliases:
            mapping[alias] = payload
    return mapping


def build_method_status_records(
    methods: List[str],
    mapping_doc_path: str,
    baselines_root: str,
) -> List[Dict[str, Any]]:
    mapping = parse_method_mapping_table(mapping_doc_path)
    out: List[Dict[str, Any]] = []
    category_map = {
        "entropy": "uncertainty-based",
        "el2n": "training-dynamics-based",
        "grand": "gradient-based",
        "gradmatch": "gradient-based",
        "glister": "bilevel/validation-based",
        "rand": "geometry/coverage-based",
        "ccs-rand": "geometry/coverage-based",
        "herd": "geometry/coverage-based",
        "ccs-herd": "geometry/coverage-based",
        "kcenter": "geometry/coverage-based",
        "ccs-kcenter": "geometry/coverage-based",
        "forget": "training-dynamics-based",
        "ccs-forget": "training-dynamics-based",
        "dq": "geometry/coverage-based",
        "dfool": "uncertainty-based",
        "nms": "geometry/coverage-based",
        "adap_sne": "geometry/coverage-based",
        "adapsne": "geometry/coverage-based",
        "presel": "geometry/coverage-based",
        "visa": "geometry/coverage-based",
        "dataprophet": "scoring-based",
        "dynamic_pruning": "training-dynamics-based",
        "infobatch": "training-dynamics-based",
    }

    eval_connected_methods = set()
    for run in scan_baseline_runs(baselines_root):
        if run.get("downstream_metrics_path"):
            eval_connected_methods.add(str(run.get("method")))

    for method in sorted(set(methods)):
        meta = mapping.get(method, {})
        cfg_path = None
        for candidate in _method_config_candidates(method):
            full = os.path.join("baselines", "configs", candidate)
            if os.path.exists(full):
                cfg_path = full
                break
        impl_file = os.path.join("baselines", "methods", f"{method.replace('-', '_')}.py")
        if not os.path.exists(impl_file):
            if method == "adapsne":
                impl_file = os.path.join("baselines", "methods", "adap_sne.py")
        out.append(
            {
                "method": method,
                "full_name": meta.get("full_name"),
                "paper_or_source": meta.get("paper_or_source"),
                "method_category": category_map.get(method),
                "reproduction_status": meta.get("reproduction_status"),
                "multimodal_adaptation": meta.get("multimodal_adaptation"),
                "implementation_file": impl_file if os.path.exists(impl_file) else None,
                "config_file": cfg_path,
                "ambiguity_note": meta.get("ambiguity_note"),
                "ready_for_main_table": bool(method in eval_connected_methods),
            }
        )
    return out


def build_protocol_alignment_records(
    methods: List[str],
    runs: List[Dict[str, Any]],
    expected_dataset: str = "flickr",
    expected_image_encoder: str = "nfnet",
    expected_text_encoder: str = "bert",
    expected_budgets: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    expected_budgets = expected_budgets or [100, 200, 500]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        method = r.get("method")
        if method:
            grouped[str(method)].append(r)

    out = []
    for method in sorted(set(methods)):
        rows = grouped.get(method, [])
        budget_set = {int(r["budget"]) for r in rows if r.get("budget") is not None}
        out.append(
            {
                "method": method,
                "dataset_aligned": bool(rows) and all(str(r.get("dataset")) == expected_dataset for r in rows),
                "encoders_aligned": bool(rows)
                and all(
                    str(r.get("image_encoder")) == expected_image_encoder
                    and str(r.get("text_encoder")) == expected_text_encoder
                    for r in rows
                ),
                "budgets_aligned": budget_set == set(int(x) for x in expected_budgets),
                "pair_level_aligned": bool(rows) and all(str(r.get("sample_unit")) == "pair_level_sample_idx" for r in rows),
                "feature_source_aligned": bool(rows) and all(r.get("feature_source") is not None for r in rows),
                "fusion_protocol_aligned": None,
                "output_isolated": bool(rows) and all("/artifacts/baselines" in str(r.get("output_dir")).replace("\\", "/") for r in rows),
                "downstream_eval_connected": bool(rows) and any(r.get("downstream_metrics_path") for r in rows),
                "notes": None if rows else "no run records found",
            }
        )
    return out


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# Main Table (Aligned)\n\nNo records.\n")
        return
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Main Table (Aligned)\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def export_all_tables(
    root: str,
    output_dir: str,
    budgets: Optional[List[int]],
    methods: List[str],
    mapping_doc_path: str,
    expected_dataset: str = "flickr",
    expected_image_encoder: str = "nfnet",
    expected_text_encoder: str = "bert",
) -> Dict[str, str]:
    runs = scan_baseline_runs(root)
    runs_sorted = sorted(
        runs,
        key=lambda r: (
            str(r.get("method") or ""),
            int(r.get("budget") or -1),
            int(r.get("seed") or -1),
        ),
    )
    budgets = budgets or sorted({int(r["budget"]) for r in runs_sorted if r.get("budget") is not None}) or [100, 200, 500]

    agg = aggregate_main_table(runs_sorted)
    wide = build_main_table_wide(agg, budgets=budgets)
    status = build_method_status_records(methods=methods, mapping_doc_path=mapping_doc_path, baselines_root=root)
    protocol = build_protocol_alignment_records(
        methods=methods,
        runs=runs_sorted,
        expected_dataset=expected_dataset,
        expected_image_encoder=expected_image_encoder,
        expected_text_encoder=expected_text_encoder,
        expected_budgets=budgets,
    )

    outputs = {
        "main_table_csv": os.path.join(output_dir, "main_table_aligned.csv"),
        "main_table_json": os.path.join(output_dir, "main_table_aligned.json"),
        "main_table_md": os.path.join(output_dir, "main_table_aligned.md"),
        "main_table_agg_csv": os.path.join(output_dir, "main_table_aligned_agg.csv"),
        "main_table_agg_json": os.path.join(output_dir, "main_table_aligned_agg.json"),
        "main_table_wide_csv": os.path.join(output_dir, "main_table_wide.csv"),
        "main_table_wide_json": os.path.join(output_dir, "main_table_wide.json"),
        "method_status_csv": os.path.join(output_dir, "baseline_method_status.csv"),
        "method_status_json": os.path.join(output_dir, "baseline_method_status.json"),
        "protocol_alignment_csv": os.path.join(output_dir, "baseline_protocol_alignment.csv"),
        "protocol_alignment_json": os.path.join(output_dir, "baseline_protocol_alignment.json"),
        "benchmark_summary_csv": os.path.join(output_dir, "benchmark_summary.csv"),
        "benchmark_summary_json": os.path.join(output_dir, "benchmark_summary.json"),
    }

    _write_csv(outputs["main_table_csv"], runs_sorted)
    _write_json(outputs["main_table_json"], runs_sorted)
    _write_md_table(outputs["main_table_md"], runs_sorted)
    _write_csv(outputs["main_table_agg_csv"], agg)
    _write_json(outputs["main_table_agg_json"], agg)
    _write_csv(outputs["main_table_wide_csv"], wide)
    _write_json(outputs["main_table_wide_json"], wide)
    _write_csv(outputs["method_status_csv"], status)
    _write_json(outputs["method_status_json"], status)
    _write_csv(outputs["protocol_alignment_csv"], protocol)
    _write_json(outputs["protocol_alignment_json"], protocol)

    benchmark_payload = {
        "root": root,
        "output_dir": output_dir,
        "budgets": budgets,
        "runs": runs_sorted,
        "aggregated": agg,
    }
    _write_csv(outputs["benchmark_summary_csv"], runs_sorted)
    _write_json(outputs["benchmark_summary_json"], benchmark_payload)
    return outputs
