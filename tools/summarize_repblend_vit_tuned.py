import argparse
import csv
import re
from pathlib import Path


EVAL_RE = re.compile(
    r"\[Eval_00\]\s+Ep(?P<epoch>\d+)\s+\|\s+"
    r"Image R@1=(?P<i2t_r1>[-+0-9.]+)\s+"
    r"R@5=(?P<i2t_r5>[-+0-9.]+)\s+"
    r"R@10=(?P<i2t_r10>[-+0-9.]+)\s+\|\s+"
    r"Text R@1=(?P<t2i_r1>[-+0-9.]+)\s+"
    r"R@5=(?P<t2i_r5>[-+0-9.]+)\s+"
    r"R@10=(?P<t2i_r10>[-+0-9.]+)\s+\|\s+"
    r"Mean=(?P<mean_recall>[-+0-9.]+)"
)


def parse_log(path: Path):
    rows = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for match in EVAL_RE.finditer(text):
        row = {"epoch": int(match.group("epoch"))}
        for key in ("i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall"):
            row[key] = float(match.group(key))
        rows.append(row)
    return rows


def config_from_path(path: Path):
    name = path.stem
    prefix = "repblend_vit_"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def format_float(value):
    if value is None:
        return ""
    return f"{float(value):.2f}"


def write_markdown(path: Path, rows):
    headers = [
        "config",
        "selected",
        "epoch",
        "i2t_r1",
        "i2t_r5",
        "i2t_r10",
        "t2i_r1",
        "t2i_r5",
        "t2i_r10",
        "mean_recall",
        "log_path",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if header not in {"config", "selected", "log_path"} and value != "":
                value = format_float(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize RepBlend ViT tuned evaluation logs.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="artifacts/repblend_vit_tuned_3pct/logs",
        help="Directory containing repblend_vit_*.log files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/repblend_vit_tuned_3pct/reports",
        help="Directory for summary CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="projection_only low_lr_finetune very_low_lr_finetune",
        help="Space-separated config names to summarize.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    eval_rows = []
    for config in args.configs.split():
        log_path = log_dir / f"repblend_vit_{config}.log"
        if not log_path.exists():
            candidates = sorted(log_dir.glob(f"*{config}*.log"))
            log_path = candidates[-1] if candidates else log_path
        parsed = parse_log(log_path) if log_path.exists() else []
        for row in parsed:
            eval_rows.append({"config": config, "log_path": str(log_path), **row})
        if not parsed:
            summary_rows.append(
                {
                    "config": config,
                    "selected": "missing",
                    "epoch": "",
                    "i2t_r1": "",
                    "i2t_r5": "",
                    "i2t_r10": "",
                    "t2i_r1": "",
                    "t2i_r5": "",
                    "t2i_r10": "",
                    "mean_recall": "",
                    "log_path": str(log_path),
                }
            )
            continue
        best = max(parsed, key=lambda item: item["mean_recall"])
        final = parsed[-1]
        summary_rows.append({"config": config, "selected": "best", "log_path": str(log_path), **best})
        summary_rows.append({"config": config, "selected": "final", "log_path": str(log_path), **final})

    summary_csv = output_dir / "repblend_vit_tuned_summary.csv"
    fields = [
        "config",
        "selected",
        "epoch",
        "i2t_r1",
        "i2t_r5",
        "i2t_r10",
        "t2i_r1",
        "t2i_r5",
        "t2i_r10",
        "mean_recall",
        "log_path",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    eval_csv = output_dir / "repblend_vit_tuned_all_evals.csv"
    with eval_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["config", "epoch", "i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall", "log_path"],
        )
        writer.writeheader()
        writer.writerows(eval_rows)

    summary_md = output_dir / "repblend_vit_tuned_summary.md"
    write_markdown(summary_md, summary_rows)

    print(f"saved summary csv: {summary_csv}")
    print(f"saved summary md: {summary_md}")
    print(f"saved all evals csv: {eval_csv}")


if __name__ == "__main__":
    main()
