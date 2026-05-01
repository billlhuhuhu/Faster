#!/usr/bin/env python3
"""
Create a reference-style ablation figure from the ablation Excel table.

The figure mirrors the provided layout:
  (a) module contribution lollipop chart using mean recall (mR)
  (b) metric-level retrieval heatmap over I2T/T2I recall metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_DATA_PATH = Path("/home/hzx/Faster/artifacts/reports/results/消融表格.xlsx")
DEFAULT_OUTPUT_PATH = Path("/home/hzx/Faster/artifacts/reports/results/ablation_reference_replica.png")

METHOD_TO_LABEL = {
    "flickr": "Full / Ours",
    "no_lsrc_lors": "w/o LSRC",
    "no_correction_no_adaptive_fusion": "w/o Correction +\nAdaptive Fusion",
    "no_wavelet_alignment": "w/o Wavelet\nAlignment",
}

LEFT_ORDER = [
    "flickr",
    "no_lsrc_lors",
    "no_correction_no_adaptive_fusion",
    "no_wavelet_alignment",
]

HEATMAP_ORDER = [
    "flickr",
    "no_correction_no_adaptive_fusion",
    "no_lsrc_lors",
    "no_wavelet_alignment",
]

METRIC_COLUMNS = [
    ("I2T-R@1", "i2t_r1"),
    ("I2T-R@5", "i2t_r5"),
    ("I2T-R@10", "i2t_r10"),
    ("T2I-R@1", "t2i_r1"),
    ("T2I-R@5", "t2i_r5"),
    ("T2I-R@10", "t2i_r10"),
]

BLUE = "#174A91"
LIGHT_BLUE = "#6E93C8"
RED = "#9A0007"
GRID = "#D7D7D7"


def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 14,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def _normalize_column_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("@", "")


def load_ablation_table(data_path: Path, budget: float) -> pd.DataFrame:
    raw = pd.read_excel(data_path)
    if "Method" not in raw.columns and "method" not in [_normalize_column_name(c) for c in raw.columns]:
        raw = pd.read_excel(data_path, header=None)

    normalized = {_normalize_column_name(col): col for col in raw.columns}
    if "method" in normalized:
        method_col = normalized["method"]
    else:
        method_col = raw.columns[0]

    if "threshold" in normalized:
        budget_col = normalized["threshold"]
    elif "budget" in normalized:
        budget_col = normalized["budget"]
    elif "ratio" in normalized:
        budget_col = normalized["ratio"]
    else:
        budget_col = raw.columns[1]

    metric_aliases = {
        "i2t_r1": ["i2t_r1", "image_r1", "img_r1", "r1_i2t"],
        "i2t_r5": ["i2t_r5", "image_r5", "img_r5", "r5_i2t"],
        "i2t_r10": ["i2t_r10", "image_r10", "img_r10", "r10_i2t"],
        "t2i_r1": ["t2i_r1", "text_r1", "txt_r1", "r1_t2i"],
        "t2i_r5": ["t2i_r5", "text_r5", "txt_r5", "r5_t2i"],
        "t2i_r10": ["t2i_r10", "text_r10", "txt_r10", "r10_t2i"],
    }

    metric_cols = {}
    for canonical, aliases in metric_aliases.items():
        for alias in aliases:
            if alias in normalized:
                metric_cols[canonical] = normalized[alias]
                break

    if len(metric_cols) < 6:
        # Fallback for headerless tables:
        # Method, Threshold, I2T-R@1, I2T-R@5, I2T-R@10, T2I-R@1, T2I-R@5, T2I-R@10, ...
        if raw.shape[1] >= 8:
            method_col = raw.columns[0]
            budget_col = raw.columns[1]
            metric_cols = {
                "i2t_r1": raw.columns[2],
                "i2t_r5": raw.columns[3],
                "i2t_r10": raw.columns[4],
                "t2i_r1": raw.columns[5],
                "t2i_r5": raw.columns[6],
                "t2i_r10": raw.columns[7],
            }
        else:
            raise ValueError("Expected six retrieval metric columns for I2T/T2I R@1/R@5/R@10.")

    table = pd.DataFrame(
        {
            "method": raw[method_col].astype(str),
            "budget": pd.to_numeric(raw[budget_col], errors="coerce"),
            **{name: pd.to_numeric(raw[col], errors="coerce") for name, col in metric_cols.items()},
        }
    )
    table = table.dropna(subset=["budget"])
    table = table[np.isclose(table["budget"].astype(float), float(budget))]
    if table.empty:
        raise ValueError(f"No rows found for budget={budget} in {data_path}")
    return table


def metric_row(table: pd.DataFrame, method: str) -> np.ndarray:
    row = table[table["method"] == method]
    if row.empty:
        raise ValueError(f"Missing method '{method}' in ablation table.")
    return row.iloc[0][[name for _, name in METRIC_COLUMNS]].to_numpy(dtype=float)


def draw_left_panel(ax: plt.Axes, table: pd.DataFrame) -> None:
    full_mr = float(metric_row(table, "flickr").mean())
    y_positions = np.arange(len(LEFT_ORDER))
    mr_values = [float(metric_row(table, method).mean()) for method in LEFT_ORDER]

    ax.axvline(full_mr, color=BLUE, linestyle="--", linewidth=1.3, zorder=1)
    ax.text(
        full_mr,
        -0.7,
        f"Full / Ours ({full_mr:.2f})",
        ha="right",
        va="bottom",
        color=BLUE,
        fontsize=14,
        fontweight="bold",
    )

    for y, method, mr in zip(y_positions, LEFT_ORDER, mr_values):
        is_full = method == "flickr"
        color = BLUE if method != "no_wavelet_alignment" else RED
        marker_face = BLUE if is_full else "white"
        marker_size = 62 if is_full else 58

        if not is_full:
            ax.hlines(y, mr, full_mr, color=color, linestyle=(0, (3, 2)), linewidth=1.2, zorder=2)
            ax.text(mr - 0.22, y, f"{mr:.2f}", ha="right", va="center", color=color, fontsize=14)
            ax.text(
                full_mr + 0.12,
                y,
                f"{mr - full_mr:.2f}",
                ha="left",
                va="center",
                color=color,
                fontsize=14,
            )

        ax.scatter(
            [mr],
            [y],
            s=marker_size,
            facecolors=marker_face,
            edgecolors=color,
            linewidths=1.4,
            zorder=4,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([METHOD_TO_LABEL[m] for m in LEFT_ORDER])
    for tick, method in zip(ax.get_yticklabels(), LEFT_ORDER):
        if method == "flickr":
            tick.set_color(BLUE)
            tick.set_fontweight("bold")

    ax.set_xlim(15.55, 21.55)
    ax.set_ylim(len(LEFT_ORDER) - 0.15, -0.55)
    ax.set_xlabel("mR")
    ax.set_title("(a) Module Contribution Analysis", fontweight="bold", pad=24)
    ax.xaxis.grid(True, linestyle=(0, (2, 3)), color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)


def draw_heatmap(ax: plt.Axes, table: pd.DataFrame, fig: plt.Figure) -> None:
    data = np.vstack([metric_row(table, method) for method in HEATMAP_ORDER])
    labels = [METHOD_TO_LABEL[m] for m in HEATMAP_ORDER]

    cmap = LinearSegmentedColormap.from_list("reference_blues", ["#F7FBFF", "#7EA2D0", "#083B88"])
    image = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=40)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            color = "white" if value >= 18 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=13,
                fontweight="bold" if value >= 18 else "normal",
            )

    ax.set_xticks(np.arange(len(METRIC_COLUMNS)))
    ax.set_xticklabels([label for label, _ in METRIC_COLUMNS], fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for tick, method in zip(ax.get_yticklabels(), HEATMAP_ORDER):
        if method == "flickr":
            tick.set_color(BLUE)
            tick.set_fontweight("bold")

    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_title("(b) Metric-Level Retrieval Summary", fontweight="bold", pad=42)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.ax.set_title("Recall (%)", fontsize=12, pad=8)
    cbar.set_ticks([0, 10, 20, 30, 40])
    cbar.ax.tick_params(labelsize=12, length=0)
    cbar.outline.set_visible(False)


def create_ablation_chart(data_path: Path, output_path: Path, budget: float = 0.03) -> None:
    configure_matplotlib()
    table = load_ablation_table(data_path, budget)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.4, 5.8), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.18],
        left=0.06,
        right=0.965,
        bottom=0.18,
        top=0.72,
        wspace=0.50,
    )
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    draw_left_panel(ax_left, table)
    draw_heatmap(ax_right, table, fig)

    fig.suptitle(f"Flickr, budget = {budget:.2f}", y=0.975, fontsize=18)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Chart saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reference-style ablation chart.")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--budget", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_ablation_chart(args.data_path, args.output_path, budget=args.budget)


if __name__ == "__main__":
    main()
