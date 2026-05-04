#!/usr/bin/env python3
"""
NeurIPS-style ablation figure for Flickr at budget = 0.03

Figure layout:
(a) Overall Ablation Summary
    - horizontal bar chart of mR
    - value labels + relative drop from Full / Ours

(b) Metric-Level Retrieval Summary
    - heatmap of six retrieval metrics + mR

Input Excel format (no header):
method, budget, i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_OURS = "#B71C1C"
COLOR_SECOND = "#90A4AE"
COLOR_THIRD = "#90A4AE"
COLOR_FOURTH = "#B0BEC5"
COLOR_TEXT = "#333333"

METHOD_ORDER = [
    "flickr",
    "no_lsrc_lors",
    "no_correction_no_adaptive_fusion",
    "no_wavelet_alignment",
]

METHOD_LABELS = {
    "flickr": "Full / Ours",
    "no_lsrc_lors": "w/o LSRC",
    "no_correction_no_adaptive_fusion": "w/o Correction +\nAdaptive Fusion",
    "no_wavelet_alignment": "w/o Wavelet\nAlignment",
}

METRIC_COLS = [
    "i2t_r1",
    "i2t_r5",
    "i2t_r10",
    "t2i_r1",
    "t2i_r5",
    "t2i_r10",
]

METRIC_LABELS = [
    "I2T-R@1",
    "I2T-R@5",
    "I2T-R@10",
    "T2I-R@1",
    "T2I-R@5",
    "T2I-R@10",
]


def load_budget_data(data_path: str, budget: float = 0.03) -> pd.DataFrame:
    df = pd.read_excel(data_path, header=None)
    df.columns = [
        "method",
        "budget",
        "i2t_r1",
        "i2t_r5",
        "i2t_r10",
        "t2i_r1",
        "t2i_r5",
        "t2i_r10",
    ]
    df["method"] = df["method"].astype(str)
    df["budget"] = df["budget"].astype(float)

    df_budget = df[np.isclose(df["budget"], budget)].copy()
    df_budget = df_budget[df_budget["method"].isin(METHOD_ORDER)].copy()

    if len(df_budget) != len(METHOD_ORDER):
        missing = sorted(set(METHOD_ORDER) - set(df_budget["method"]))
        raise ValueError(f"Missing methods for budget={budget}: {missing}")

    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    df_budget["method_order"] = df_budget["method"].map(order_map)
    df_budget = df_budget.sort_values("method_order").reset_index(drop=True)
    df_budget["mR"] = df_budget[METRIC_COLS].mean(axis=1)
    df_budget["display_name"] = df_budget["method"].map(METHOD_LABELS)
    return df_budget


def create_ablation_chart(data_path: str, output_prefix: str, budget: float = 0.03) -> None:
    df = load_budget_data(data_path, budget=budget)

    heatmap_cols = METRIC_COLS + ["mR"]
    heatmap_labels = METRIC_LABELS + ["mR"]
    heatmap_data = df[heatmap_cols].to_numpy(dtype=float)

    method_labels = df["display_name"].tolist()
    mR_values = df["mR"].to_numpy(dtype=float)
    full_mR = float(df.loc[df["method"] == "flickr", "mR"].iloc[0])
    drops = mR_values - full_mR

    fig = plt.figure(figsize=(10.2, 3.8), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.26)
    fig.suptitle("Flickr, budget = 0.03", fontsize=12, y=0.985)

    ax1 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(method_labels))
    colors = [COLOR_OURS, COLOR_SECOND, COLOR_THIRD, COLOR_FOURTH]
    bars = ax1.barh(
        y_pos,
        mR_values,
        height=0.55,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    for i, (bar, val) in enumerate(zip(bars, mR_values)):
        y_center = bar.get_y() + bar.get_height() / 2
        if i == 0:
            label = f"{val:.2f}"
            color = COLOR_OURS
            weight = "bold"
        else:
            label = f"{val:.2f} ({drops[i]:.2f})"
            color = COLOR_TEXT
            weight = "normal"
        ax1.text(
            val + 0.08,
            y_center,
            label,
            ha="left",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight=weight,
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(method_labels, fontsize=7.5)
    ax1.invert_yaxis()
    ax1.set_xlim(17.0, 21.6)
    ax1.set_xlabel("Mean Recall (mR)", fontsize=8)
    ax1.xaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("(a) Overall Ablation Summary", fontweight="bold", fontsize=9, pad=9)

    ax2 = fig.add_subplot(gs[0, 1])
    cmap = LinearSegmentedColormap.from_list(
        "custom_heatmap",
        ["#F7FBFF", "#D8E8F7", "#8CB9DD", "#4A79B8", "#1E4E95"],
    )
    im = ax2.imshow(
        heatmap_data,
        cmap=cmap,
        aspect="auto",
        vmin=np.min(heatmap_data),
        vmax=np.max(heatmap_data),
    )

    max_val = np.max(heatmap_data)
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data[i, j]
            text_color = "white" if val > 0.60 * max_val else COLOR_TEXT
            fontweight = "bold" if (i == 0 or j == heatmap_data.shape[1] - 1) else "normal"
            ax2.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color=text_color,
                fontweight=fontweight,
            )

    ax2.set_xticks(np.arange(len(heatmap_labels)))
    ax2.set_xticklabels(heatmap_labels, rotation=25, ha="right", fontsize=7.5)
    ax2.set_yticks(np.arange(len(method_labels)))
    ax2.set_yticklabels(method_labels, fontsize=7.5)
    ax2.axvline(len(heatmap_labels) - 1.5, color="white", linewidth=1.3)
    ax2.set_title("(b) Metric-Level Retrieval Summary", fontweight="bold", fontsize=9, pad=9)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax2, fraction=0.036, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Recall", fontsize=7)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.965, top=0.84, bottom=0.18, wspace=0.26)

    png_path = output_prefix + ".png"
    pdf_path = output_prefix + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")

    # Extra export: no global title text
    st = fig._suptitle
    if st is not None:
        st.set_visible(False)
    png_path_notitle = output_prefix + "_notitle.png"
    pdf_path_notitle = output_prefix + "_notitle.pdf"
    fig.savefig(png_path_notitle, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path_notitle, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG (no title): {png_path_notitle}")
    print(f"Saved PDF (no title): {pdf_path_notitle}")
    print("\nBudget = 0.03 summary:")
    print(df[["display_name"] + METRIC_COLS + ["mR"]].to_string(index=False))


if __name__ == "__main__":
    data_path = "/home/hzx/Faster/artifacts/reports/results/消融表格.xlsx"
    output_prefix = "/home/hzx/Faster/artifacts/reports/results/ablation_budget003_neurips_fixed"
    create_ablation_chart(data_path=data_path, output_prefix=output_prefix, budget=0.03)
