#!/usr/bin/env python3
"""
Generate three schematic graph-wavelet scale illustrations.

Outputs:
  wavelet_low.png
  wavelet_mid.png
  wavelet_high.png
  wavelet_scales_triptych.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PALE_BLUE = "#7FB2E6"
DEEP_BLUE = "#3D79BD"
PALE_GREEN = "#8FC49A"
DEEP_GREEN = "#43865A"
GRID_BLUE = "#A8C9EA"
GRID_GREEN = "#A8CFB0"


def gaussian2d(x: np.ndarray, y: np.ndarray, cx: float, cy: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))


def mexican_hat2d(x: np.ndarray, y: np.ndarray, cx: float, cy: float, sigma: float, amp: float) -> np.ndarray:
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return amp * (1.0 - r2 / (sigma**2)) * np.exp(-r2 / (2.0 * sigma**2))


def build_surface(scale: str, n: int = 84) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    x = np.linspace(-3.2, 3.2, n)
    y = np.linspace(-3.2, 3.2, n)
    xx, yy = np.meshgrid(x, y)

    if scale == "low":
        zz = (
            gaussian2d(xx, yy, -1.9, -1.2, 0.60, 0.70)
            + gaussian2d(xx, yy, 1.15, 1.05, 0.72, 0.92)
            + gaussian2d(xx, yy, -0.35, 0.35, 0.55, 0.42)
        )
        title = "Low-scale / Coarse"
        return xx, yy, zz, title, PALE_BLUE, GRID_BLUE

    if scale == "mid":
        zz = (
            mexican_hat2d(xx, yy, -1.45, -0.95, 0.46, 0.62)
            + mexican_hat2d(xx, yy, 0.65, 0.65, 0.50, 0.72)
            + 0.24 * np.sin(1.7 * xx + 0.4) * np.exp(-0.13 * (xx**2 + yy**2))
        )
        title = "Mid-scale / Transition"
        return xx, yy, zz, title, "#78A8D8", "#9EC0E0"

    if scale == "high":
        zz = (
            mexican_hat2d(xx, yy, -1.65, -0.8, 0.30, 0.72)
            + mexican_hat2d(xx, yy, 0.6, 0.65, 0.32, 0.78)
            + mexican_hat2d(xx, yy, 1.75, -1.0, 0.25, 0.40)
            + 0.15 * np.sin(4.0 * xx) * np.cos(3.2 * yy) * np.exp(-0.12 * (xx**2 + yy**2))
        )
        title = "High-scale / Fine"
        return xx, yy, zz, title, PALE_GREEN, GRID_GREEN

    raise ValueError(f"Unsupported scale: {scale}")


def style_3d_axis(ax, title: str | None = None) -> None:
    ax.view_init(elev=26, azim=-58)
    ax.set_axis_off()
    ax.set_box_aspect((1.4, 1.0, 0.45))
    ax.set_xlim(-3.25, 3.25)
    ax.set_ylim(-3.25, 3.25)
    ax.set_zlim(-0.65, 1.15)
    ax.dist = 7.5
    if title:
        ax.set_title(title, fontsize=13, pad=-2, fontfamily="serif")


def draw_wavelet(ax, scale: str, show_title: bool = False) -> None:
    xx, yy, zz, title, face_color, grid_color = build_surface(scale)

    ax.plot_wireframe(
        xx,
        yy,
        np.zeros_like(zz) - 0.08,
        rstride=5,
        cstride=5,
        color=grid_color,
        linewidth=0.55,
        alpha=0.45,
    )
    ax.plot_surface(
        xx,
        yy,
        zz,
        rstride=1,
        cstride=1,
        color=face_color,
        edgecolor="none",
        linewidth=0,
        antialiased=True,
        alpha=0.78,
        shade=True,
    )
    ax.contour(
        xx,
        yy,
        zz,
        zdir="z",
        offset=-0.10,
        levels=7,
        colors=grid_color,
        linewidths=0.55,
        alpha=0.45,
    )
    style_3d_axis(ax, title if show_title else None)


def save_single(scale: str, output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(3.2, 2.7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    draw_wavelet(ax, scale, show_title=False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_dir / f"wavelet_{scale}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_triptych(output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(9.6, 3.0), facecolor="white")
    for idx, scale in enumerate(["low", "mid", "high"], start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        draw_wavelet(ax, scale, show_title=True)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.02)
    fig.savefig(output_dir / "wavelet_scales_triptych.png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate low/mid/high wavelet schematic figures.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/reports/results/wavelet_schematics"),
        help="Directory to save generated figures.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scale in ["low", "mid", "high"]:
        save_single(scale, args.output_dir, args.dpi)
    save_triptych(args.output_dir, args.dpi)
    print(f"Saved wavelet schematic figures to {args.output_dir}")


if __name__ == "__main__":
    main()
