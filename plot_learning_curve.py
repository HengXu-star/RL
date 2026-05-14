"""Plot learning curves with the same style used for Q-learning results.

This version writes SVG using only the Python standard library so it works even
when matplotlib is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError:
    plt = None
    np = None


WIDTH = 1600
HEIGHT = 900
LEFT = 126
RIGHT = 34
TOP = 64
BOTTOM = 105
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = HEIGHT - TOP - BOTTOM


def read_rewards(csv_path: Path) -> Tuple[List[int], List[float]]:
    """Read episode reward data from a two-column CSV file."""

    episodes: List[int] = []
    rewards: List[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))

    return episodes, rewards


def moving_average(values: List[float], window: int) -> List[float]:
    """Return a simple moving average that starts after one full window."""

    if len(values) < window:
        return []

    averages: List[float] = []
    window_sum = sum(values[:window])
    averages.append(window_sum / window)

    for idx in range(window, len(values)):
        window_sum += values[idx] - values[idx - window]
        averages.append(window_sum / window)

    return averages


def plot_png_with_matplotlib(
    csv_path: Path,
    output_path: Path,
    method_name: str,
    window: int,
) -> bool:
    """Create a PNG with matplotlib when it is available."""

    if plt is None or np is None:
        return False

    episodes, rewards = read_rewards(csv_path)
    averages = np.convolve(np.array(rewards), np.ones(window) / window, mode="valid")
    average_episodes = episodes[window - 1 :]

    plt.figure(figsize=(16.2, 9), dpi=100)
    plt.plot(episodes, rewards, color="#1f77b4", alpha=0.22, linewidth=1.0, label="Episode reward")
    plt.plot(
        average_episodes,
        averages,
        color="#ff7f0e",
        linewidth=2.2,
        label=f"{window}-episode moving average",
    )
    plt.title(f"{method_name} Learning Curve", fontsize=22, pad=10)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Total reward", fontsize=18)
    plt.tick_params(axis="both", labelsize=16)
    plt.legend(loc="upper right", fontsize=17)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def scale_points(
    episodes: List[int],
    rewards: List[float],
    x_max: int,
    y_min: float,
    y_max: float,
) -> str:
    """Convert data points into an SVG polyline point string."""

    points: List[str] = []
    y_span = y_max - y_min

    for episode, reward in zip(episodes, rewards):
        x = LEFT + (episode / x_max) * PLOT_WIDTH
        y = TOP + ((y_max - reward) / y_span) * PLOT_HEIGHT
        points.append(f"{x:.2f},{y:.2f}")

    return " ".join(points)


def add_text(
    parts: List[str],
    x: float,
    y: float,
    text: str,
    size: int,
    anchor: str = "middle",
    rotate: bool = False,
) -> None:
    """Append an SVG text element."""

    escaped = html.escape(text)
    transform = f' transform="rotate(-90 {x:.2f} {y:.2f})"' if rotate else ""
    parts.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="black"{transform}>{escaped}</text>'
    )


def plot_learning_curve(
    csv_path: Path,
    output_path: Path,
    method_name: str,
    window: int = 100,
) -> None:
    """Create a learning curve matching the Q-learning figure style."""

    if output_path.suffix.lower() == ".png" and plot_png_with_matplotlib(
        csv_path,
        output_path,
        method_name,
        window,
    ):
        return

    episodes, rewards = read_rewards(csv_path)
    averages = moving_average(rewards, window)
    average_episodes = episodes[window - 1 :]

    x_max = max(episodes)
    y_min = 0.0
    y_max = max(170.0, ((max(rewards) // 10) + 2) * 10)

    episode_points = scale_points(episodes, rewards, x_max, y_min, y_max)
    average_points = scale_points(average_episodes, averages, x_max, y_min, y_max)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    add_text(parts, WIDTH / 2, 49, f"{method_name} Learning Curve", 31)
    add_text(parts, WIDTH / 2, 867, "Episode", 27)
    add_text(parts, 45, HEIGHT / 2, "Total reward", 27, rotate=True)

    parts.append(
        f'<rect x="{LEFT}" y="{TOP}" width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" '
        'fill="none" stroke="black" stroke-width="2"/>'
    )

    for tick in range(0, x_max + 1, 1000):
        x = LEFT + (tick / x_max) * PLOT_WIDTH
        parts.append(f'<line x1="{x:.2f}" y1="{TOP + PLOT_HEIGHT}" x2="{x:.2f}" y2="{TOP + PLOT_HEIGHT + 9}" stroke="black" stroke-width="2"/>')
        add_text(parts, x, TOP + PLOT_HEIGHT + 37, str(tick), 25)

    for tick in range(20, int(y_max), 20):
        y = TOP + ((y_max - tick) / (y_max - y_min)) * PLOT_HEIGHT
        parts.append(f'<line x1="{LEFT - 9}" y1="{y:.2f}" x2="{LEFT}" y2="{y:.2f}" stroke="black" stroke-width="2"/>')
        add_text(parts, LEFT - 18, y + 8, str(tick), 25, anchor="end")

    parts.append(f'<polyline points="{episode_points}" fill="none" stroke="#1f77b4" stroke-opacity="0.22" stroke-width="1.1"/>')
    parts.append(f'<polyline points="{average_points}" fill="none" stroke="#ff7f0e" stroke-width="4"/>')

    legend_x = 1126
    legend_y = 77
    legend_w = 458
    legend_h = 76
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" '
        'rx="4" fill="white" stroke="#d0d0d0" stroke-width="2"/>'
    )
    parts.append(f'<line x1="{legend_x + 10}" y1="{legend_y + 20}" x2="{legend_x + 61}" y2="{legend_y + 20}" stroke="#1f77b4" stroke-opacity="0.22" stroke-width="2"/>')
    add_text(parts, legend_x + 79, legend_y + 29, "Episode reward", 25, anchor="start")
    parts.append(f'<line x1="{legend_x + 10}" y1="{legend_y + 57}" x2="{legend_x + 61}" y2="{legend_y + 57}" stroke="#ff7f0e" stroke-width="4"/>')
    add_text(parts, legend_x + 79, legend_y + 66, f"{window}-episode moving average", 25, anchor="start")

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a learning curve from reward CSV data.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    plot_learning_curve(args.csv_path, args.output_path, args.method_name, args.window)
    print(f"Saved {args.method_name} learning curve to {args.output_path}")


if __name__ == "__main__":
    main()
