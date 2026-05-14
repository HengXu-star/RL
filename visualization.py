"""Generate project figures from evaluation results and trained policies."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline import QTablePolicy, run_policy_episode
from environment import TaxiGridEnvironment
from mc import train_monte_carlo_control
from q_learning import train_q_learning
from sarsa import train_sarsa


def load_evaluation_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_method_comparison(csv_path: Path, output_path: Path, metric: str = "avg_total_reward") -> None:
    rows = load_evaluation_rows(csv_path)
    methods = [row["method"] for row in rows]
    values = [float(row[metric]) for row in rows]

    colors = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500", "#6a4c93"][: len(methods)]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, values, color=colors)
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("Method")
    plt.title("Method Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_reward_series(csv_path: Path) -> Tuple[List[int], List[float]]:
    episodes: List[int] = []
    rewards: List[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
    return episodes, rewards


def moving_average(values: List[float], window: int) -> List[float]:
    if len(values) < window:
        return values[:]

    averages: List[float] = []
    window_sum = sum(values[:window])
    averages.append(window_sum / window)
    for idx in range(window, len(values)):
        window_sum += values[idx] - values[idx - window]
        averages.append(window_sum / window)
    return averages


def plot_combined_learning_curves(
    output_path: Path,
    q_learning_csv: Path,
    sarsa_csv: Path,
    mc_csv: Path,
    window: int = 100,
) -> None:
    series = [
        ("Q-learning", q_learning_csv, "#1f77b4"),
        ("SARSA", sarsa_csv, "#d62728"),
        ("Monte Carlo", mc_csv, "#2ca02c"),
    ]

    plt.figure(figsize=(11, 6.5))
    for label, csv_path, color in series:
        episodes, rewards = load_reward_series(csv_path)
        smoothed = moving_average(rewards, window)
        smoothed_episodes = episodes[window - 1 :] if len(episodes) >= window else episodes

        plt.plot(episodes, rewards, color=color, alpha=0.10, linewidth=0.8)
        plt.plot(smoothed_episodes, smoothed, color=color, linewidth=2.2, label=label)

    plt.title("Learning Curves Comparison", fontsize=18)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_environment(seed: int, grid_size: int, time_slots: int, steps: int) -> TaxiGridEnvironment:
    return TaxiGridEnvironment(
        grid_size=grid_size,
        num_time_slots=time_slots,
        max_steps=steps,
        seed=seed,
    )


def train_method(
    method: str,
    env: TaxiGridEnvironment,
    episodes: int,
    seed: int,
):
    if method == "q_learning":
        return train_q_learning(env, episodes=episodes, seed=seed)
    if method == "sarsa":
        return train_sarsa(env, episodes=episodes, seed=seed)
    if method == "mc":
        return train_monte_carlo_control(env, episodes=episodes, seed=seed)
    raise ValueError(f"Unsupported method: {method}")


def collect_zone_visits(
    method: str,
    training_episodes: int,
    rollout_episodes: int,
    grid_size: int,
    time_slots: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    training_env = build_environment(seed=seed, grid_size=grid_size, time_slots=time_slots, steps=steps)
    q_table, _ = train_method(method, training_env, training_episodes, seed)
    policy_name = {
        "q_learning": "Q-learning",
        "sarsa": "SARSA",
        "mc": "Monte Carlo",
    }[method]
    policy = QTablePolicy(q_table=q_table, name=policy_name)

    visit_counts = np.zeros((grid_size, grid_size), dtype=float)
    eval_env = build_environment(seed=seed + 999, grid_size=grid_size, time_slots=time_slots, steps=steps)

    for _ in range(rollout_episodes):
        state = eval_env.reset()
        done = False

        zone, _ = state
        visit_counts[zone[0], zone[1]] += 1

        while not done:
            action = policy.select_action(eval_env, state)
            result = eval_env.step(action)
            state = result.state
            done = result.done
            zone, _ = state
            visit_counts[zone[0], zone[1]] += 1

    return visit_counts


def plot_zone_heatmap(
    output_path: Path,
    method: str = "q_learning",
    training_episodes: int = 3000,
    rollout_episodes: int = 200,
    grid_size: int = 4,
    time_slots: int = 6,
    steps: int = 24,
    seed: int = 1,
) -> None:
    visit_counts = collect_zone_visits(
        method=method,
        training_episodes=training_episodes,
        rollout_episodes=rollout_episodes,
        grid_size=grid_size,
        time_slots=time_slots,
        steps=steps,
        seed=seed,
    )

    display_name = {
        "q_learning": "Q-learning",
        "sarsa": "SARSA",
        "mc": "Monte Carlo",
    }[method]

    plt.figure(figsize=(6.5, 5.8))
    image = plt.imshow(visit_counts, cmap="YlOrRd")
    plt.colorbar(image, label="Visit count")
    plt.title(f"{display_name} Zone Visit Heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")

    for row in range(grid_size):
        for col in range(grid_size):
            plt.text(col, row, int(visit_counts[row, col]), ha="center", va="center", color="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create bar chart and heatmap figures for the RL project.")
    parser.add_argument("--evaluation-csv", type=Path, default=Path("results/evaluation_summary.csv"))
    parser.add_argument("--bar-chart-output", type=Path, default=Path("results/method_comparison_bar_chart.png"))
    parser.add_argument("--heatmap-output", type=Path, default=Path("results/q_learning_heatmap.png"))
    parser.add_argument("--combined-curves-output", type=Path, default=Path("results/combined_learning_curves.png"))
    parser.add_argument("--q-learning-csv", type=Path, default=Path("q_learning_rewards_github_env.csv"))
    parser.add_argument("--sarsa-csv", type=Path, default=Path("sarsa_rewards.csv"))
    parser.add_argument("--mc-csv", type=Path, default=Path("mc_rewards.csv"))
    parser.add_argument("--metric", type=str, default="avg_total_reward")
    parser.add_argument("--heatmap-method", choices=["q_learning", "sarsa", "mc"], default="q_learning")
    parser.add_argument("--training-episodes", type=int, default=3000)
    parser.add_argument("--rollout-episodes", type=int, default=200)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--time-slots", type=int, default=6)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    plot_combined_learning_curves(
        output_path=args.combined_curves_output,
        q_learning_csv=args.q_learning_csv,
        sarsa_csv=args.sarsa_csv,
        mc_csv=args.mc_csv,
        window=args.window,
    )
    plot_method_comparison(args.evaluation_csv, args.bar_chart_output, metric=args.metric)
    plot_zone_heatmap(
        output_path=args.heatmap_output,
        method=args.heatmap_method,
        training_episodes=args.training_episodes,
        rollout_episodes=args.rollout_episodes,
        grid_size=args.grid_size,
        time_slots=args.time_slots,
        steps=args.steps,
        seed=args.seed,
    )

    print(f"Saved combined learning curves to {args.combined_curves_output}")
    print(f"Saved bar chart to {args.bar_chart_output}")
    print(f"Saved heatmap to {args.heatmap_output}")


if __name__ == "__main__":
    main()
