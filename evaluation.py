"""Evaluate baseline and learned policies for taxi repositioning.

The main comparison metrics are total reward, pickup rate, idle steps, and
average reward. These match the business-facing evaluation plan for the project.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from baseline import GreedyDemandPolicy, QTablePolicy, RandomPolicy, run_policy_episode, summarize_episode_metrics
from environment import TaxiGridEnvironment


METRIC_FIELDS = [
    "method",
    "episodes",
    "avg_total_reward",
    "avg_reward_per_step",
    "pickup_rate",
    "avg_idle_steps",
    "avg_moved_steps",
    "moving_rate",
    "avg_total_fare",
    "avg_total_cost",
]


def evaluate_policy(env: Any, policy: Any, episodes: int) -> Dict[str, float | str]:
    """Evaluate a policy over many episodes and return one table row."""

    episode_metrics = [run_policy_episode(env, policy) for _ in range(episodes)]
    summary = summarize_episode_metrics(episode_metrics)

    return {
        "method": policy.name,
        "episodes": summary["episodes"],
        "avg_total_reward": summary["total_reward"],
        "avg_reward_per_step": summary["total_reward"] / summary["steps"] if summary["steps"] else 0.0,
        "pickup_rate": summary["pickup_rate"],
        "avg_idle_steps": summary["idle_steps"],
        "avg_moved_steps": summary["moved_steps"],
        "moving_rate": summary["moving_rate"],
        "avg_total_fare": summary["total_fare"],
        "avg_total_cost": summary["total_cost"],
    }


def build_default_environment(seed: int, grid_size: int, time_slots: int, steps: int) -> TaxiGridEnvironment:
    """Create the shared synthetic environment used for method comparison."""

    return TaxiGridEnvironment(
        grid_size=grid_size,
        num_time_slots=time_slots,
        max_steps=steps,
        seed=seed,
    )


def run_evaluation(
    episodes: int = 500,
    training_episodes: int = 3000,
    grid_size: int = 4,
    time_slots: int = 6,
    steps: int = 24,
    seed: int = 1,
    include_q_learning: bool = True,
    include_sarsa: bool = True,
    include_mc: bool = True,
) -> List[Dict[str, float | str]]:
    """Run baseline comparison and optionally include learned RL methods."""

    rows: List[Dict[str, float | str]] = []

    policies: List[Any] = [
        RandomPolicy(seed=seed),
        GreedyDemandPolicy(),
    ]

    if include_q_learning:
        from q_learning import train_q_learning

        training_env = build_default_environment(seed=seed, grid_size=grid_size, time_slots=time_slots, steps=steps)
        q_table, _ = train_q_learning(training_env, episodes=training_episodes, seed=seed)
        policies.append(QTablePolicy(q_table=q_table))

    if include_sarsa:
        from sarsa import train_sarsa

        training_env = build_default_environment(seed=seed, grid_size=grid_size, time_slots=time_slots, steps=steps)
        q_table, _ = train_sarsa(training_env, episodes=training_episodes, seed=seed)
        policies.append(QTablePolicy(q_table=q_table, name="SARSA"))

    if include_mc:
        from mc import train_monte_carlo_control

        training_env = build_default_environment(seed=seed, grid_size=grid_size, time_slots=time_slots, steps=steps)
        q_table, _ = train_monte_carlo_control(training_env, episodes=training_episodes, seed=seed)
        policies.append(QTablePolicy(q_table=q_table, name="Monte Carlo"))

    for idx, policy in enumerate(policies):
        eval_env = build_default_environment(
            seed=seed + 1000 + idx,
            grid_size=grid_size,
            time_slots=time_slots,
            steps=steps,
        )
        rows.append(evaluate_policy(eval_env, policy, episodes=episodes))

    return rows


def write_results_csv(rows: Iterable[Dict[str, float | str]], output_path: Path) -> None:
    """Write evaluation rows to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in METRIC_FIELDS})


def print_results_table(rows: List[Dict[str, float | str]]) -> None:
    """Print a compact terminal table for quick inspection."""

    header = f"{'Method':<12} {'Avg Reward':>12} {'Pickup Rate':>12} {'Idle Steps':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:<12} "
            f"{float(row['avg_total_reward']):>12.2f} "
            f"{float(row['pickup_rate']):>12.2%} "
            f"{float(row['avg_idle_steps']):>12.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate taxi repositioning baselines.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--training-episodes", type=int, default=3000)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--time-slots", type=int, default=6)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation_summary.csv"))
    parser.add_argument("--no-q-learning", action="store_true")
    parser.add_argument("--no-sarsa", action="store_true")
    parser.add_argument("--no-mc", action="store_true")
    args = parser.parse_args()

    rows = run_evaluation(
        episodes=args.episodes,
        training_episodes=args.training_episodes,
        grid_size=args.grid_size,
        time_slots=args.time_slots,
        steps=args.steps,
        seed=args.seed,
        include_q_learning=not args.no_q_learning,
        include_sarsa=not args.no_sarsa,
        include_mc=not args.no_mc,
    )
    write_results_csv(rows, args.output)
    print_results_table(rows)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
