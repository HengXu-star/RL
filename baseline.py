"""Baseline policies for the taxi repositioning project.

These policies share the same environment used by the tabular RL methods, so
their results can be compared directly against Q-learning and SARSA.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Protocol


class Policy(Protocol):
    """Minimal policy interface used by the evaluator."""

    name: str

    def select_action(self, env: Any, state: Any) -> int:
        """Choose an action for the current environment state."""


@dataclass
class RandomPolicy:
    """Choose uniformly among all available actions."""

    seed: int = 1
    name: str = "Random"

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def select_action(self, env: Any, state: Any) -> int:
        actions = list(env.ACTIONS) if hasattr(env, "ACTIONS") else list(range(5))
        return self._random.choice(actions)


@dataclass
class GreedyDemandPolicy:
    """Move to the neighboring cell with the highest immediate demand."""

    name: str = "Greedy"

    def select_action(self, env: Any, state: Any) -> int:
        _, time_slot = state
        best_action = 0
        best_score = float("-inf")

        for action in sorted(env.ACTIONS):
            next_position, _ = env._transition(env.position, action)
            demand = env._get_pickup_probability(next_position, time_slot)
            cost = env.move_cost if next_position != env.position else env.wait_cost
            score = demand * env._get_fare(next_position, time_slot) - cost

            if score > best_score:
                best_score = score
                best_action = action

        return best_action


@dataclass
class QTablePolicy:
    """Greedy policy induced by a learned tabular Q-table."""

    q_table: Any
    name: str = "Q-learning"

    def select_action(self, env: Any, state: Any) -> int:
        from q_learning import encode_state

        state_id = encode_state(env, state)
        return int(self.q_table[state_id].argmax())


def run_policy_episode(env: Any, policy: Policy) -> Dict[str, float]:
    """Run one episode and collect business-facing metrics."""

    state = env.reset()
    done = False

    total_reward = 0.0
    pickups = 0
    idle_steps = 0
    moved_steps = 0
    total_cost = 0.0
    total_fare = 0.0
    steps = 0

    while not done:
        action = policy.select_action(env, state)
        result = env.step(action)
        state = result.state
        done = result.done

        pickup = int(result.info.get("pickup", 0.0))
        moved = int(result.info.get("moved", 0.0))

        total_reward += float(result.reward)
        pickups += pickup
        idle_steps += 1 - pickup
        moved_steps += moved
        total_cost += float(result.info.get("action_cost", 0.0))
        total_fare += float(result.info.get("fare", 0.0))
        steps += 1

    return {
        "total_reward": total_reward,
        "steps": float(steps),
        "pickups": float(pickups),
        "pickup_rate": pickups / steps if steps else 0.0,
        "idle_steps": float(idle_steps),
        "moved_steps": float(moved_steps),
        "moving_rate": moved_steps / steps if steps else 0.0,
        "total_fare": total_fare,
        "total_cost": total_cost,
    }


def summarize_episode_metrics(episode_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-episode metrics into one comparison row."""

    if not episode_metrics:
        raise ValueError("episode_metrics must contain at least one episode")

    summary: Dict[str, float] = {}
    keys = episode_metrics[0].keys()
    for key in keys:
        summary[key] = sum(metrics[key] for metrics in episode_metrics) / len(episode_metrics)

    summary["episodes"] = float(len(episode_metrics))
    return summary
