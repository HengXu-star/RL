"""First-visit Monte Carlo control for the taxi repositioning project.

Monte Carlo control is included as an optional comparison method. It estimates
Q-values from complete episode returns, then improves an epsilon-greedy policy.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from q_learning import encode_state, get_num_actions, get_num_states, unpack_step_result


def epsilon_greedy_action(
    q_table: np.ndarray,
    state_id: int,
    n_actions: int,
    epsilon: float,
) -> int:
    """Choose an action using epsilon-greedy exploration."""

    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)

    return int(np.argmax(q_table[state_id]))


def train_monte_carlo_control(
    env: Any,
    episodes: int = 5000,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    seed: int = 1,
) -> Tuple[np.ndarray, List[float]]:
    """Train a first-visit Monte Carlo control agent.

    Parameters:
        env: Taxi environment with reset() and step().
        episodes: Number of training episodes.
        gamma: Discount factor.
        epsilon: Exploration probability for epsilon-greedy action selection.
        seed: Random seed for reproducible results.

    Returns:
        q_table: Learned Q-values with shape (n_states, n_actions).
        rewards_per_episode: Total reward collected in each episode.
    """

    random.seed(seed)
    np.random.seed(seed)

    n_states = get_num_states(env)
    n_actions = get_num_actions(env)
    q_table = np.zeros((n_states, n_actions))

    returns_sum: Dict[Tuple[int, int], float] = defaultdict(float)
    returns_count: Dict[Tuple[int, int], int] = defaultdict(int)
    rewards_per_episode: List[float] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        episode_history: List[Tuple[int, int, float]] = []

        while not done:
            state_id = encode_state(env, state)
            action = epsilon_greedy_action(q_table, state_id, n_actions, epsilon)
            next_state, reward, done, info = unpack_step_result(env.step(action))

            episode_history.append((state_id, action, reward))
            total_reward += reward
            state = next_state

        returns_by_step = [0.0] * len(episode_history)
        return_so_far = 0.0

        for idx in range(len(episode_history) - 1, -1, -1):
            _, _, reward = episode_history[idx]
            return_so_far = reward + gamma * return_so_far
            returns_by_step[idx] = return_so_far

        visited_state_actions = set()
        for idx, (state_id, action, _) in enumerate(episode_history):
            state_action = (state_id, action)
            if state_action in visited_state_actions:
                continue

            visited_state_actions.add(state_action)
            returns_sum[state_action] += returns_by_step[idx]
            returns_count[state_action] += 1
            q_table[state_id, action] = returns_sum[state_action] / returns_count[state_action]

        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode


def extract_greedy_policy(q_table: np.ndarray) -> np.ndarray:
    """Extract the best action for each state from a trained Q-table."""

    return np.argmax(q_table, axis=1)


def save_learning_curve_data(rewards_per_episode: List[float], filename: str = "mc_rewards.csv") -> None:
    """Save Monte Carlo episode rewards as learning curve data."""

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "total_reward"])
        for episode, reward in enumerate(rewards_per_episode, start=1):
            writer.writerow([episode, reward])


if __name__ == "__main__":
    from environment import TaxiGridEnvironment

    env = TaxiGridEnvironment(grid_size=4, num_time_slots=6, max_steps=24, seed=1)
    q_table, rewards = train_monte_carlo_control(env, episodes=3000, seed=1)
    save_learning_curve_data(rewards, "mc_rewards.csv")

    final_window = min(100, len(rewards))
    average_reward = sum(rewards[-final_window:]) / final_window
    print(f"Saved Monte Carlo learning curve data to mc_rewards.csv")
    print(f"Average reward over final {final_window} episodes: {average_reward:.2f}")
