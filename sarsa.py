"""Tabular SARSA for the taxi repositioning project.

SARSA is an on-policy temporal-difference method. Unlike Q-learning, its
target uses the next action actually selected by the current epsilon-greedy
policy:

    Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

This file uses the same environment helpers as `q_learning.py`, so the learned
policy can be compared directly against Q-learning and the baselines.
"""

from __future__ import annotations

import csv
import random
from typing import Any, List, Tuple

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


def train_sarsa(
    env: Any,
    episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    seed: int = 1,
) -> Tuple[np.ndarray, List[float]]:
    """Train a tabular SARSA agent on the shared taxi environment.

    Parameters:
        env: Taxi environment with reset() and step().
        episodes: Number of training episodes.
        alpha: Learning rate.
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
    rewards_per_episode: List[float] = []

    for _ in range(episodes):
        state = env.reset()
        state_id = encode_state(env, state)
        action = epsilon_greedy_action(q_table, state_id, n_actions, epsilon)

        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, done, info = unpack_step_result(env.step(action))
            next_state_id = encode_state(env, next_state)

            if done:
                target = reward
            else:
                next_action = epsilon_greedy_action(q_table, next_state_id, n_actions, epsilon)
                target = reward + gamma * q_table[next_state_id, next_action]

            old_q = q_table[state_id, action]
            q_table[state_id, action] = old_q + alpha * (target - old_q)

            total_reward += reward
            state_id = next_state_id
            if not done:
                action = next_action

        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode


def extract_greedy_policy(q_table: np.ndarray) -> np.ndarray:
    """Extract the best action for each state from a trained Q-table."""

    return np.argmax(q_table, axis=1)


def save_learning_curve_data(rewards_per_episode: List[float], filename: str = "sarsa_rewards.csv") -> None:
    """Save SARSA episode rewards as learning curve data."""

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "total_reward"])
        for episode, reward in enumerate(rewards_per_episode, start=1):
            writer.writerow([episode, reward])


if __name__ == "__main__":
    from environment import TaxiGridEnvironment

    env = TaxiGridEnvironment(grid_size=4, num_time_slots=6, max_steps=24, seed=1)
    q_table, rewards = train_sarsa(env, episodes=3000, seed=1)
    save_learning_curve_data(rewards, "sarsa_rewards.csv")

    final_window = min(100, len(rewards))
    average_reward = sum(rewards[-final_window:]) / final_window
    print(f"Saved SARSA learning curve data to sarsa_rewards.csv")
    print(f"Average reward over final {final_window} episodes: {average_reward:.2f}")
