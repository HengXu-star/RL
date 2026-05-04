"""Tabular Q-learning for the taxi repositioning project.

This file only implements the Q-learning part of the project. It works with the
group environment in the GitHub repo:

    state = env.reset()
    result = env.step(action)

The GitHub environment returns states like ((row, col), time_slot), so this file
converts those tabular states into integer IDs before using the Q-table.
"""

from __future__ import annotations

import csv
import random
from typing import Any, List, Tuple

import numpy as np


def get_num_states(env: Any) -> int:
    """Return the number of states in the environment.

    Some environments expose env.n_states directly. The group GitHub
    environment does not, so we calculate it from grid size and time slots.
    """

    if hasattr(env, "n_states"):
        return int(env.n_states)

    grid_size = int(env.grid_size)
    if hasattr(env, "num_time_slots"):
        num_time_slots = int(env.num_time_slots)
    else:
        num_time_slots = int(env.time_slots)

    return grid_size * grid_size * num_time_slots


def get_num_actions(env: Any) -> int:
    """Return the number of available actions."""

    if hasattr(env, "n_actions"):
        return int(env.n_actions)

    if hasattr(env, "ACTIONS"):
        return len(env.ACTIONS)

    # Our project uses five actions: stay, up, down, left, right.
    return 5


def encode_state(env: Any, state: Any) -> int:
    """Convert an environment state into an integer Q-table row index.

    Supported state formats:
        1. integer state ID, already encoded
        2. (zone, time_slot)
        3. ((row, col), time_slot), used by the current GitHub environment
    """

    if isinstance(state, (int, np.integer)):
        return int(state)

    if hasattr(env, "state_to_index"):
        return int(env.state_to_index(state))

    zone, time_slot = state

    if isinstance(zone, tuple):
        row, col = zone
        zone_id = int(row) * int(env.grid_size) + int(col)
    else:
        zone_id = int(zone)

    if hasattr(env, "num_time_slots"):
        num_time_slots = int(env.num_time_slots)
    else:
        num_time_slots = int(env.time_slots)

    return zone_id * num_time_slots + int(time_slot)


def unpack_step_result(step_result: Any) -> Tuple[Any, float, bool, dict]:
    """Read step output from either a tuple or the project's StepResult object."""

    if isinstance(step_result, tuple):
        next_state, reward, done, info = step_result
        return next_state, float(reward), bool(done), info

    if hasattr(step_result, "next_state"):
        next_state = step_result.next_state
    else:
        next_state = step_result.state

    return next_state, float(step_result.reward), bool(step_result.done), step_result.info


def epsilon_greedy_action(
    q_table: np.ndarray,
    state_id: int,
    n_actions: int,
    epsilon: float,
) -> int:
    """Choose an action using epsilon-greedy exploration.

    With probability epsilon, the agent explores by choosing a random action.
    With probability 1 - epsilon, the agent exploits by choosing the action
    with the highest Q-value for the current state.
    """

    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)

    return int(np.argmax(q_table[state_id]))


def train_q_learning(
    env,
    episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    seed: int = 1,
) -> Tuple[np.ndarray, List[float]]:
    """Train a tabular Q-learning agent.

    Parameters:
        env: Taxi environment with reset() and step().
        episodes: Number of training episodes.
        alpha: Learning rate. Controls how much new information updates Q.
        gamma: Discount factor. Controls importance of future rewards.
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

    # Rows are states, columns are actions. Values start at zero because the
    # agent has not learned which actions are good yet.
    q_table = np.zeros((n_states, n_actions))
    rewards_per_episode: List[float] = []

    for _ in range(episodes):
        state = env.reset()
        state_id = encode_state(env, state)
        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy_action(q_table, state_id, n_actions, epsilon)
            next_state, reward, done, info = unpack_step_result(env.step(action))
            next_state_id = encode_state(env, next_state)

            # Q-learning target:
            # immediate reward + discounted value of the best next action.
            # If the episode is done, there is no future state value to add.
            best_next_q = 0.0 if done else np.max(q_table[next_state_id])
            target = reward + gamma * best_next_q

            # Q-learning update formula:
            # Q(s,a) = Q(s,a) + alpha * [target - Q(s,a)]
            old_q = q_table[state_id, action]
            q_table[state_id, action] = old_q + alpha * (target - old_q)

            total_reward += reward
            state = next_state
            state_id = next_state_id

        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode


def extract_greedy_policy(q_table: np.ndarray) -> np.ndarray:
    """Extract the best action for each state from a trained Q-table.

    The greedy policy chooses the action with the highest Q-value in every
    state. The returned array has one action per state.
    """

    return np.argmax(q_table, axis=1)


def save_learning_curve_data(rewards_per_episode: List[float], filename: str = "q_learning_rewards.csv") -> None:
    """Save episode rewards as learning curve data.

    The CSV can be used later to plot episode number versus total reward.
    """

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "total_reward"])
        for episode, reward in enumerate(rewards_per_episode, start=1):
            writer.writerow([episode, reward])
