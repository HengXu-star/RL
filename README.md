# Smarter Streets: Training a Reinforcement Learning Agent for Dynamic Taxi Dispatch and Zone Navigation

Group 27 | Heng Xu, Xinyu Tian, Yuanheng Xia, Binqi Zhu, Yu Liang

## Project Overview

This project studies how reinforcement learning can help taxi drivers decide where to reposition between trips in order to increase earnings while reducing idle travel. The repository currently includes a custom grid-world environment, a simulator, and utilities for estimating demand and fare patterns from taxi trip data.

## 1. Title

Smarter Streets: Training a Reinforcement Learning Agent for Dynamic Taxi Dispatch and Zone Navigation

## 2. Problem & "Why RL?"

Urban taxi drivers face a continuous challenge: where to reposition between trips to maximize daily earnings while minimizing idle travel. This is not a one-shot prediction problem. Each repositioning decision affects future state because moving to a busy zone now may lead to a fare, but that fare may then place the driver in a lower-demand area later.

Reinforcement learning is a natural fit because the driver, treated as the agent, must make repeated decisions under uncertainty and learn from delayed rewards over an entire shift. The environment is stochastic because demand changes across zones and time slots. Tabular RL methods such as Q-learning and SARSA are well-suited to this setup because the environment is discrete and finite.

## 3. MDP Formulation

### State Space

The state at time `t` is defined as `(zone, time_slot)`, where:

- `zone` is the driver's current discrete city location in the grid
- `time_slot` is the discretized time period within the shift

This representation captures the two main drivers of demand: location and time of day.

### Action Space

At each step, the agent chooses one of the following discrete actions:

- `stay`
- `move_north`
- `move_south`
- `move_east`
- `move_west`

Actions are pruned automatically by grid boundaries.

### Reward Function

The reward is defined as:

`reward = fare_revenue - repositioning_cost`

- If a passenger is picked up in the destination zone during the next time slot, the agent receives a positive reward based on fare
- If no passenger is picked up, the agent receives a small negative penalty representing idle repositioning or waiting cost

This reward structure encourages the agent to seek profitable zones without rewarding unnecessary movement.

## 4. Environment and Data Strategy

We build a custom Python grid-world simulator that models a simplified city map. The current implementation supports two data modes:

- Synthetic demand: a built-in demand map creates a simple stochastic environment where central zones and selected time slots have higher pickup probability.
- Data-driven demand: historical taxi trip records can be converted into `demand_map` and `fare_map` values using `data_utils.py`. Pickup locations are discretized into grid cells, pickup times are converted into time slots, and historical fares are averaged by zone and time slot.

The intended real-world data source is the NYC Taxi Trip Record dataset from NYC TLC.

## 5. Baselines and Evaluation

We plan to compare trained RL agents such as Q-learning and SARSA against simple baseline policies:

- Random policy: select a neighboring zone uniformly at random at each step.
- Greedy heuristic: move to the zone with the highest historically observed demand for the current time slot.

Primary evaluation metric:

- Cumulative episode reward

Secondary evaluation metrics:

- Pickup rate
- Average idle time
- Learning curves across training episodes

We will also compare Q-learning and SARSA to study off-policy versus on-policy behavior in this environment.

## Repository Contents

- `environment.py`: grid-world taxi repositioning environment with `reset()` and `step()`
- `simulator.py`: runnable simulator entry point
- `data_utils.py`: utilities for estimating demand and fare maps from taxi trip data

## Quick Start

Run the synthetic simulator:

```bash
python3 -u simulator.py --steps 5
```

Run with a taxi CSV file:

```bash
python3 -u simulator.py --csv /path/to/taxi_data.csv --steps 10
```
