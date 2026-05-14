# Smarter Streets: Training a Reinforcement Learning Agent for Dynamic Taxi Dispatch and Zone Navigation

Group 27 | Heng Xu, Xinyu Tian, Yuanheng Xia, Binqi Zhu, Yu Liang

## Overview

This project studies how reinforcement learning can help taxi drivers decide where to reposition between trips in order to increase earnings while reduce idle travel. We model the problem as a small grid-world Markov Decision Process and compare:

- Random policy
- Greedy heuristic
- Q-learning
- SARSA
- Monte Carlo control

The repository includes:

- `environment.py`: taxi grid-world environment
- `simulator.py`: runnable simulator demo
- `q_learning.py`, `sarsa.py`, `mc.py`: tabular RL methods
- `baseline.py`: random and greedy baselines
- `evaluation.py`: comparison metrics and summary table
- `visualization.py`: generates plots from results

## Problem Formulation

We define the taxi repositioning problem as a tabular MDP.

- State: `(zone, time_slot)`
- Actions: `stay`, `up`, `down`, `left`, `right`
- Reward: `fare_revenue - repositioning_cost`

This is a sequential decision-making problem because each move changes the taxi's future location and future revenue opportunities.

## Requirements

This project was tested with Python 3.11.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Fastest Way to Verify the Project

If you only want to confirm that the code runs end-to-end, execute these three commands:

```bash
python3 simulator.py --steps 5
python3 evaluation.py --episodes 20 --training-episodes 100 --output results/evaluation_summary.csv
python3 visualization.py
```

After that, check:

- `results/evaluation_summary.csv`
- `results/method_comparison_bar_chart.png`
- `results/q_learning_heatmap.png`
- `results/combined_learning_curves.png`

## Full Reproducible Workflow

### 1. Run the Environment Demo

```bash
python3 -u simulator.py --steps 5
```

Optional data-driven mode:

```bash
python3 -u simulator.py --csv /path/to/taxi_data.csv --steps 10
```

### 2. Train Individual RL Methods

Q-learning:

```bash
python3 q_learning.py
```

SARSA:

```bash
python3 sarsa.py
```

Monte Carlo:

```bash
python3 mc.py
```

These commands generate reward CSV files that can be used for learning curves.

### 3. Run Evaluation

This compares Random, Greedy, Q-learning, SARSA, and Monte Carlo in the same environment.

```bash
python3 evaluation.py --episodes 500 --training-episodes 3000 --output results/evaluation_summary.csv
```

For a faster smoke test:

```bash
python3 evaluation.py --episodes 20 --training-episodes 100 --output results/evaluation_summary.csv
```

### 4. Generate Plots

This command generates:

- a combined learning curve figure
- a bar chart comparing methods
- a heatmap showing learned zone visitation behavior

```bash
python3 visualization.py
```

### 5. Generate One Learning Curve Only

Example:

```bash
python3 plot_learning_curve.py q_learning_rewards_github_env.csv q_learning_learning_curve.png --method-name "Q-learning" --window 100
```

## Expected Outputs

Main result files are stored in `results/`:

- `evaluation_summary.csv`
- `method_comparison_bar_chart.png`
- `q_learning_heatmap.png`
- `combined_learning_curves.png`

Other repository outputs include:

- `q_learning_rewards_github_env.csv`
- `sarsa_rewards.csv`
- `mc_rewards.csv`

## Notes

- The default environment is a simplified 4x4 synthetic grid-world.
- `data_utils.py` supports converting historical taxi data into demand and fare maps.
- The plotting scripts are configured to work in headless environments.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── environment.py
├── simulator.py
├── data_utils.py
├── q_learning.py
├── sarsa.py
├── mc.py
├── baseline.py
├── evaluation.py
├── visualization.py
├── plot_learning_curve.py
└── results/
```
