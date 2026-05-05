# Smarter Streets Presentation Outline

## Slide 1: Title

Smarter Streets: Training a Reinforcement Learning Agent for Dynamic Taxi Dispatch and Zone Navigation

Group 27  
Heng Xu, Xinyu Tian, Yuanheng Xia, Binqi Zhu, Yu Liang

## Slide 2: Problem

- Taxi drivers must repeatedly decide where to move between trips.
- A good move now can affect future opportunities and future revenue.
- This is a sequential decision-making problem under uncertainty, not a one-time prediction task.

Speaker note:
We model taxi repositioning as a reinforcement learning problem because each decision affects the next state, and demand changes by zone and time.

## Slide 3: MDP Formulation

- State: `(zone, time_slot)`
- Actions: `stay`, `up`, `down`, `left`, `right`
- Reward:
  `reward = fare_revenue - repositioning_cost`
- Goal: maximize long-run cumulative reward

Speaker note:
The state includes both location and time because those are the two main factors affecting demand.

## Slide 4: Environment and Data

- Custom Python grid-world simulator
- 4x4 city grid with stochastic pickup demand
- Demand and fare can come from:
  - synthetic demand map
  - historical taxi data processed into `demand_map` and `fare_map`

Files:
- `environment.py`
- `simulator.py`
- `data_utils.py`

## Slide 5: Methods Compared

- Random policy
- Greedy demand heuristic
- Q-learning
- SARSA
- Monte Carlo control

Speaker note:
This gives us both business baselines and RL baselines for comparison.

## Slide 6: Learning Curves

Show:
- `q_learning_learning_curve.png`
- `sarsa_learning_curve.png`
- `mc_learning_curve.png`

Speaker note:
These curves show whether the algorithms improve over training episodes and whether performance stabilizes.

## Slide 7: Final Evaluation Results

Use:
- `results/method_comparison_bar_chart.png`
- `results/evaluation_summary.csv`

Key results from 500 evaluation episodes:

| Method | Avg Total Reward | Pickup Rate | Avg Idle Steps |
| --- | ---: | ---: | ---: |
| Random | 73.04 | 47.04% | 12.71 |
| Greedy | 81.66 | 46.28% | 12.89 |
| Q-learning | 79.61 | 49.62% | 12.09 |
| SARSA | 88.84 | 53.87% | 11.07 |
| Monte Carlo | 86.76 | 52.26% | 11.46 |

Speaker note:
SARSA achieved the best overall performance in our current environment, with the highest average reward and pickup rate.

## Slide 8: Heatmap Insight

Show:
- `results/q_learning_heatmap.png`

Speaker note:
The heatmap shows which zones the learned policy visits most frequently. This helps us interpret policy behavior instead of only reporting reward numbers.

## Slide 9: Conclusion

- RL outperformed the random baseline
- SARSA performed best in the current experiments
- Learned policies improved pickup rate and reduced idle behavior
- The framework can be extended with richer taxi demand data and larger state spaces

## Slide 10: Team Contribution

- Member 1: Environment + data pipeline
- Member 2: Q-learning
- Member 3: SARSA + Monte Carlo
- Member 4: Baselines + evaluation
- Member 5: Visualization + presentation
