from __future__ import annotations

import argparse

from data_utils import estimate_maps_from_taxi_data
from environment import TaxiGridEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the taxi grid-world simulator.")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--time-slots", type=int, default=6)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default=None, help="Optional path to taxi trip CSV.")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--zone-column", type=str, default=None)
    args = parser.parse_args()

    demand_map = None
    fare_map = None

    if args.csv:
        result = estimate_maps_from_taxi_data(
            csv_path=args.csv,
            grid_size=args.grid_size,
            num_time_slots=args.time_slots,
            zone_column=args.zone_column,
            sample_size=args.sample_size,
        )
        demand_map = result.demand_map
        fare_map = result.fare_map
        print(f"Loaded {len(demand_map)} zone-time demand estimates from {args.csv}")

    env = TaxiGridEnvironment(
        grid_size=args.grid_size,
        num_time_slots=args.time_slots,
        max_steps=args.steps,
        seed=args.seed,
        demand_map=demand_map,
        fare_map=fare_map,
    )

    state = env.reset()
    print("Initial state:", state)
    print(env.render())

    for step_idx in range(args.steps):
        action = env.sample_action()
        result = env.step(action)
        print(
            f"\nStep {step_idx + 1}: action={env.ACTION_NAMES[action]}, "
            f"next_state={result.state}, reward={result.reward:.2f}, info={result.info}"
        )
        print(env.render())
        if result.done:
            break


if __name__ == "__main__":
    main()
