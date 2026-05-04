from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple


State = Tuple[Tuple[int, int], int]


@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    done: bool
    info: Dict[str, float]


class TaxiGridEnvironment:
    """
    A small grid-world simulator for taxi repositioning.

    State:
        ((row, col), time_slot)

    Actions:
        0 -> stay
        1 -> up
        2 -> down
        3 -> left
        4 -> right
    """

    ACTIONS = {
        0: (0, 0),   # stay
        1: (-1, 0),  # up
        2: (1, 0),   # down
        3: (0, -1),  # left
        4: (0, 1),   # right
    }

    ACTION_NAMES = {
        0: "stay",
        1: "up",
        2: "down",
        3: "left",
        4: "right",
    }

    def __init__(
        self,
        grid_size: int = 4,
        num_time_slots: int = 6,
        max_steps: int = 24,
        move_cost: float = 1.0,
        wait_cost: float = 0.3,
        base_fare: float = 8.0,
        pickup_bonus: float = 0.0,
        seed: Optional[int] = None,
        demand_map: Optional[Dict[Tuple[Tuple[int, int], int], float]] = None,
        fare_map: Optional[Dict[Tuple[Tuple[int, int], int], float]] = None,
    ) -> None:
        self.grid_size = grid_size
        self.num_time_slots = num_time_slots
        self.max_steps = max_steps
        self.move_cost = move_cost
        self.wait_cost = wait_cost
        self.base_fare = base_fare
        self.pickup_bonus = pickup_bonus
        self.random = random.Random(seed)

        self.demand_map = demand_map or self._build_default_demand_map()
        self.fare_map = fare_map or {}

        self.position = (0, 0)
        self.time_slot = 0
        self.steps_taken = 0

    def reset(
        self,
        start_zone: Optional[Tuple[int, int]] = None,
        start_time_slot: int = 0,
    ) -> State:
        if start_zone is None:
            start_zone = (
                self.random.randrange(self.grid_size),
                self.random.randrange(self.grid_size),
            )

        self.position = start_zone
        self.time_slot = start_time_slot % self.num_time_slots
        self.steps_taken = 0
        return self.state

    @property
    def state(self) -> State:
        return (self.position, self.time_slot)

    def step(self, action: int) -> StepResult:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Valid actions: {list(self.ACTIONS)}")

        next_position, moved = self._transition(self.position, action)
        current_time_slot = self.time_slot

        pickup_probability = self._get_pickup_probability(next_position, current_time_slot)
        pickup = self.random.random() < pickup_probability
        fare = self._get_fare(next_position, current_time_slot) if pickup else 0.0

        action_cost = self.move_cost if moved else self.wait_cost
        reward = fare + (self.pickup_bonus if pickup else 0.0) - action_cost

        self.position = next_position
        self.time_slot = (self.time_slot + 1) % self.num_time_slots
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps

        info = {
            "pickup": float(pickup),
            "pickup_probability": pickup_probability,
            "fare": fare,
            "action_cost": action_cost,
            "moved": float(moved),
        }
        return StepResult(state=self.state, reward=reward, done=done, info=info)

    def sample_action(self) -> int:
        return self.random.choice(list(self.ACTIONS))

    def render(self) -> str:
        rows: List[str] = []
        for r in range(self.grid_size):
            cells: List[str] = []
            for c in range(self.grid_size):
                cells.append("T" if (r, c) == self.position else ".")
            rows.append(" ".join(cells))
        return "\n".join(rows) + f"\ntime_slot={self.time_slot}"

    def _transition(self, position: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], bool]:
        row, col = position
        d_row, d_col = self.ACTIONS[action]
        next_row = min(max(row + d_row, 0), self.grid_size - 1)
        next_col = min(max(col + d_col, 0), self.grid_size - 1)
        next_position = (next_row, next_col)
        moved = next_position != position
        return next_position, moved

    def _get_pickup_probability(self, zone: Tuple[int, int], time_slot: int) -> float:
        return self.demand_map.get((zone, time_slot), 0.1)

    def _get_fare(self, zone: Tuple[int, int], time_slot: int) -> float:
        return self.fare_map.get((zone, time_slot), self.base_fare)

    def _build_default_demand_map(self) -> Dict[Tuple[Tuple[int, int], int], float]:
        """
        Create a simple synthetic demand pattern.
        Center cells and late time slots have slightly higher demand.
        """
        demand_map: Dict[Tuple[Tuple[int, int], int], float] = {}
        center = (self.grid_size - 1) / 2

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                distance_from_center = abs(row - center) + abs(col - center)
                zone_score = max(0.15, 0.55 - 0.08 * distance_from_center)

                for time_slot in range(self.num_time_slots):
                    rush_multiplier = 0.12 if time_slot in {1, 2, 4, 5} else 0.0
                    demand = min(0.95, zone_score + rush_multiplier)
                    demand_map[((row, col), time_slot)] = demand

        return demand_map


def build_uniform_demand_map(
    grid_size: int,
    num_time_slots: int,
    pickup_probability: float,
) -> Dict[Tuple[Tuple[int, int], int], float]:
    demand_map: Dict[Tuple[Tuple[int, int], int], float] = {}
    for row in range(grid_size):
        for col in range(grid_size):
            for time_slot in range(num_time_slots):
                demand_map[((row, col), time_slot)] = pickup_probability
    return demand_map


if __name__ == "__main__":
    env = TaxiGridEnvironment(grid_size=4, num_time_slots=6, max_steps=8, seed=7)
    state = env.reset(start_zone=(0, 0), start_time_slot=0)
    print("Initial state:", state)
    print(env.render())

    for step_idx in range(env.max_steps):
        action = env.sample_action()
        result = env.step(action)
        print(
            f"\nStep {step_idx + 1}: action={env.ACTION_NAMES[action]}, "
            f"next_state={result.state}, reward={result.reward:.2f}, done={result.done}, info={result.info}"
        )
        print(env.render())
        if result.done:
            break
