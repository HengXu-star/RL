from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


GridCell = Tuple[int, int]
ZoneTimeKey = Tuple[GridCell, int]


@dataclass(frozen=True)
class DemandEstimationResult:
    demand_map: Dict[ZoneTimeKey, float]
    fare_map: Dict[ZoneTimeKey, float]
    cell_trip_counts: Dict[ZoneTimeKey, int]


def estimate_maps_from_taxi_data(
    csv_path: str,
    grid_size: int = 4,
    num_time_slots: int = 6,
    pickup_time_column: str = "tpep_pickup_datetime",
    fare_column: str = "fare_amount",
    longitude_column: str = "pickup_longitude",
    latitude_column: str = "pickup_latitude",
    zone_column: Optional[str] = None,
    min_trip_count: int = 1,
    sample_size: Optional[int] = None,
) -> DemandEstimationResult:
    """
    Build demand and fare maps from historical taxi trip data.

    Two supported input styles:
    1. Raw coordinates with longitude/latitude columns.
    2. Predefined discrete zone IDs via `zone_column`.
    """
    df = pd.read_csv(csv_path)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    required_columns = [pickup_time_column, fare_column]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    if zone_column is None:
        for column in (longitude_column, latitude_column):
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")

    working = df.copy()
    working[pickup_time_column] = pd.to_datetime(working[pickup_time_column], errors="coerce")
    working = working.dropna(subset=[pickup_time_column, fare_column])
    working = working[working[fare_column] > 0]

    if zone_column is not None:
        if zone_column not in working.columns:
            raise ValueError(f"Missing required zone column: {zone_column}")
        working = working.dropna(subset=[zone_column])
        working["grid_cell"] = _zone_ids_to_grid(working[zone_column], grid_size)
    else:
        working = working.dropna(subset=[longitude_column, latitude_column])
        working["grid_cell"] = _coordinates_to_grid(
            working[longitude_column],
            working[latitude_column],
            grid_size,
        )

    working["time_slot"] = working[pickup_time_column].dt.hour.apply(
        lambda hour: _hour_to_time_slot(hour, num_time_slots)
    )

    trip_counts = (
        working.groupby(["grid_cell", "time_slot"], observed=True)
        .size()
        .rename("trip_count")
        .reset_index()
    )

    total_trips_per_slot = (
        working.groupby(["time_slot"], observed=True)
        .size()
        .rename("slot_total")
        .reset_index()
    )

    fare_stats = (
        working.groupby(["grid_cell", "time_slot"], observed=True)[fare_column]
        .mean()
        .rename("avg_fare")
        .reset_index()
    )

    merged = trip_counts.merge(total_trips_per_slot, on="time_slot", how="left")
    merged = merged.merge(fare_stats, on=["grid_cell", "time_slot"], how="left")
    merged = merged[merged["trip_count"] >= min_trip_count]

    demand_map: Dict[ZoneTimeKey, float] = {}
    fare_map: Dict[ZoneTimeKey, float] = {}
    cell_trip_counts: Dict[ZoneTimeKey, int] = {}

    for row in merged.itertuples(index=False):
        key = (row.grid_cell, int(row.time_slot))
        demand_map[key] = min(1.0, float(row.trip_count) / float(row.slot_total))
        fare_map[key] = float(row.avg_fare)
        cell_trip_counts[key] = int(row.trip_count)

    return DemandEstimationResult(
        demand_map=demand_map,
        fare_map=fare_map,
        cell_trip_counts=cell_trip_counts,
    )


def _coordinates_to_grid(
    longitudes: pd.Series,
    latitudes: pd.Series,
    grid_size: int,
) -> pd.Series:
    lon_min, lon_max = longitudes.min(), longitudes.max()
    lat_min, lat_max = latitudes.min(), latitudes.max()

    if lon_min == lon_max or lat_min == lat_max:
        raise ValueError("Coordinate columns do not vary enough to define a grid.")

    lon_bins = pd.cut(
        longitudes,
        bins=grid_size,
        labels=False,
        include_lowest=True,
    )
    lat_bins = pd.cut(
        latitudes,
        bins=grid_size,
        labels=False,
        include_lowest=True,
    )

    # Reverse row order so larger latitude appears nearer the top of the grid.
    rows = (grid_size - 1) - lat_bins.astype(int)
    cols = lon_bins.astype(int)
    return pd.Series(list(zip(rows, cols)), index=longitudes.index)


def _zone_ids_to_grid(zone_ids: Iterable[int], grid_size: int) -> pd.Series:
    normalized = pd.Series(zone_ids).astype(int)
    total_cells = grid_size * grid_size
    mapped = normalized % total_cells
    rows = mapped // grid_size
    cols = mapped % grid_size
    return pd.Series(list(zip(rows, cols)), index=normalized.index)


def _hour_to_time_slot(hour: int, num_time_slots: int) -> int:
    slot_width = 24 / num_time_slots
    slot = int(hour / slot_width)
    return min(slot, num_time_slots - 1)
