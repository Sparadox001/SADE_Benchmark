"""Configuration schema and defaults for DSI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DSIConfig:
    max_evaluations: int = 300
    population_size: int = 30
    wmax: int = 10
    seed: int = 1
    constraint_tolerance: float = 0.0
    save_candidate_pools: bool = False

    def __post_init__(self) -> None:
        if self.population_size < 5:
            raise ValueError("population_size must be at least 5 for C2oDE.")
        if self.max_evaluations < self.population_size:
            raise ValueError("max_evaluations must be at least population_size.")
        if self.wmax < 1:
            raise ValueError("wmax must be positive.")

