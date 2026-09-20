"""Configuration schema and defaults for continuous SADE."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SADEConfig:
    max_evaluations: int = 300
    population_size: int = 30
    batch_size: int = 3
    seed: int = 1
    trials_per_target: int = 6
    local_fraction: float = 0.2
    surrogate_min_samples: int = 30
    p_best_fraction: float = 0.2
    constraint_tolerance: float = 0.0
    distance_threshold: float = 1e-3
    save_candidate_pools: bool = False
    dynamic_constraint_tightening: bool = False
    tightening_trigger_feasible_count: int | None = None
    tightening_keep_ratio: float = 0.3
    tightening_patience: int = 3
    tightening_improvement_tol: float = 1e-3
    tightening_max_tightens: int = 100
    tightening_min_ratio: float = 1e-3

    def __post_init__(self) -> None:
        if self.population_size < 6:
            raise ValueError("population_size must be at least 6.")
        if self.max_evaluations < self.population_size:
            raise ValueError("max_evaluations must be at least population_size.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.trials_per_target < 1:
            raise ValueError("trials_per_target must be positive.")
        if self.trials_per_target > 10:
            raise ValueError("trials_per_target cannot exceed the 10 retained strategies.")
        if not 0.0 <= self.local_fraction <= 1.0:
            raise ValueError("local_fraction must be in [0, 1].")
        if self.surrogate_min_samples < 2:
            raise ValueError("surrogate_min_samples must be at least 2.")
        if not 0.0 < self.p_best_fraction <= 1.0:
            raise ValueError("p_best_fraction must be in (0, 1].")
        if self.distance_threshold < 0.0:
            raise ValueError("distance_threshold cannot be negative.")
        if self.tightening_trigger_feasible_count is not None and not (
            1 <= self.tightening_trigger_feasible_count <= self.population_size
        ):
            raise ValueError(
                "tightening_trigger_feasible_count must be between 1 and "
                "population_size."
            )
        if not 0.0 < self.tightening_keep_ratio <= 1.0:
            raise ValueError("tightening_keep_ratio must be in (0, 1].")
        if self.tightening_patience < 1:
            raise ValueError("tightening_patience must be positive.")
        if self.tightening_improvement_tol < 0.0:
            raise ValueError("tightening_improvement_tol cannot be negative.")
        if self.tightening_max_tightens < 1:
            raise ValueError("tightening_max_tightens must be positive.")
        if self.tightening_min_ratio < 0.0:
            raise ValueError("tightening_min_ratio cannot be negative.")
