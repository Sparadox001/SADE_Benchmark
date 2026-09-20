"""Configuration for DSI with dynamic objective-constraint tightening."""

from __future__ import annotations

from dataclasses import dataclass

from ..dsi.config import DSIConfig


@dataclass(frozen=True)
class DSIDynamicConfig(DSIConfig):
    """DSI parameters plus the SADE dynamic objective-constraint schedule."""

    dynamic_constraint_tightening: bool = True
    tightening_trigger_feasible_count: int | None = None
    tightening_keep_ratio: float = 0.3
    tightening_patience: int = 3
    tightening_improvement_tol: float = 1e-3
    tightening_max_tightens: int = 100
    tightening_min_ratio: float = 1e-3

    def __post_init__(self) -> None:
        super().__post_init__()
        trigger = self.tightening_trigger_feasible_count
        if trigger is not None and not 1 <= trigger <= self.population_size:
            raise ValueError(
                "tightening_trigger_feasible_count must be between 1 and "
                "population_size."
            )
        if not 0.0 <= self.tightening_keep_ratio <= 1.0:
            raise ValueError("tightening_keep_ratio must be in [0, 1].")
        if self.tightening_patience < 1:
            raise ValueError("tightening_patience must be positive.")
        if self.tightening_improvement_tol < 0.0:
            raise ValueError("tightening_improvement_tol must be non-negative.")
        if self.tightening_max_tightens < 1:
            raise ValueError("tightening_max_tightens must be positive.")
        if self.tightening_min_ratio < 0.0:
            raise ValueError("tightening_min_ratio must be non-negative.")
