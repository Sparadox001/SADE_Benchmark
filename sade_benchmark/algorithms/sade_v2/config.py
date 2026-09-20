"""Configuration for the constraint-aware SADE V2.1 experiment."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade.config import SADEConfig


@dataclass(frozen=True)
class SADEV2Config(SADEConfig):
    """V2.1 defaults to a 30-point DOE and 30-sample RBF activation."""

    population_size: int = 30
    surrogate_min_samples: int = 30
    near_feasible_fraction: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.near_feasible_fraction <= 1.0:
            raise ValueError("near_feasible_fraction must be in (0, 1].")
