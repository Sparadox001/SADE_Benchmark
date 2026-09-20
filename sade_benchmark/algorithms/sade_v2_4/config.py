"""Configuration for SADE V2.4 hybrid feasibility acquisition."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_3.config import SADEV2_3Config


@dataclass(frozen=True)
class SADEV2_4Config(SADEV2_3Config):
    """V2.3 plus a continuous fallback for sparse hard-feasible pools."""

    hard_feasible_min_fraction: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.hard_feasible_min_fraction <= 1.0:
            raise ValueError("hard_feasible_min_fraction must be in [0, 1].")
