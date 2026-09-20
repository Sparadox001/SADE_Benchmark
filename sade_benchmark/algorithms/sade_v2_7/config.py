"""Configuration for SADE V2.7 dynamic two-point batches."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_6.config import SADEV2_6Config


@dataclass(frozen=True)
class SADEV2_7Config(SADEV2_6Config):
    """V2.6 settings plus a true-progress boundary trigger."""

    boundary_stagnation_batches: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.boundary_stagnation_batches < 1:
            raise ValueError("boundary_stagnation_batches must be positive.")
