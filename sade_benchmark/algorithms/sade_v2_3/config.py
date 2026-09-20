"""Configuration for SADE V2.3 constraint-error calibration."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_2.config import SADEV2_2Config


@dataclass(frozen=True)
class SADEV2_3Config(SADEV2_2Config):
    """V2.2 plus online one-sided constraint-error calibration."""

    constraint_error_quantile: float = 0.9
    constraint_error_window: int = 30
    constraint_error_min_samples: int = 12
    constraint_safety_factor: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.constraint_error_quantile <= 1.0:
            raise ValueError("constraint_error_quantile must be in (0, 1].")
        if self.constraint_error_min_samples < 1:
            raise ValueError("constraint_error_min_samples must be positive.")
        if self.constraint_error_window < self.constraint_error_min_samples:
            raise ValueError(
                "constraint_error_window must be at least constraint_error_min_samples."
            )
        if self.constraint_safety_factor < 0.0:
            raise ValueError("constraint_safety_factor cannot be negative.")
