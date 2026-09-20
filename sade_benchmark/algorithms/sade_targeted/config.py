"""Configuration for tightening one existing objective-derived constraint."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade.config import SADEConfig


@dataclass(frozen=True)
class SADETargetedConfig(SADEConfig):
    dynamic_constraint_tightening: bool = True
    tightening_constraint_index: int = -1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.dynamic_constraint_tightening:
            raise ValueError("SADETargeted requires dynamic_constraint_tightening=True.")
        if isinstance(self.tightening_constraint_index, bool) or not isinstance(
            self.tightening_constraint_index, int
        ):
            raise ValueError("tightening_constraint_index must be an integer.")
