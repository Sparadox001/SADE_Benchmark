"""Replace one existing objective constraint in SADE's search state."""

from __future__ import annotations

import numpy as np

from ...core.problem import FloatArray
from ..sade.tightening import ObjectiveConstraintTightener
from .config import SADETargetedConfig


class TargetedObjectiveConstraintTightener(ObjectiveConstraintTightener):
    """Search with ``f <= active_limit`` in the existing constraint slot."""

    def __init__(
        self,
        config: SADETargetedConfig,
        *,
        base_limit: float,
        constraint_index: int,
    ) -> None:
        super().__init__(config)
        self.base_limit = float(base_limit)
        self.constraint_index = int(constraint_index)
        self.constraint_tolerance = float(config.constraint_tolerance)
        self.active_limit = self.base_limit

    def search_violations(
        self,
        objective: FloatArray,
        original_violations: FloatArray,
    ) -> FloatArray:
        original = np.asarray(original_violations, dtype=float)
        values = np.asarray(objective, dtype=float).reshape(-1)
        if original.ndim != 2 or original.shape[0] != len(values):
            raise ValueError("Search violations and objectives have incompatible shapes.")
        search = original.copy()
        search[:, self.constraint_index] = np.maximum(
            values - float(self.active_limit) - self.constraint_tolerance, 0.0
        )
        search[~np.isfinite(values), self.constraint_index] = np.inf
        return search

    def state(self) -> dict | None:
        state = super().state()
        if state is None:
            return None
        state.update(
            {
                "virtual_constraint": None,
                "constraint_mode": "replace_existing_objective_constraint",
                "target_constraint_index": self.constraint_index,
                "base_limit": self.base_limit,
                "active_limit": self.active_limit,
            }
        )
        return state
