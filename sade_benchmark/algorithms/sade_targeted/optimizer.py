"""Continuous SADE with targeted tightening of a real objective constraint."""

from __future__ import annotations

import math

from ..sade.optimizer import SADE
from .config import SADETargetedConfig
from .tightening import TargetedObjectiveConstraintTightener


class SADETargeted(SADE):
    """Keep the fixed problem constraint; tighten only its search-time limit."""

    def __init__(self, problem, config: SADETargetedConfig | None = None):
        resolved = config or SADETargetedConfig()
        super().__init__(problem, resolved)
        base_limit = getattr(problem, "objective_cap", None)
        objective_index = getattr(problem, "objective_constraint_index", None)
        if base_limit is None or objective_index is None:
            raise ValueError(
                "SADETargeted requires a problem with objective_cap and "
                "objective_constraint_index metadata."
            )
        if not math.isfinite(float(base_limit)):
            raise ValueError("The objective cap must be finite.")
        requested_index = resolved.tightening_constraint_index
        count = int(problem.n_constraints)
        if not -count <= requested_index < count:
            raise ValueError(
                f"Constraint index {requested_index} is outside the {count} constraints."
            )
        index = requested_index % count
        if index != int(objective_index):
            raise ValueError(
                f"Constraint index {index} is not the objective-derived "
                f"constraint at index {objective_index}."
            )
        self.objective_tightener = TargetedObjectiveConstraintTightener(
            resolved, base_limit=float(base_limit), constraint_index=index
        )
