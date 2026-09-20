"""SADE V2.7 with stagnation-triggered boundary exploration."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2_6.optimizer import SADEV2_6
from ...core.problem import FloatArray, InequalityProblem
from ...core.result import OptimizationResult
from .config import SADEV2_7Config


class SADEV2_7(SADEV2_6):
    """Switch the second two-point-batch slot only after true stagnation."""

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_7Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_7Config())
        self.config: SADEV2_7Config
        self._stagnant_batches = 0
        self._selection_stagnant_batches = 0
        self._selection_used_boundary = False

    def optimize(self) -> OptimizationResult:
        self._stagnant_batches = 0
        self._selection_stagnant_batches = 0
        self._selection_used_boundary = False
        return super().optimize()

    def _batch_slots(self, phase: str, batch_size: int) -> list[tuple[str, str]]:
        """Keep global slot one and switch the second slot when stagnant."""

        if batch_size != 2:
            self._selection_stagnant_batches = self._stagnant_batches
            self._selection_used_boundary = False
            return super()._batch_slots(phase, batch_size)

        global_role = (
            "global_min_cv"
            if phase == "feasibility"
            else "global_feasible_max_ei"
        )
        local_role = (
            "local_min_cv"
            if phase == "feasibility"
            else "local_feasible_max_ei"
        )
        use_boundary = (
            self._stagnant_batches >= self.config.boundary_stagnation_batches
        )
        self._selection_stagnant_batches = self._stagnant_batches
        self._selection_used_boundary = use_boundary
        second = (
            ("global_de", "boundary_exploration")
            if use_boundary
            else ("local_search", local_role)
        )
        return [("global_de", global_role), second]

    def _after_expensive_batch(
        self,
        objective: FloatArray,
        violations: FloatArray,
        new_objective: FloatArray,
        new_violations: FloatArray,
        constraint_scales: FloatArray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Update stagnation from the same true feasible-first comparison."""

        old_feasible = np.isfinite(objective) & np.all(violations <= 0.0, axis=1)
        new_feasible = np.isfinite(new_objective) & np.all(
            new_violations <= 0.0, axis=1
        )
        if np.any(old_feasible):
            incumbent = float(np.min(objective[old_feasible]))
            improved = bool(
                np.any(new_feasible & (new_objective < incumbent))
            )
        elif np.any(new_feasible):
            improved = True
        else:
            old_cv = np.sum(violations / constraint_scales, axis=1)
            new_cv = np.sum(new_violations / constraint_scales, axis=1)
            improved = bool(np.min(new_cv) < np.min(old_cv))

        used_boundary = any(
            row.get("acquisition_role") == "boundary_exploration"
            for row in metadata
        )
        self._stagnant_batches = (
            0 if improved or used_boundary else self._stagnant_batches + 1
        )

    def _select_candidates_v2(
        self,
        pools: tuple[FloatArray, FloatArray],
        batch_size: int,
        archive_x: FloatArray,
        objective: FloatArray,
        constraints: FloatArray,
        violations: FloatArray,
        constraint_scales: FloatArray,
        evaluation_generation: int,
    ) -> tuple[
        FloatArray,
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
        str,
        dict[str, Any],
    ]:
        """Add dynamic-allocation state to the normal V2.5 trace."""

        result = super()._select_candidates_v2(
            pools,
            batch_size,
            archive_x,
            objective,
            constraints,
            violations,
            constraint_scales,
            evaluation_generation,
        )
        selected, method, metadata, pools_trace, phase, diagnostics = result
        state = {
            "stagnant_batches_before_selection": (
                self._selection_stagnant_batches
            ),
            "dynamic_boundary_triggered": self._selection_used_boundary,
            "boundary_stagnation_batches": (
                self.config.boundary_stagnation_batches
            ),
        }
        for row in metadata:
            row.update(state)
        diagnostics.update(state)
        return selected, method, metadata, pools_trace, phase, diagnostics
