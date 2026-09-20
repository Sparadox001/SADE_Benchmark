"""SADE V2.9 with phase-dependent constraint acquisition."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2_7.optimizer import SADEV2_7
from ...core.problem import FloatArray, InequalityProblem
from ...core.result import OptimizationResult
from .config import SADEV2_9Config


class SADEV2_9(SADEV2_7):
    """Use raw predicted CV until feasibility, then retain robust P90 logic.

    V2.9 is intentionally branched from V2.7.  Candidate generation, the
    dynamic two-point batch, continuous CEI, population updates, and local
    search are unchanged.  Only the CV used to rank candidates before the
    first true feasible point changes.
    """

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_9Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_9Config())
        self.config: SADEV2_9Config
        self._raw_feasibility_acquisition = False

    def optimize(self) -> OptimizationResult:
        self._raw_feasibility_acquisition = False
        return super().optimize()

    def _acquisition_cv(self, scores: dict[str, Any]) -> FloatArray:
        """Return phase-appropriate CV without changing either estimator."""

        if self._raw_feasibility_acquisition:
            return np.asarray(scores["predicted_total_violation"], dtype=float)
        return super()._acquisition_cv(scores)

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
        """Select with raw CV only while no true feasible sample exists."""

        feasible = np.isfinite(objective) & np.all(violations <= 0.0, axis=1)
        use_raw = not np.any(feasible)
        self._raw_feasibility_acquisition = use_raw
        try:
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
        finally:
            self._raw_feasibility_acquisition = False

        selected, method, metadata, pools_trace, phase, diagnostics = result
        mode = "raw_predicted_cv" if use_raw else "robust_p90_cv"
        for row in metadata:
            row["constraint_acquisition_mode"] = mode
        diagnostics["constraint_acquisition_mode"] = mode
        return selected, method, metadata, pools_trace, phase, diagnostics

    @staticmethod
    def _v2_history_entry(
        generation: int,
        objective: FloatArray,
        violations: FloatArray,
        weights: FloatArray,
        evaluations: int,
        sampling_method: str | None,
        population: np.ndarray,
        local_radius: float | None,
        constraint_scales: FloatArray,
        rbf_active: bool,
        acquisition_phase: str,
        surrogate_diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        entry = SADEV2_7._v2_history_entry(
            generation,
            objective,
            violations,
            weights,
            evaluations,
            sampling_method,
            population,
            local_radius,
            constraint_scales,
            rbf_active,
            acquisition_phase,
            surrogate_diagnostics,
        )
        entry["constraint_acquisition"] = (
            "raw_predicted_cv_before_feasible_then_p90"
        )
        return entry
