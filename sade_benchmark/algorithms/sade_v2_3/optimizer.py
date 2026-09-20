"""SADE V2.3 with calibrated conservative constraint acquisition."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2.surrogate import SADEV2Surrogates
from ..sade_v2_2.optimizer import SADEV2_2
from ...core.problem import FloatArray, InequalityProblem
from ...core.result import OptimizationResult
from .config import SADEV2_3Config


class SADEV2_3(SADEV2_2):
    """Calibrate false-feasible RBF errors without changing V2.2 evolution."""

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_3Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_3Config())
        self.config: SADEV2_3Config
        self._constraint_underprediction_errors: list[FloatArray] = []

    def optimize(self) -> OptimizationResult:
        self._constraint_underprediction_errors = []
        return super().optimize()

    def _constraint_error_margin(self) -> tuple[FloatArray, bool]:
        """Return recent Pq positive normalized underprediction errors."""

        count = len(self._constraint_underprediction_errors)
        if count < self.config.constraint_error_min_samples:
            return np.zeros(int(self.problem.n_constraints), dtype=float), False
        recent = np.asarray(
            self._constraint_underprediction_errors[
                -self.config.constraint_error_window :
            ],
            dtype=float,
        )
        margin = np.quantile(
            recent, self.config.constraint_error_quantile, axis=0
        )
        return np.maximum(margin, 0.0), True

    def _robust_constraint_scores(
        self,
        predicted_constraints: FloatArray,
        constraint_scales: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, bool]:
        """Build conservative normalized constraints, CV, and feasibility."""

        predicted = np.asarray(predicted_constraints, dtype=float)
        margin, active = self._constraint_error_margin()
        normalized = (
            predicted - self.config.constraint_tolerance
        ) / constraint_scales
        conservative = normalized + self.config.constraint_safety_factor * margin
        robust_cv = np.sum(np.maximum(conservative, 0.0), axis=1)
        robust_feasible = np.all(conservative <= 0.0, axis=1)
        return robust_cv, robust_feasible, margin, active

    def _pool_scores(
        self,
        points: FloatArray,
        models: SADEV2Surrogates,
        constraint_scales: FloatArray,
        incumbent_normalized: float | None,
    ) -> dict[str, Any]:
        scores: dict[str, Any] = super()._pool_scores(
            points, models, constraint_scales, incumbent_normalized
        )
        robust_cv, robust_feasible, margin, active = (
            self._robust_constraint_scores(
                scores["predicted_constraints"], constraint_scales
            )
        )
        scores.update(
            robust_predicted_total_violation=robust_cv,
            robust_predicted_feasible=robust_feasible,
            constraint_error_quantiles=margin,
            constraint_calibration_active=active,
        )
        return scores

    def _acquisition_cv(self, scores: dict[str, Any]) -> FloatArray:
        return np.asarray(scores["robust_predicted_total_violation"], dtype=float)

    def _acquisition_feasible(self, scores: dict[str, Any]) -> FloatArray:
        return np.asarray(scores["robust_predicted_feasible"], dtype=bool)

    def _after_true_evaluation(
        self,
        metadata: list[dict[str, Any]],
        constraints: FloatArray,
        constraint_scales: FloatArray,
    ) -> None:
        """Accumulate only predictions made before their expensive evaluation."""

        for row, true_constraints in zip(metadata, constraints, strict=True):
            predicted = row.get("predicted_constraints")
            if predicted is None:
                continue
            predicted_values = np.asarray(predicted, dtype=float)
            true_values = np.asarray(true_constraints, dtype=float)
            if not (
                np.all(np.isfinite(predicted_values))
                and np.all(np.isfinite(true_values))
            ):
                continue
            underprediction = np.maximum(
                (true_values - predicted_values) / constraint_scales, 0.0
            )
            self._constraint_underprediction_errors.append(underprediction)

    def _extra_acquisition_metadata(
        self, scores: dict[str, Any], index: int
    ) -> dict[str, Any]:
        return {
            "robust_predicted_total_violation": float(
                scores["robust_predicted_total_violation"][index]
            ),
            "robust_predicted_feasible": bool(
                scores["robust_predicted_feasible"][index]
            ),
            "constraint_error_quantiles": np.asarray(
                scores["constraint_error_quantiles"], dtype=float
            ).tolist(),
            "constraint_calibration_active": bool(
                scores["constraint_calibration_active"]
            ),
        }

    def _train_surrogates(
        self, archive_x: FloatArray, objective: FloatArray, constraints: FloatArray
    ) -> tuple[SADEV2Surrogates | None, dict[str, Any]]:
        model, diagnostics = super()._train_surrogates(
            archive_x, objective, constraints
        )
        margin, active = self._constraint_error_margin()
        diagnostics.update(
            constraint_error_samples_used=len(
                self._constraint_underprediction_errors
            ),
            constraint_error_quantiles=margin.tolist(),
            constraint_calibration_active=active,
            constraint_error_window=self.config.constraint_error_window,
        )
        return model, diagnostics

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
        entry = SADEV2_2._v2_history_entry(
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
        entry["constraint_acquisition"] = "online_one_sided_p90"
        return entry
