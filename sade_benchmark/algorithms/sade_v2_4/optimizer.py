"""SADE V2.4 with hybrid hard/continuous feasibility acquisition."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2.surrogate import SADEV2Surrogates
from ..sade_v2_3.optimizer import SADEV2_3
from ...core.problem import FloatArray, InequalityProblem
from ...core.result import OptimizationResult
from .config import SADEV2_4Config


class SADEV2_4(SADEV2_3):
    """Use continuous constrained EI when the conservative set is sparse.

    V2.3's hard P90 gate remains the preferred objective-stage rule.  When
    too few candidates pass that gate, signed out-of-sample constraint
    residual vectors define empirical joint feasibility scenarios instead of
    falling back to minimum predicted constraint violation.
    """

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_4Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_4Config())
        self.config: SADEV2_4Config
        self._signed_constraint_residuals: list[FloatArray] = []

    def optimize(self) -> OptimizationResult:
        self._signed_constraint_residuals = []
        return super().optimize()

    def _recent_signed_residuals(self) -> FloatArray | None:
        """Return the recent joint residual scenarios once calibration is ready."""

        if len(self._signed_constraint_residuals) < (
            self.config.constraint_error_min_samples
        ):
            return None
        return np.asarray(
            self._signed_constraint_residuals[
                -self.config.constraint_error_window :
            ],
            dtype=float,
        )

    def _continuous_feasibility_scores(
        self,
        predicted_constraints: FloatArray,
        predicted_cv: FloatArray,
        constraint_scales: FloatArray,
        objective_ei: FloatArray,
    ) -> dict[str, Any]:
        """Estimate joint feasibility and combine it continuously with EI."""

        residuals = self._recent_signed_residuals()
        n_candidates = len(predicted_constraints)
        if residuals is None:
            probability = np.full(n_candidates, np.nan, dtype=float)
            expected_cv = np.asarray(predicted_cv, dtype=float).copy()
            weight = 1.0 / (1.0 + expected_cv)
            sample_count = 0
            active = False
        else:
            normalized_prediction = (
                np.asarray(predicted_constraints, dtype=float)
                - self.config.constraint_tolerance
            ) / constraint_scales
            scenario_constraints = (
                normalized_prediction[:, None, :] + residuals[None, :, :]
            )
            scenario_cv = np.sum(
                np.maximum(scenario_constraints, 0.0), axis=2
            )
            jointly_feasible = np.all(scenario_constraints <= 0.0, axis=2)
            sample_count = len(residuals)
            probability = (
                1.0 + np.count_nonzero(jointly_feasible, axis=1)
            ) / (sample_count + 2.0)
            expected_cv = np.mean(scenario_cv, axis=1)
            weight = probability / (1.0 + expected_cv)
            active = True

        constrained_ei = np.asarray(objective_ei, dtype=float) * weight
        return {
            "empirical_joint_feasibility_probability": probability,
            "expected_constraint_violation": expected_cv,
            "continuous_feasibility_weight": weight,
            "constrained_expected_improvement": constrained_ei,
            "constraint_residual_samples": sample_count,
            "continuous_feasibility_active": active,
        }

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
        scores.update(
            self._continuous_feasibility_scores(
                scores["predicted_constraints"],
                scores["predicted_total_violation"],
                constraint_scales,
                scores["objective_ei"],
            )
        )
        return scores

    def _objective_candidate_index(
        self,
        scores: dict[str, Any],
        acquisition_cv: FloatArray,
    ) -> tuple[int, str]:
        """Prefer hard P90 EI, then use continuous constrained EI."""

        del acquisition_cv
        robust_feasible = np.asarray(
            scores["robust_predicted_feasible"], dtype=bool
        )
        feasible_indices = np.flatnonzero(robust_feasible)
        required = max(
            1,
            int(
                np.ceil(
                    self.config.hard_feasible_min_fraction
                    * len(robust_feasible)
                )
            ),
        )
        if len(feasible_indices) >= required:
            index = int(
                feasible_indices[
                    np.argmax(scores["objective_ei"][feasible_indices])
                ]
            )
            return index, "hard_feasible"

        constrained_ei = np.nan_to_num(
            np.asarray(scores["constrained_expected_improvement"], dtype=float),
            nan=-np.inf,
            posinf=np.finfo(float).max,
            neginf=-np.inf,
        )
        weight = np.nan_to_num(
            np.asarray(scores["continuous_feasibility_weight"], dtype=float),
            nan=-np.inf,
            posinf=np.finfo(float).max,
            neginf=-np.inf,
        )
        uncertainty = np.nan_to_num(
            np.asarray(scores["surrogate_uncertainty"], dtype=float),
            nan=-np.inf,
            posinf=np.finfo(float).max,
            neginf=-np.inf,
        )
        order = np.lexsort((-uncertainty, -weight, -constrained_ei))
        return int(order[0]), "continuous_cei"

    def _after_true_evaluation(
        self,
        metadata: list[dict[str, Any]],
        constraints: FloatArray,
        constraint_scales: FloatArray,
    ) -> None:
        """Retain signed joint residuals as well as V2.3's P90 errors."""

        super()._after_true_evaluation(metadata, constraints, constraint_scales)
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
            self._signed_constraint_residuals.append(
                (true_values - predicted_values) / constraint_scales
            )

    def _extra_acquisition_metadata(
        self, scores: dict[str, Any], index: int
    ) -> dict[str, Any]:
        metadata = super()._extra_acquisition_metadata(scores, index)
        probability = scores["empirical_joint_feasibility_probability"][index]
        metadata.update(
            empirical_joint_feasibility_probability=(
                None if not np.isfinite(probability) else float(probability)
            ),
            expected_constraint_violation=float(
                scores["expected_constraint_violation"][index]
            ),
            continuous_feasibility_weight=float(
                scores["continuous_feasibility_weight"][index]
            ),
            constrained_expected_improvement=float(
                scores["constrained_expected_improvement"][index]
            ),
            constraint_residual_samples=int(
                scores["constraint_residual_samples"]
            ),
            continuous_feasibility_active=bool(
                scores["continuous_feasibility_active"]
            ),
        )
        return metadata

    def _train_surrogates(
        self, archive_x: FloatArray, objective: FloatArray, constraints: FloatArray
    ) -> tuple[SADEV2Surrogates | None, dict[str, Any]]:
        model, diagnostics = super()._train_surrogates(
            archive_x, objective, constraints
        )
        residuals = self._recent_signed_residuals()
        diagnostics.update(
            signed_constraint_residual_samples=len(
                self._signed_constraint_residuals
            ),
            continuous_feasibility_active=residuals is not None,
            hard_feasible_min_fraction=(
                self.config.hard_feasible_min_fraction
            ),
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
        entry = SADEV2_3._v2_history_entry(
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
        entry["constraint_acquisition"] = "hybrid_p90_continuous_cei"
        return entry
