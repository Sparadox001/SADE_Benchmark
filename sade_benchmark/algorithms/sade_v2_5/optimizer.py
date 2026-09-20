"""SADE V2.5 with minimum predicted objective on the robust feasible set."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2_4.optimizer import SADEV2_4
from ...core.problem import FloatArray, InequalityProblem
from .config import SADEV2_5Config


class SADEV2_5(SADEV2_4):
    """Replace V2.4's hard-feasible maximum EI with minimum RBF mean.

    If too few candidates pass the robust P90 feasibility gate, V2.4's
    continuous CEI fallback is retained. Feasibility-stage selection and the
    boundary-exploration slot are unchanged, keeping this a focused ablation.
    """

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_5Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_5Config())
        self.config: SADEV2_5Config

    def _objective_candidate_index(
        self,
        scores: dict[str, Any],
        acquisition_cv: FloatArray,
    ) -> tuple[int, str]:
        """Minimize predicted objective when robust-feasible supply is enough."""

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
            predicted = np.nan_to_num(
                np.asarray(scores["predicted_objective"], dtype=float),
                nan=np.inf,
                posinf=np.inf,
                neginf=-np.inf,
            )
            uncertainty = np.nan_to_num(
                np.asarray(scores["surrogate_uncertainty"], dtype=float),
                nan=np.inf,
                posinf=np.inf,
                neginf=0.0,
            )
            order = np.lexsort(
                (uncertainty[feasible_indices], predicted[feasible_indices])
            )
            return int(feasible_indices[order[0]]), "hard_feasible"

        return super()._objective_candidate_index(scores, acquisition_cv)

    @staticmethod
    def _objective_acquisition_role(
        source: str, default_role: str, objective_rule: str
    ) -> str:
        del default_role
        prefix = "global" if source == "global_de" else "local"
        if objective_rule == "hard_feasible":
            return f"{prefix}_robust_feasible_min_predicted_objective"
        if objective_rule == "continuous_cei":
            return f"{prefix}_continuous_cei_fallback"
        return f"{prefix}_{objective_rule}"

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
        entry = SADEV2_4._v2_history_entry(
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
        entry["objective_acquisition"] = (
            "min_predicted_objective_with_continuous_cei_fallback"
        )
        return entry

