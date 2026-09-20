"""SADE V2.13 with a feasibility-first objective fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2_7.optimizer import SADEV2_7
from ...core.problem import InequalityProblem
from .config import SADEV2_13Config


class SADEV2_13(SADEV2_7):
    """Replace only V2.7's continuous-CEI objective fallback."""

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_13Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_13Config())
        self.config: SADEV2_13Config

    def _objective_candidate_index(
        self,
        scores: dict[str, Any],
        acquisition_cv: np.ndarray,
    ) -> tuple[int, str]:
        """Keep the P90 gate, then rank by joint feasibility probability."""

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
            return super()._objective_candidate_index(scores, acquisition_cv)

        probability = np.asarray(
            scores["empirical_joint_feasibility_probability"], dtype=float
        )
        finite = np.isfinite(probability)
        if np.any(finite):
            best_probability = float(np.max(probability[finite]))
            shortlist = np.flatnonzero(
                finite
                & np.isclose(
                    probability,
                    best_probability,
                    rtol=1e-12,
                    atol=1e-15,
                )
            )
            predicted = np.asarray(
                scores["predicted_objective"], dtype=float
            )
            uncertainty = np.asarray(
                scores["surrogate_uncertainty"], dtype=float
            )
            order = np.lexsort(
                (uncertainty[shortlist], predicted[shortlist])
            )
            return int(shortlist[order[0]]), "probability_feasible_fallback"

        raw_feasible = np.asarray(scores["predicted_feasible"], dtype=bool)
        raw_indices = np.flatnonzero(raw_feasible)
        predicted = np.asarray(scores["predicted_objective"], dtype=float)
        uncertainty = np.asarray(
            scores["surrogate_uncertainty"], dtype=float
        )
        if len(raw_indices):
            order = np.lexsort(
                (uncertainty[raw_indices], predicted[raw_indices])
            )
            return int(raw_indices[order[0]]), "raw_feasible_fallback"

        raw_cv = np.asarray(scores["predicted_total_violation"], dtype=float)
        minimum = float(np.min(raw_cv))
        tied = np.flatnonzero(
            np.isclose(raw_cv, minimum, rtol=1e-12, atol=1e-15)
        )
        index = int(tied[np.argmax(uncertainty[tied])])
        return index, "raw_min_cv_fallback"
