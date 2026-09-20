"""SADE V2.8 with local-neighbor reranking of a global-RBF shortlist."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2_7.optimizer import SADEV2_7
from ...core.problem import FloatArray, InequalityProblem
from .config import SADEV2_8Config


class SADEV2_8(SADEV2_7):
    """Use global RBF for a coarse shortlist and true neighbors to rerank it.

    Only the robust-feasible objective-stage choice changes. Feasibility
    search, continuous-CEI fallback, batch roles, and true-point ranking are
    inherited unchanged from V2.7.
    """

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_8Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_8Config())
        self.config: SADEV2_8Config
        self._objective_archive_x = np.empty((0, self.dimension), dtype=float)
        self._objective_archive_f = np.empty(0, dtype=float)
        self._scored_points = np.empty((0, self.dimension), dtype=float)

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
        """Expose the current true archive to the focused objective reranker."""

        valid = np.isfinite(objective) & np.all(np.isfinite(archive_x), axis=1)
        self._objective_archive_x = np.asarray(archive_x[valid], dtype=float)
        self._objective_archive_f = np.asarray(objective[valid], dtype=float)
        return super()._select_candidates_v2(
            pools,
            batch_size,
            archive_x,
            objective,
            constraints,
            violations,
            constraint_scales,
            evaluation_generation,
        )

    def _pool_scores(
        self,
        points: FloatArray,
        models: Any,
        constraint_scales: FloatArray,
        incumbent_normalized: float | None,
    ) -> dict[str, FloatArray]:
        """Remember the points belonging to the score dictionary being ranked."""

        scores = super()._pool_scores(
            points, models, constraint_scales, incumbent_normalized
        )
        self._scored_points = np.asarray(points, dtype=float)
        return scores

    def _local_neighbor_objective(self, points: FloatArray) -> FloatArray:
        """Return inverse-distance weighted objectives from true archive sites."""

        candidates = np.atleast_2d(np.asarray(points, dtype=float))
        if not len(self._objective_archive_x):
            return np.full(len(candidates), np.inf)
        archive = (self._objective_archive_x - self.lower) / self.span
        transformed = (candidates - self.lower) / self.span
        squared = np.sum(
            (transformed[:, None, :] - archive[None, :, :]) ** 2,
            axis=2,
        ) / self.dimension
        distances = np.sqrt(np.maximum(squared, 0.0))
        neighbors = min(self.config.objective_knn_neighbors, len(archive))
        indices = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
        local_distances = np.take_along_axis(distances, indices, axis=1)
        weights = 1.0 / np.maximum(local_distances, 1e-12) ** 2
        values = self._objective_archive_f[indices]
        return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)

    def _objective_candidate_index(
        self,
        scores: dict[str, Any],
        acquisition_cv: FloatArray,
    ) -> tuple[int, str]:
        """Rerank the global-RBF shortlist by a five-neighbor true estimate."""

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
        if len(feasible_indices) < required:
            return super()._objective_candidate_index(scores, acquisition_cv)

        predicted = np.nan_to_num(
            np.asarray(scores["predicted_objective"], dtype=float),
            nan=np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        )
        shortlist_size = min(
            len(feasible_indices),
            max(
                self.config.objective_shortlist_min_size,
                int(
                    np.ceil(
                        self.config.objective_shortlist_fraction
                        * len(feasible_indices)
                    )
                ),
            ),
        )
        predicted_order = np.argsort(
            predicted[feasible_indices], kind="stable"
        )
        shortlist = feasible_indices[predicted_order[:shortlist_size]]
        local_objective = self._local_neighbor_objective(
            self._scored_points[shortlist]
        )
        order = np.lexsort((predicted[shortlist], local_objective))
        return int(shortlist[order[0]]), "hard_feasible_knn_rerank"

    @staticmethod
    def _objective_acquisition_role(
        source: str, default_role: str, objective_rule: str
    ) -> str:
        if objective_rule == "hard_feasible_knn_rerank":
            prefix = "global" if source == "global_de" else "local"
            return f"{prefix}_robust_feasible_knn_rerank"
        return SADEV2_7._objective_acquisition_role(
            source, default_role, objective_rule
        )

    def _extra_acquisition_metadata(
        self, scores: dict[str, FloatArray], index: int
    ) -> dict[str, Any]:
        extra = super()._extra_acquisition_metadata(scores, index)
        if len(self._scored_points) == len(scores["predicted_objective"]):
            estimate = self._local_neighbor_objective(
                self._scored_points[index : index + 1]
            )[0]
            extra["local_neighbor_objective"] = float(estimate)
        return extra

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
        entry["objective_acquisition"] = (
            "global_rbf_shortlist_then_local_neighbor_objective"
        )
        return entry
