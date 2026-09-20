"""SADE V2.12 with hybrid feasibility-stage population survival."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2_7.optimizer import SADEV2_7
from ...core.problem import FloatArray, InequalityProblem
from ...core.result import OptimizationResult
from .config import SADEV2_12Config


class SADEV2_12(SADEV2_7):
    """Apply global and nearest-neighbor survival to their batch roles."""

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_12Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_12Config())
        self.config: SADEV2_12Config
        self._population_survival_mode = "initial_population"
        self._global_candidates_processed = 0
        self._global_candidates_admitted = 0
        self._nearest_candidates_processed = 0
        self._nearest_replacements = 0

    def optimize(self) -> OptimizationResult:
        self._population_survival_mode = "initial_population"
        self._reset_survival_diagnostics()
        return super().optimize()

    def _reset_survival_diagnostics(self) -> None:
        self._global_candidates_processed = 0
        self._global_candidates_admitted = 0
        self._nearest_candidates_processed = 0
        self._nearest_replacements = 0

    def _update_population(
        self,
        population: np.ndarray,
        new_indices: np.ndarray,
        archive_x: FloatArray,
        objective: FloatArray,
        violations: FloatArray,
        generation: int,
    ) -> np.ndarray:
        """Use one global and one nearest-neighbor survivor competition.

        V2.7 always emits the global slot first.  The optional second slot is
        either the local-search point or the stagnation-triggered boundary
        point.  Once any truly feasible point exists, both slots return to
        V2.7's global feasible-first truncation immediately.
        """

        self._reset_survival_diagnostics()
        feasible = np.isfinite(objective) & np.all(
            violations <= 0.0, axis=1
        )
        if np.any(feasible):
            self._population_survival_mode = "global_feasible_first"
            return super()._update_population(
                population,
                new_indices,
                archive_x,
                objective,
                violations,
                generation,
            )

        self._population_survival_mode = "hybrid_infeasible"
        updated = np.asarray(population, dtype=int).copy()
        new_values = np.asarray(new_indices, dtype=int)
        if len(new_values) == 0:
            return updated

        global_index = int(new_values[0])
        self._global_candidates_processed = 1
        updated = self._select_survivors(
            np.concatenate((updated, [global_index])),
            objective,
            violations,
            generation,
        )
        self._global_candidates_admitted = int(global_index in updated)

        if len(new_values) == 1:
            return updated

        normalized = (archive_x - self.lower) / self.span
        for new_index in new_values[1:]:
            distances = np.sqrt(
                np.mean(
                    (normalized[updated] - normalized[new_index]) ** 2,
                    axis=1,
                )
            )
            nearest_position = int(np.argmin(distances))
            incumbent_index = int(updated[nearest_position])
            pair = np.asarray([incumbent_index, int(new_index)], dtype=int)
            pair_order = self._feasible_first_order(
                objective[pair], violations[pair]
            )
            self._nearest_candidates_processed += 1
            if int(pair_order[0]) == 1:
                updated[nearest_position] = int(new_index)
                self._nearest_replacements += 1

        return updated

    def _v2_history_entry(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        entry = super()._v2_history_entry(*args, **kwargs)
        entry["population_survival"] = self._population_survival_mode
        entry["global_candidates_processed"] = (
            self._global_candidates_processed
        )
        entry["global_candidates_admitted"] = self._global_candidates_admitted
        entry["nearest_candidates_processed"] = (
            self._nearest_candidates_processed
        )
        entry["nearest_replacements"] = self._nearest_replacements
        return entry
