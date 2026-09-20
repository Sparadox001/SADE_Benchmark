"""SADE V2.11 with feasibility-stage nearest-neighbor survival."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2_7.optimizer import SADEV2_7
from ...core.problem import FloatArray, InequalityProblem
from ...core.result import OptimizationResult
from .config import SADEV2_11Config


class SADEV2_11(SADEV2_7):
    """Preserve parent niches until the first true feasible point appears."""

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_11Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_11Config())
        self.config: SADEV2_11Config
        self._population_survival_mode = "initial_population"
        self._nearest_candidates_processed = 0
        self._nearest_replacements = 0

    def optimize(self) -> OptimizationResult:
        self._population_survival_mode = "initial_population"
        self._nearest_candidates_processed = 0
        self._nearest_replacements = 0
        return super().optimize()

    def _update_population(
        self,
        population: np.ndarray,
        new_indices: np.ndarray,
        archive_x: FloatArray,
        objective: FloatArray,
        violations: FloatArray,
        generation: int,
    ) -> np.ndarray:
        """Use nearest-neighbor competition only before true feasibility."""

        feasible = np.isfinite(objective) & np.all(violations <= 0.0, axis=1)
        self._nearest_candidates_processed = 0
        self._nearest_replacements = 0
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

        self._population_survival_mode = "nearest_neighbor_infeasible"
        updated = np.asarray(population, dtype=int).copy()
        new_values = np.asarray(new_indices, dtype=int)
        new_order = self._feasible_first_order(
            objective[new_values], violations[new_values]
        )
        normalized = (archive_x - self.lower) / self.span

        for new_index in new_values[new_order]:
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
        entry["nearest_candidates_processed"] = (
            self._nearest_candidates_processed
        )
        entry["nearest_replacements"] = self._nearest_replacements
        return entry
