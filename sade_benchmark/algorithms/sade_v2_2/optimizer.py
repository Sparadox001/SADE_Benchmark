"""SADE V2.2 with one feasible-first rule throughout population evolution."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade.de import generate_trial_candidates_group
from ..sade.sampling import distance_filter_log
from ..sade_v2.optimizer import SADEV2
from ...core.problem import FloatArray, InequalityProblem
from .config import SADEV2_2Config


class SADEV2_2(SADEV2):
    """Unify true-point ranking while retaining V2.1 surrogate acquisition."""

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_2Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_2Config())
        self.config: SADEV2_2Config

    def _fixed_total_violation(self, violations: FloatArray) -> FloatArray:
        values = np.asarray(violations, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        valid = np.all(np.isfinite(values), axis=1)
        total = np.full(len(values), np.inf)
        total[valid] = np.sum(values[valid] / self.constraint_scales_, axis=1)
        return total

    def _feasible_first_order(
        self, objective: FloatArray, violations: FloatArray
    ) -> np.ndarray:
        objective_values = np.asarray(objective, dtype=float).reshape(-1)
        total_violation = self._fixed_total_violation(violations)
        valid = np.isfinite(objective_values) & np.isfinite(total_violation)
        feasible = valid & (total_violation <= 0.0)
        objective_key = np.where(np.isfinite(objective_values), objective_values, np.inf)
        # Primary: feasible before infeasible. Secondary: fixed-scale CV.
        # Tertiary: objective, which ranks feasible points and breaks CV ties.
        return np.lexsort((objective_key, total_violation, (~feasible).astype(int)))

    def _population_guide_scores(
        self,
        objective: FloatArray,
        violations: FloatArray,
        generation: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        del generation
        order = self._feasible_first_order(objective, violations)
        rank_score = np.empty(len(order), dtype=float)
        rank_score[order] = np.arange(len(order), dtype=float)
        if len(order) > 1:
            rank_score /= len(order) - 1
        total_violation = self._fixed_total_violation(violations)
        return rank_score, total_violation, 1.0 / self.constraint_scales_

    def _select_survivors(
        self,
        indices: np.ndarray,
        objective: FloatArray,
        violations: FloatArray,
        generation: int,
    ) -> np.ndarray:
        del generation
        candidates = np.unique(indices)
        order = self._feasible_first_order(
            objective[candidates], violations[candidates]
        )
        return candidates[order[: self.config.population_size]]

    def _final_index(
        self, objective: FloatArray, violations: FloatArray
    ) -> int:
        """Apply the same fixed-scale feasible-first rule to the final archive."""

        objective_values = np.asarray(objective, dtype=float).reshape(-1)
        total_violation = self._fixed_total_violation(violations)
        valid = np.isfinite(objective_values) & np.isfinite(total_violation)
        if not np.any(valid):
            raise RuntimeError("The evaluator did not return any finite point.")
        return int(self._feasible_first_order(objective, violations)[0])

    def _complete_batch_v2(
        self,
        selected: FloatArray,
        sources: list[str],
        global_pool: FloatArray,
        local_pool: FloatArray,
        batch_size: int,
        archive_x: FloatArray,
    ) -> tuple[FloatArray, list[str]]:
        """Fill shortages while preserving archive and within-batch distance."""

        selected_values = np.asarray(selected, dtype=float).reshape(-1, self.dimension)
        _, unique_indices = np.unique(selected_values, axis=0, return_index=True)
        keep = np.sort(unique_indices)
        completed = selected_values[keep]
        completed_sources = [sources[int(index)] for index in keep]
        candidate_pool = np.vstack((global_pool, local_pool))

        while len(completed) < batch_size:
            reference = (
                archive_x
                if not len(completed)
                else np.vstack((archive_x, completed))
            )
            filtered, _ = distance_filter_log(
                reference,
                candidate_pool,
                self.lower,
                self.upper,
                threshold=self.config.distance_threshold,
            )
            if not len(filtered):
                break
            chosen = filtered[int(self.rng.integers(len(filtered)))].copy()
            completed = (
                chosen[None, :]
                if not len(completed)
                else np.vstack((completed, chosen))
            )
            completed_sources.append("batch_shortage_distance_fallback")

        attempts = 0
        while len(completed) < batch_size and attempts < 20:
            reference = (
                archive_x
                if not len(completed)
                else np.vstack((archive_x, completed))
            )
            random_pool = self.lower + self.rng.random(
                (max(20, 4 * batch_size), self.dimension)
            ) * self.span
            filtered, _ = distance_filter_log(
                reference,
                random_pool,
                self.lower,
                self.upper,
                threshold=self.config.distance_threshold,
            )
            if len(filtered):
                chosen = filtered[int(self.rng.integers(len(filtered)))].copy()
                completed = (
                    chosen[None, :]
                    if not len(completed)
                    else np.vstack((completed, chosen))
                )
                completed_sources.append("random_domain_distance_fallback")
            attempts += 1
        if len(completed) < batch_size:
            raise RuntimeError(
                "Could not generate enough candidates satisfying the distance threshold."
            )
        return completed[:batch_size], completed_sources[:batch_size]

    def _generate_candidate_pool(
        self,
        population_x: FloatArray,
        population_objective: FloatArray,
        population_violations: FloatArray,
        population_fitness: FloatArray,
        population_penalty: FloatArray,
        generation: int,
    ) -> tuple[FloatArray, FloatArray]:
        """Generate original DE trials using V2.2's unified guide order."""

        n_population = len(population_x)
        unified_order = self._feasible_first_order(
            population_objective, population_violations
        )
        global_pool = np.vstack(
            [
                generate_trial_candidates_group(
                    population_x,
                    target_index,
                    population_objective,
                    population_penalty,
                    population_fitness,
                    population_violations,
                    self.lower,
                    self.upper,
                    self.rng,
                    num_trials=self.config.trials_per_target,
                    p_pbest=self.config.p_best_fraction,
                    unified_order=unified_order,
                )
                for target_index in range(n_population)
            ]
        )

        feasible = np.all(population_violations <= 0.0, axis=1) & np.isfinite(
            population_objective
        )
        if np.any(feasible):
            # unified_order[0] is the best true feasible objective.
            center_index = int(unified_order[0])
        else:
            pbest_count = max(
                1, int(self.config.p_best_fraction * n_population)
            )
            center_index = int(self.rng.choice(unified_order[:pbest_count]))
        local_pool = self.local_search(
            population_x[center_index],
            generation,
            n_local=self.config.batch_size * n_population,
        )
        return global_pool, local_pool

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
        entry = SADEV2._v2_history_entry(
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
        entry["population_ranking"] = "fixed_scale_feasible_first"
        return entry
