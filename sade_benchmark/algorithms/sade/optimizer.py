"""A benchmark-oriented continuous SADE implementation.

The circuit-only classifier and simulator failure rules from ``SADE_TED`` are
intentionally absent. The retained core is differential-evolution candidate
generation, adaptive constraint penalties, cubic-RBF prescreening, and a small
local-search share in every expensive-evaluation batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .de import generate_trial_candidates_group
from .penalty import adaptive_penalty_fitness
from ...core.problem import FloatArray, InequalityProblem, as_2d_points
from ...core.result import OptimizationResult
from ...core.initialization import latin_hypercube_maximin
from .config import SADEConfig
from .sampling import distance_filter_log, sample_fun, supplement_inds
from .surrogate import CubicRBF, expected_improvement
from .tightening import ObjectiveConstraintTightener


class SADE:
    """Surrogate-assisted differential evolution for costly constrained problems."""

    def __init__(self, problem: InequalityProblem, config: SADEConfig | None = None):
        self.problem = problem
        self.config = config or SADEConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.lower = np.asarray(problem.lower_bounds, dtype=float).reshape(-1)
        self.upper = np.asarray(problem.upper_bounds, dtype=float).reshape(-1)
        self.dimension = int(problem.dimension)
        if self.lower.shape != (self.dimension,) or self.upper.shape != (self.dimension,):
            raise ValueError("Problem bounds must have shape (dimension,).")
        if np.any(~np.isfinite(self.lower)) or np.any(~np.isfinite(self.upper)):
            raise ValueError("All problem bounds must be finite.")
        if np.any(self.lower >= self.upper):
            raise ValueError("Each lower bound must be strictly below its upper bound.")
        if int(problem.n_constraints) < 1:
            raise ValueError("At least one inequality constraint is required.")
        self.span = self.upper - self.lower
        self.objective_tightener = ObjectiveConstraintTightener(self.config)

    def optimize(self) -> OptimizationResult:
        """Run one seeded optimization and return a feasible-first result."""

        archive_x = self._latin_hypercube(self.config.population_size)
        objective, constraints, violations = self._evaluate(archive_x)
        population = np.arange(len(archive_x), dtype=int)
        generation = 0
        history: list[dict[str, Any]] = []
        evaluation_metadata = [
            {
                "generation": 0,
                "source": "initial_lhs",
                "sampling_method": "latin_hypercube_maximin",
            }
            for _ in range(len(archive_x))
        ]
        population_history: list[dict[str, Any]] = []
        candidate_pools: list[dict[str, Any]] | None = (
            [] if self.config.save_candidate_pools else None
        )

        search_violations = self.objective_tightener.search_violations(
            objective, violations
        )
        archive_fitness, _, _, weights = adaptive_penalty_fitness(
            objective, search_violations, generation
        )
        history.append(
            self._history_entry(
                generation,
                objective,
                violations,
                weights,
                len(archive_x),
                sampling_method=None,
                population=population,
                local_radius=None,
                search_violations=search_violations,
                tightening_state=self.objective_tightener.state(),
                tightened=False,
            )
        )
        population_history.append(
            self._population_snapshot(
                generation,
                len(archive_x),
                population,
                archive_x,
                objective,
                violations,
            )
        )

        while len(archive_x) < self.config.max_evaluations:
            remaining = self.config.max_evaluations - len(archive_x)
            batch_size = min(self.config.batch_size, remaining)

            # As in SADE_TED, DE guides are ranked using scores normalized only
            # within the current population. The RBF target below is normalized
            # separately over the complete evaluated archive.
            search_violations = self.objective_tightener.search_violations(
                objective, violations
            )
            population_fitness, _, population_penalty, _ = adaptive_penalty_fitness(
                objective[population],
                search_violations[population],
                generation,
            )

            candidates = self._generate_candidate_pool(
                archive_x[population],
                objective[population],
                search_violations[population],
                population_fitness,
                population_penalty,
                generation,
            )
            selected, sampling_method, selected_metadata, pool_trace = self._select_candidates(
                candidates,
                batch_size,
                archive_x,
                archive_fitness,
                generation + 1,
            )
            new_objective, new_constraints, new_violations = self._evaluate(selected)
            evaluation_metadata.extend(selected_metadata)
            if candidate_pools is not None:
                candidate_pools.extend(pool_trace)

            first_new = len(archive_x)
            new_indices = np.arange(first_new, first_new + len(selected), dtype=int)
            archive_x = np.vstack((archive_x, selected))
            objective = np.concatenate((objective, new_objective))
            constraints = np.vstack((constraints, new_constraints))
            violations = np.vstack((violations, new_violations))

            search_violations = self.objective_tightener.search_violations(
                objective, violations
            )
            archive_fitness, _, _, weights = adaptive_penalty_fitness(
                objective, search_violations, generation
            )
            population = self._select_survivors(
                np.concatenate((population, new_indices)),
                objective,
                search_violations,
                generation,
            )
            tightened = self.objective_tightener.maybe_tighten(
                objective,
                violations,
                population,
                generation,
                archive_x,
            )
            if tightened:
                search_violations = self.objective_tightener.search_violations(
                    objective, violations
                )
                archive_fitness, _, _, weights = adaptive_penalty_fitness(
                    objective, search_violations, generation
                )
                population = self._select_survivors(
                    np.arange(len(archive_x), dtype=int),
                    objective,
                    search_violations,
                    generation,
                )
                event = self.objective_tightener.history[-1]
                event["selected_population_size"] = int(len(population))
                event["selected_search_feasible_count"] = int(
                    np.count_nonzero(
                        np.all(search_violations[population] <= 0.0, axis=1)
                    )
                )
            completed_generation = generation
            generation += 1
            history.append(
                self._history_entry(
                    generation,
                    objective,
                    violations,
                    weights,
                    len(archive_x),
                    sampling_method=sampling_method,
                    population=population,
                    local_radius=self._local_radius(completed_generation),
                    search_violations=search_violations,
                    tightening_state=self.objective_tightener.state(),
                    tightened=tightened,
                )
            )
            population_history.append(
                self._population_snapshot(
                    generation,
                    len(archive_x),
                    population,
                    archive_x,
                    objective,
                    violations,
                )
            )

        best = self._final_index(objective, violations)
        return OptimizationResult(
            x=archive_x[best].copy(),
            objective=float(objective[best]),
            constraints=constraints[best].copy(),
            violation=violations[best].copy(),
            feasible=bool(np.all(violations[best] <= 0.0)),
            evaluations=len(archive_x),
            generations=generation,
            history=history,
            archive_x=archive_x,
            archive_objective=objective,
            archive_constraints=constraints,
            archive_violation=violations,
            evaluation_metadata=evaluation_metadata,
            population_history=population_history,
            candidate_pools=candidate_pools,
            dynamic_constraint_state=self.objective_tightener.state(),
        )

    def _latin_hypercube(self, n_points: int) -> FloatArray:
        """Use exactly the same best-of-five maximin LHS as Python DSI."""

        return latin_hypercube_maximin(
                n_points, self.lower, self.upper, self.rng, attempts=5
            )
        # unit = np.empty((n_points, self.dimension), dtype=float)
        # for j in range(self.dimension):
        #     strata = (np.arange(n_points) + self.rng.random(n_points)) / n_points
        #     unit[:, j] = strata[self.rng.permutation(n_points)]
        # return self.lower + unit * self.span

    def _evaluate(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        points = as_2d_points(x, self.dimension)
        objective, constraints = self.problem.evaluate(points)
        objective = np.asarray(objective, dtype=float).reshape(-1)
        constraints = np.asarray(constraints, dtype=float)
        if constraints.ndim == 1:
            constraints = constraints.reshape(-1, 1)
        expected = (len(points), int(self.problem.n_constraints))
        if objective.shape != (len(points),) or constraints.shape != expected:
            raise ValueError(
                "Problem evaluator returned inconsistent shapes: "
                f"f={objective.shape}, g={constraints.shape}, expected g={expected}."
            )

        invalid = ~np.isfinite(objective) | ~np.all(np.isfinite(constraints), axis=1)
        objective = objective.copy()
        constraints = constraints.copy()
        objective[invalid] = np.inf
        constraints[invalid] = np.inf
        violations = np.maximum(
            constraints - self.config.constraint_tolerance,
            0.0,
        )
        return objective, constraints, violations

    def _select_survivors(
        self,
        indices: np.ndarray,
        objective: FloatArray,
        violations: FloatArray,
        generation: int,
    ) -> np.ndarray:
        indices = np.unique(indices)
        fitness, _, penalty, _ = adaptive_penalty_fitness(
            objective[indices],
            violations[indices],
            generation,
        )
        # Match SADE_Discrete: adaptive normalized violation is the primary
        # survivor key, with the complete penalized fitness as the tie-breaker.
        order = np.lexsort((fitness, penalty))
        return indices[order[: self.config.population_size]]

    def _generate_candidate_pool(
        self,
        population_x: FloatArray,
        population_objective: FloatArray,
        population_violations: FloatArray,
        population_fitness: FloatArray,
        population_penalty: FloatArray,
        generation: int,
    ) -> tuple[FloatArray, FloatArray]:
        """Create the original SADE global-DE and local-search candidate pools."""

        n_population = len(population_x)
        global_groups = [
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
            )
            for target_index in range(n_population)
        ]
        global_pool = np.vstack(global_groups)

        violation_count = np.sum(population_violations > 0.0, axis=1)
        pbest_order = np.lexsort((population_fitness, violation_count))
        pbest_count = max(1, int(self.config.p_best_fraction * n_population))
        if generation % 10 == 0:
            center_index = int(self.rng.integers(n_population))
        else:
            center_index = int(self.rng.choice(pbest_order[:pbest_count]))
        local_pool = self.local_search(
            population_x[center_index],
            generation,
            n_local=self.config.batch_size * n_population,
        )
        return global_pool, local_pool

    def local_search(
        self,
        best_vec: FloatArray,
        current_gen: int,
        *,
        n_local: int,
    ) -> FloatArray:
        """Continuous local search with SADE_Discrete's radius schedule."""

        radius_t = self._local_radius(current_gen)
        perturbation_range = self.span * radius_t
        n_change = max(1, int(radius_t * self.dimension))
        candidates = np.repeat(best_vec[None, :], n_local, axis=0)
        for candidate in candidates:
            changed = self.rng.choice(self.dimension, n_change, replace=False)
            noise = self.rng.uniform(-1.0, 1.0, n_change) * perturbation_range[changed]
            candidate[changed] = np.clip(
                candidate[changed] + noise,
                self.lower[changed],
                self.upper[changed],
            )
        return candidates

    def _local_radius(self, generation: int) -> float:
        remaining = self.config.max_evaluations - self.config.population_size
        total_generations = max(1, int(np.ceil(remaining / self.config.batch_size)))
        decay = (0.30 - 0.10) / total_generations
        return max(0.10, 0.30 - decay * generation)

    def _select_candidates(
        self,
        pools: tuple[FloatArray, FloatArray],
        batch_size: int,
        archive_x: FloatArray,
        archive_fitness: FloatArray,
        evaluation_generation: int,
    ) -> tuple[FloatArray, str, list[dict[str, Any]], list[dict[str, Any]]]:
        """Distance-filter candidates, then sample randomly or by EI."""

        global_pool, local_pool = pools
        para_rbf = self.train_rbf(archive_x, archive_fitness)

        local_count = int(round(batch_size * self.config.local_fraction))
        if self.config.local_fraction > 0.0 and batch_size > 1:
            local_count = max(1, local_count)
        local_count = min(local_count, batch_size)
        global_count = batch_size - local_count

        selected_global, global_sources = self._filter_and_sample_group(
            global_pool,
            global_count,
            archive_x,
            archive_fitness,
            para_rbf,
            source="global_de",
        )
        selected_local, local_sources = self._filter_and_sample_group(
            local_pool,
            local_count,
            archive_x,
            archive_fitness,
            para_rbf,
            source="local_search",
        )
        groups = [group for group in (selected_global, selected_local) if len(group)]
        selected = np.vstack(groups)
        sources = global_sources + local_sources
        completed, sources = self._complete_batch(
            selected,
            sources,
            global_pool,
            local_pool,
            batch_size,
        )
        method = "expected_improvement" if para_rbf is not None else "random"
        prediction = np.full(batch_size, np.nan)
        variance = np.full(batch_size, np.nan)
        ei = np.full(batch_size, np.nan)
        if para_rbf is not None:
            prediction, variance = para_rbf.predict_with_uncertainty(completed)
            y_min = float(np.min(archive_fitness[np.isfinite(archive_fitness)]))
            ei = -expected_improvement(completed, para_rbf, y_min)
        metadata = [
            {
                "generation": evaluation_generation,
                "source": sources[i],
                "sampling_method": method,
                "predicted_fitness": float(prediction[i]),
                "rbf_variance": float(variance[i]),
                "expected_improvement": float(ei[i]),
                "dynamic_objective_constraint_active": (
                    self.objective_tightener.threshold_active
                ),
                "dynamic_objective_limit": self.objective_tightener.active_limit,
                "dynamic_constraint_phase": self.objective_tightener.phase_id,
            }
            for i in range(batch_size)
        ]
        pool_trace: list[dict[str, Any]] = []
        if self.config.save_candidate_pools:
            for source, pool in (("global_de", global_pool), ("local_search", local_pool)):
                record: dict[str, Any] = {
                    "generation": evaluation_generation,
                    "stage": source,
                    "x": pool.copy(),
                }
                if para_rbf is not None:
                    predicted, pool_variance = para_rbf.predict_with_uncertainty(pool)
                    y_min = float(np.min(archive_fitness[np.isfinite(archive_fitness)]))
                    record["predicted_fitness"] = predicted
                    record["rbf_variance"] = pool_variance
                    record["expected_improvement"] = -expected_improvement(
                        pool, para_rbf, y_min
                    )
                pool_trace.append(record)
        return completed, method, metadata, pool_trace

    def _filter_and_sample_group(
        self,
        candidate_pool: FloatArray,
        n_samples: int,
        archive_x: FloatArray,
        archive_fitness: FloatArray,
        para_rbf: CubicRBF | None,
        *,
        source: str,
    ) -> tuple[FloatArray, list[str]]:
        if n_samples == 0:
            return np.empty((0, self.dimension), dtype=float), []
        filtered, _ = distance_filter_log(
            archive_x,
            candidate_pool,
            self.lower,
            self.upper,
            threshold=self.config.distance_threshold,
        )
        if len(filtered) < n_samples:
            selected = supplement_inds(filtered, candidate_pool, n_samples, self.rng)
            labels = [
                source
                if len(filtered) and np.any(np.all(filtered == point, axis=1))
                else f"{source}_shortage_fallback"
                for point in selected
            ]
            return selected, labels
        sampled = sample_fun(
            filtered,
            archive_fitness,
            para_rbf,
            self.rng,
            n_samples=n_samples,
        )
        if sampled is None:  # defensive; the size check above normally prevents this
            sampled = supplement_inds(filtered, candidate_pool, n_samples, self.rng)
            return sampled, [f"{source}_shortage_fallback"] * len(sampled)
        return sampled, [source] * len(sampled)

    def train_rbf(
        self,
        archive_x: FloatArray,
        archive_fitness: FloatArray,
    ) -> CubicRBF | None:
        """Train the cubic RBF after the configured number of finite samples."""

        valid = np.isfinite(archive_fitness)
        if np.count_nonzero(valid) < self.config.surrogate_min_samples:
            return None
        try:
            return CubicRBF(self.lower, self.upper).fit(
                archive_x[valid],
                archive_fitness[valid],
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None

    def _complete_batch(
        self,
        selected: FloatArray,
        sources: list[str],
        global_pool: FloatArray,
        local_pool: FloatArray,
        batch_size: int,
    ) -> tuple[FloatArray, list[str]]:
        """Guarantee an exact evaluation batch without duplicate selected rows."""

        _, unique_indices = np.unique(selected, axis=0, return_index=True)
        keep = np.sort(unique_indices)
        completed = selected[keep]
        completed_sources = [sources[int(index)] for index in keep]
        if len(completed) < batch_size:
            old_size = len(completed)
            completed = supplement_inds(
                completed,
                np.vstack((global_pool, local_pool)),
                batch_size,
                self.rng,
            )
            completed_sources.extend(
                ["batch_shortage_fallback"] * (len(completed) - old_size)
            )
        attempts = 0
        while len(completed) < batch_size and attempts < 20:
            old_size = len(completed)
            random_pool = self.lower + self.rng.random(
                (max(20, 4 * batch_size), self.dimension)
            ) * self.span
            completed = supplement_inds(
                completed,
                random_pool,
                batch_size,
                self.rng,
            )
            completed_sources.extend(
                ["random_domain_fallback"] * (len(completed) - old_size)
            )
            attempts += 1
        if len(completed) < batch_size:
            raise RuntimeError("Could not generate enough distinct candidates.")
        return completed[:batch_size], completed_sources[:batch_size]

    @staticmethod
    def _final_index(objective: FloatArray, violations: FloatArray) -> int:
        valid = np.isfinite(objective) & np.all(np.isfinite(violations), axis=1)
        feasible = valid & np.all(violations <= 0.0, axis=1)
        if np.any(feasible):
            feasible_indices = np.flatnonzero(feasible)
            return int(feasible_indices[np.argmin(objective[feasible_indices])])
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            raise RuntimeError("The evaluator did not return any finite point.")
        scale = np.maximum(np.max(violations[valid_indices], axis=0), 1e-12)
        total = np.sum(violations[valid_indices] / scale, axis=1)
        order = np.lexsort((objective[valid_indices], total))
        return int(valid_indices[order[0]])

    @staticmethod
    def _history_entry(
        generation: int,
        objective: FloatArray,
        original_violations: FloatArray,
        weights: FloatArray,
        evaluations: int,
        sampling_method: str | None,
        population: np.ndarray,
        local_radius: float | None,
        search_violations: FloatArray | None = None,
        tightening_state: dict[str, Any] | None = None,
        tightened: bool = False,
    ) -> dict[str, Any]:
        if search_violations is None:
            search_violations = original_violations
        feasible = np.all(original_violations <= 0.0, axis=1) & np.isfinite(objective)
        search_feasible = np.all(search_violations <= 0.0, axis=1) & np.isfinite(
            objective
        )
        valid = np.isfinite(objective) & np.all(
            np.isfinite(original_violations), axis=1
        )
        best_feasible = float(np.min(objective[feasible])) if np.any(feasible) else None
        minimum_violation = None
        if np.any(valid):
            minimum_violation = float(
                np.min(np.sum(original_violations[valid], axis=1))
            )
        entry = {
            "generation": generation,
            "evaluations": evaluations,
            "feasible_count": int(np.count_nonzero(feasible)),
            "best_feasible_objective": best_feasible,
            "minimum_total_violation": minimum_violation,
            "constraint_weights": weights.tolist(),
            "sampling_method": sampling_method,
            "rbf_active": sampling_method == "expected_improvement",
            "local_radius": local_radius,
            "population_feasible_count": int(np.count_nonzero(feasible[population])),
            "search_feasible_count": int(np.count_nonzero(search_feasible)),
            "population_search_feasible_count": int(
                np.count_nonzero(search_feasible[population])
            ),
            "dynamic_constraint_tightened": bool(tightened),
        }
        if tightening_state is None:
            entry.update(
                {
                    "dynamic_constraint_enabled": False,
                    "dynamic_constraint_active": False,
                    "dynamic_objective_limit": None,
                    "dynamic_constraint_phase": 0,
                    "dynamic_tighten_count": 0,
                    "dynamic_phase_progress_count": 0,
                    "dynamic_current_range_limit": None,
                }
            )
            return entry

        progress = tightening_state["current_phase_progress"]
        entry.update(
            {
                "dynamic_constraint_enabled": True,
                "dynamic_constraint_active": tightening_state["threshold_active"],
                "dynamic_objective_limit": tightening_state["active_limit"],
                "dynamic_constraint_phase": tightening_state["phase_id"],
                "dynamic_tighten_count": tightening_state["tighten_count"],
                "dynamic_phase_progress_count": len(progress),
                "dynamic_current_range_limit": (
                    progress[-1]["current_range_limit"] if progress else None
                ),
            }
        )
        return entry

    @staticmethod
    def _population_snapshot(
        generation: int,
        evaluations: int,
        population: np.ndarray,
        archive_x: FloatArray,
        objective: FloatArray,
        violations: FloatArray,
    ) -> dict[str, Any]:
        return {
            "generation": generation,
            "evaluations": evaluations,
            "archive_indices": population.copy(),
            "x": archive_x[population].copy(),
            "objective": objective[population].copy(),
            "total_violation": np.sum(violations[population], axis=1),
        }
