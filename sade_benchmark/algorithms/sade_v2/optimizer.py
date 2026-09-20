"""SADE V2.1: constraint-aware two-stage surrogate prescreening."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade.optimizer import SADE
from ..sade.penalty import adaptive_penalty_fitness
from ..sade.sampling import distance_filter_log
from ...core.problem import FloatArray, InequalityProblem
from ...core.result import OptimizationResult
from ...core.initialization import latin_hypercube_maximin
from .config import SADEV2Config
from .surrogate import SADEV2Surrogates, standardized_expected_improvement


def _minimum_cv_indices(cv: FloatArray) -> np.ndarray:
    """Return every numerically tied minimum-CV candidate."""

    values = np.asarray(cv, dtype=float).reshape(-1)
    minimum = float(np.min(values))
    return np.flatnonzero(np.isclose(values, minimum, rtol=1e-12, atol=1e-15))


def _low_cv_shortlist(cv: FloatArray, fraction: float) -> np.ndarray:
    """Keep a rank fraction plus all candidates tied at its CV cutoff."""

    values = np.asarray(cv, dtype=float).reshape(-1)
    count = max(1, int(np.ceil(fraction * len(values))))
    order = np.argsort(values, kind="stable")
    cutoff = float(values[order[count - 1]])
    tolerance = max(1e-15, 1e-12 * abs(cutoff))
    return np.flatnonzero(values <= cutoff + tolerance)


class SADEV2(SADE):
    """V2.1 retains legacy DE/survival and replaces only surrogate acquisition."""

    def __init__(
        self, problem: InequalityProblem, config: SADEV2Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2Config())
        self.config: SADEV2Config

    def optimize(self) -> OptimizationResult:
        archive_x = self._latin_hypercube(self.config.population_size)
        objective, constraints, violations = self._evaluate(archive_x)
        constraint_scales = self._initial_constraint_scales(constraints)
        self.constraint_scales_ = constraint_scales.copy()
        population = np.arange(len(archive_x), dtype=int)
        generation = 0
        history: list[dict[str, Any]] = []
        evaluation_metadata = [
            {
                "generation": 0,
                "source": "initial_lhs_maximin",
                "sampling_method": "latin_hypercube_maximin",
                "acquisition_phase": "initialization",
                "acquisition_role": "initial_design",
            }
            for _ in range(len(archive_x))
        ]
        population_history: list[dict[str, Any]] = []
        candidate_pools: list[dict[str, Any]] | None = (
            [] if self.config.save_candidate_pools else None
        )

        _, _, weights = self._population_guide_scores(
            objective, violations, generation
        )
        initial_valid = np.isfinite(objective) & np.all(
            np.isfinite(constraints), axis=1
        )
        initial_unique = len(
            np.unique(np.round(archive_x[initial_valid], decimals=14), axis=0)
        )
        initial_surrogate_diagnostics = {
            "surrogate_train_size": int(np.count_nonzero(initial_valid)),
            "surrogate_unique_sites": initial_unique,
            "surrogate_required_samples": max(
                self.config.surrogate_min_samples, self.dimension + 1
            ),
            "rbf_system_size": None,
            "objective_rbf_rank": None,
            "constraint_rbf_rank": None,
            "rbf_condition_number": None,
            "rbf_inactive_reason": "initialization_not_fitted",
        }
        history.append(
            self._v2_history_entry(
                generation, objective, violations, weights, len(archive_x), None,
                population, None, constraint_scales, False, "initialization",
                initial_surrogate_diagnostics,
            )
        )
        population_history.append(
            self._population_snapshot(
                generation, len(archive_x), population, archive_x, objective, violations
            )
        )

        while len(archive_x) < self.config.max_evaluations:
            remaining = self.config.max_evaluations - len(archive_x)
            batch_size = min(self.config.batch_size, remaining)
            population_fitness, population_penalty, _ = self._population_guide_scores(
                objective[population], violations[population], generation
            )
            pools = self._generate_candidate_pool(
                archive_x[population], objective[population], violations[population],
                population_fitness, population_penalty, generation,
            )
            (
                selected,
                sampling_method,
                selected_metadata,
                pool_trace,
                sampling_phase,
                surrogate_diagnostics,
            ) = self._select_candidates_v2(
                pools, batch_size, archive_x, objective, constraints, violations,
                constraint_scales, generation + 1,
            )
            new_objective, new_constraints, new_violations = self._evaluate(selected)
            self._after_true_evaluation(
                selected_metadata, new_constraints, constraint_scales
            )
            self._after_expensive_batch(
                objective,
                violations,
                new_objective,
                new_violations,
                constraint_scales,
                selected_metadata,
            )
            evaluation_metadata.extend(selected_metadata)
            if candidate_pools is not None:
                candidate_pools.extend(pool_trace)

            first_new = len(archive_x)
            new_indices = np.arange(first_new, first_new + len(selected), dtype=int)
            archive_x = np.vstack((archive_x, selected))
            objective = np.concatenate((objective, new_objective))
            constraints = np.vstack((constraints, new_constraints))
            violations = np.vstack((violations, new_violations))

            # Full-archive legacy scores are not surrogate targets. This call is
            # retained only for the historical constraint-weight trace.
            _, _, weights = self._population_guide_scores(
                objective, violations, generation
            )
            population = self._update_population(
                population,
                new_indices,
                archive_x,
                objective,
                violations,
                generation,
            )
            completed_generation = generation
            generation += 1
            rbf_active = sampling_method == "two_stage_rbf"
            history.append(
                self._v2_history_entry(
                    generation, objective, violations, weights, len(archive_x),
                    sampling_method, population, self._local_radius(completed_generation),
                    constraint_scales, rbf_active, sampling_phase,
                    surrogate_diagnostics,
                )
            )
            population_history.append(
                self._population_snapshot(
                    generation, len(archive_x), population, archive_x, objective, violations
                )
            )

        best = self._final_index(objective, violations)
        return OptimizationResult(
            x=archive_x[best].copy(), objective=float(objective[best]),
            constraints=constraints[best].copy(), violation=violations[best].copy(),
            feasible=bool(np.all(violations[best] <= 0.0)), evaluations=len(archive_x),
            generations=generation, history=history, archive_x=archive_x,
            archive_objective=objective, archive_constraints=constraints,
            archive_violation=violations, evaluation_metadata=evaluation_metadata,
            population_history=population_history, candidate_pools=candidate_pools,
        )

    def _update_population(
        self,
        population: np.ndarray,
        new_indices: np.ndarray,
        archive_x: FloatArray,
        objective: FloatArray,
        violations: FloatArray,
        generation: int,
    ) -> np.ndarray:
        """Update the DE parent population after one expensive batch.

        The default hook preserves the historical global survivor truncation.
        Variants that need the evaluated coordinates can override this method
        without copying the optimization loop.
        """

        del archive_x
        return self._select_survivors(
            np.concatenate((population, new_indices)),
            objective,
            violations,
            generation,
        )

    def _population_guide_scores(
        self,
        objective: FloatArray,
        violations: FloatArray,
        generation: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return legacy fitness/penalty scores used by V2.1 population logic."""

        fitness, _, penalty, weights = adaptive_penalty_fitness(
            objective, violations, generation
        )
        return fitness, penalty, weights

    def _latin_hypercube(self, n_points: int) -> FloatArray:
        """Use exactly the same best-of-five maximin LHS as Python DSI."""

        return latin_hypercube_maximin(
            n_points, self.lower, self.upper, self.rng, attempts=5
        )

    def _initial_constraint_scales(self, constraints: FloatArray) -> FloatArray:
        """Freeze robust positive-violation scales from the complete initial DOE."""

        shifted = np.asarray(constraints, dtype=float) - self.config.constraint_tolerance
        scales = np.empty(shifted.shape[1], dtype=float)
        for column in range(shifted.shape[1]):
            finite = shifted[np.isfinite(shifted[:, column]), column]
            positive = finite[finite > 0.0]
            source = positive if len(positive) else np.abs(finite)
            scales[column] = np.percentile(source, 90.0) if len(source) else 1.0
        return np.maximum(scales, 1e-12)

    def _train_surrogates(
        self, archive_x: FloatArray, objective: FloatArray, constraints: FloatArray
    ) -> tuple[SADEV2Surrogates | None, dict[str, Any]]:
        valid = np.isfinite(objective) & np.all(np.isfinite(constraints), axis=1)
        valid_count = int(np.count_nonzero(valid))
        valid_x = archive_x[valid]
        unique_count = len(np.unique(np.round(valid_x, decimals=14), axis=0))
        required = max(self.config.surrogate_min_samples, self.dimension + 1)
        diagnostics: dict[str, Any] = {
            "surrogate_train_size": valid_count,
            "surrogate_unique_sites": unique_count,
            "surrogate_required_samples": required,
            "rbf_system_size": None,
            "objective_rbf_rank": None,
            "constraint_rbf_rank": None,
            "rbf_condition_number": None,
            "rbf_inactive_reason": None,
        }
        if unique_count < required:
            diagnostics["rbf_inactive_reason"] = "insufficient_unique_sites"
            return None, diagnostics
        try:
            model = SADEV2Surrogates(self.lower, self.upper).fit(
                valid_x, objective[valid], constraints[valid]
            )
            system_size = unique_count + self.dimension + 1
            diagnostics.update(
                rbf_system_size=system_size,
                objective_rbf_rank=int(model.objective.rank_),
                constraint_rbf_rank=int(model.constraints.rank_),
                rbf_condition_number=float(
                    max(
                        model.objective.condition_number_,
                        model.constraints.condition_number_,
                    )
                ),
            )
            if (
                model.objective.rank_ < system_size
                or model.constraints.rank_ < system_size
            ):
                diagnostics["rbf_inactive_reason"] = "rank_deficient_system"
                return None, diagnostics
            return model, diagnostics
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            diagnostics["rbf_inactive_reason"] = f"fit_failure:{type(error).__name__}"
            return None, diagnostics

    def _pool_scores(
        self,
        points: FloatArray,
        models: SADEV2Surrogates,
        constraint_scales: FloatArray,
        incumbent_normalized: float | None,
    ) -> dict[str, FloatArray]:
        predicted_objective = models.objective.predict(points)[:, 0]
        normalized_objective = models.objective.predict_normalized(points)[:, 0]
        predicted_constraints = models.constraints.predict(points)
        predicted_violations = np.maximum(
            predicted_constraints - self.config.constraint_tolerance, 0.0
        )
        predicted_cv = np.sum(predicted_violations / constraint_scales, axis=1)
        predicted_feasible = np.all(
            predicted_constraints <= self.config.constraint_tolerance, axis=1
        )
        uncertainty = models.objective.geometric_uncertainty(points)
        objective_ei = np.zeros(len(points), dtype=float)
        if incumbent_normalized is not None:
            objective_ei = standardized_expected_improvement(
                normalized_objective, incumbent_normalized, uncertainty
            )
        return {
            "predicted_objective": predicted_objective,
            "predicted_constraints": predicted_constraints,
            "predicted_total_violation": predicted_cv,
            "predicted_feasible": predicted_feasible,
            "objective_ei": objective_ei,
            "surrogate_uncertainty": uncertainty,
        }

    def _acquisition_cv(self, scores: dict[str, FloatArray]) -> FloatArray:
        """Return the CV used to rank unevaluated candidates."""

        return scores["predicted_total_violation"]

    def _acquisition_feasible(self, scores: dict[str, FloatArray]) -> FloatArray:
        """Return the predicted-feasible mask used by objective-stage EI."""

        return scores["predicted_feasible"]

    def _after_true_evaluation(
        self,
        metadata: list[dict[str, Any]],
        constraints: FloatArray,
        constraint_scales: FloatArray,
    ) -> None:
        """Hook for variants that learn from genuine out-of-sample errors."""

        del metadata, constraints, constraint_scales

    def _after_expensive_batch(
        self,
        objective: FloatArray,
        violations: FloatArray,
        new_objective: FloatArray,
        new_violations: FloatArray,
        constraint_scales: FloatArray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Hook for variants that react to true batch-level progress."""

        del (
            objective,
            violations,
            new_objective,
            new_violations,
            constraint_scales,
            metadata,
        )

    def _extra_acquisition_metadata(
        self, scores: dict[str, FloatArray], index: int
    ) -> dict[str, Any]:
        """Return variant-specific per-evaluation diagnostics."""

        del scores, index
        return {}

    def _complete_batch_v2(
        self,
        selected: FloatArray,
        sources: list[str],
        global_pool: FloatArray,
        local_pool: FloatArray,
        batch_size: int,
        archive_x: FloatArray,
    ) -> tuple[FloatArray, list[str]]:
        """Hook retaining V2.1's historical candidate-shortage behavior."""

        del archive_x
        return self._complete_batch(
            selected, sources, global_pool, local_pool, batch_size
        )

    def _objective_candidate_index(
        self,
        scores: dict[str, Any],
        acquisition_cv: FloatArray,
    ) -> tuple[int, str]:
        """Choose one objective-stage point and report the selection rule.

        Variants override this hook to change only the objective-stage
        acquisition rule while retaining the shared batch construction and
        logging path.
        """

        feasible_indices = np.flatnonzero(self._acquisition_feasible(scores))
        if len(feasible_indices):
            index = int(
                feasible_indices[
                    np.argmax(scores["objective_ei"][feasible_indices])
                ]
            )
            return index, "hard_feasible"

        minimum_cv = _minimum_cv_indices(acquisition_cv)
        index = int(
            minimum_cv[
                np.argmax(scores["surrogate_uncertainty"][minimum_cv])
            ]
        )
        return index, "fallback_min_cv"

    @staticmethod
    def _objective_acquisition_role(
        source: str, default_role: str, objective_rule: str
    ) -> str:
        """Return a truthful trace label for an objective-stage selection.

        Later variants can change the objective acquisition without copying the
        shared batch assembly logic. The default preserves historical labels.
        """

        del source
        if objective_rule == "hard_feasible":
            return default_role
        return f"{default_role}_{objective_rule}"

    def _batch_slots(self, phase: str, batch_size: int) -> list[tuple[str, str]]:
        """Return source and role for each requested expensive sample."""

        local_count = int(round(batch_size * self.config.local_fraction))
        if self.config.local_fraction > 0.0 and batch_size > 1:
            local_count = max(1, local_count)
        local_count = min(local_count, batch_size)
        global_count = batch_size - local_count
        slots: list[tuple[str, str]] = []
        for global_slot in range(global_count):
            role = (
                "global_min_cv"
                if phase == "feasibility"
                else "global_feasible_max_ei"
            )
            if global_slot > 0:
                role = "boundary_exploration"
            slots.append(("global_de", role))
        local_role = (
            "local_min_cv"
            if phase == "feasibility"
            else "local_feasible_max_ei"
        )
        slots.extend(("local_search", local_role) for _ in range(local_count))
        return slots

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
        """Select 2 global + 1 local points using the approved two-stage policy."""

        global_pool, local_pool = pools
        actual_feasible = np.isfinite(objective) & np.all(violations <= 0.0, axis=1)
        phase = "objective" if np.any(actual_feasible) else "feasibility"
        models, surrogate_diagnostics = self._train_surrogates(
            archive_x, objective, constraints
        )
        incumbent_normalized: float | None = None
        if models is not None and np.any(actual_feasible):
            incumbent = float(np.min(objective[actual_feasible]))
            incumbent_normalized = float(
                models.objective.normalize_outputs(np.asarray([incumbent]))[0, 0]
            )

        selected: list[FloatArray] = []
        sources: list[str] = []
        roles: list[str] = []

        def take(pool: FloatArray, source: str, role: str) -> None:
            reference = archive_x if not selected else np.vstack((archive_x, selected))
            filtered, _ = distance_filter_log(
                reference, pool, self.lower, self.upper,
                threshold=self.config.distance_threshold,
            )
            if not len(filtered):
                return
            if models is None:
                index = int(self.rng.integers(len(filtered)))
                chosen_role = f"{source}_random"
            else:
                scores = self._pool_scores(
                    filtered, models, constraint_scales, incumbent_normalized
                )
                cv = self._acquisition_cv(scores)
                if role == "boundary_exploration":
                    shortlist = _low_cv_shortlist(
                        cv, self.config.near_feasible_fraction
                    )
                    index = int(shortlist[np.argmax(
                        scores["surrogate_uncertainty"][shortlist]
                    )])
                    chosen_role = role
                elif phase == "objective":
                    index, objective_rule = self._objective_candidate_index(
                        scores, cv
                    )
                    chosen_role = self._objective_acquisition_role(
                        source, role, objective_rule
                    )
                else:
                    minimum_cv = _minimum_cv_indices(cv)
                    index = int(minimum_cv[np.argmax(
                        scores["surrogate_uncertainty"][minimum_cv]
                    )])
                    chosen_role = role
            selected.append(filtered[index].copy())
            sources.append(source)
            roles.append(chosen_role)

        for source, role in self._batch_slots(phase, batch_size):
            pool = global_pool if source == "global_de" else local_pool
            take(pool, source, role)

        selected_array = (
            np.vstack(selected) if selected else np.empty((0, self.dimension), dtype=float)
        )
        completed, completed_sources = self._complete_batch_v2(
            selected_array, sources, global_pool, local_pool, batch_size, archive_x
        )
        roles.extend(["candidate_shortage_fallback"] * (len(completed) - len(roles)))
        method = "two_stage_rbf" if models is not None else "random"

        metadata: list[dict[str, Any]] = []
        completed_scores = (
            self._pool_scores(completed, models, constraint_scales, incumbent_normalized)
            if models is not None else None
        )
        for index in range(batch_size):
            row: dict[str, Any] = {
                "generation": evaluation_generation,
                "source": completed_sources[index],
                "sampling_method": method,
                "acquisition_phase": phase,
                "acquisition_role": roles[index],
            }
            if completed_scores is not None:
                row.update(
                    predicted_objective=float(completed_scores["predicted_objective"][index]),
                    predicted_constraints=completed_scores["predicted_constraints"][index].tolist(),
                    predicted_total_violation=float(
                        completed_scores["predicted_total_violation"][index]
                    ),
                    predicted_feasible=bool(completed_scores["predicted_feasible"][index]),
                    objective_ei=float(completed_scores["objective_ei"][index]),
                    surrogate_uncertainty=float(
                        completed_scores["surrogate_uncertainty"][index]
                    ),
                )
                row.update(self._extra_acquisition_metadata(completed_scores, index))
            metadata.append(row)

        pool_trace: list[dict[str, Any]] = []
        if self.config.save_candidate_pools:
            for source, pool in (("global_de", global_pool), ("local_search", local_pool)):
                record: dict[str, Any] = {
                    "generation": evaluation_generation, "stage": source,
                    "acquisition_phase": phase, "x": pool.copy(),
                }
                if models is not None:
                    record.update(
                        self._pool_scores(pool, models, constraint_scales, incumbent_normalized)
                    )
                pool_trace.append(record)
        return completed, method, metadata, pool_trace, phase, surrogate_diagnostics

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
        base = SADE._history_entry(
            generation, objective, violations, weights, evaluations,
            sampling_method, population, local_radius,
        )
        base.update(
            rbf_active=rbf_active,
            acquisition_phase=acquisition_phase,
            constraint_scales=constraint_scales.tolist(),
        )
        base.update(surrogate_diagnostics)
        return base
