"""DSI with SADE's dynamic objective constraint as the only search change."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.distance import pairwise_distance
from ...core.problem import FloatArray, InequalityProblem
from ...core.result import OptimizationResult
from ..dsi.optimizer import DSI, _Population
from ..dsi.surrogate import DSICubicRBF
from .config import DSIDynamicConfig
from .tightening import DSIObjectiveConstraintTightener


class DSIDynamic(DSI):
    """Preserve DSI and append the active objective limit to its search CV."""

    def __init__(
        self,
        problem: InequalityProblem,
        config: DSIDynamicConfig | None = None,
    ):
        resolved = config or DSIDynamicConfig()
        super().__init__(problem, resolved)
        self.config = resolved
        self.objective_tightener = DSIObjectiveConstraintTightener(resolved)

    def optimize(self) -> OptimizationResult:
        initial_x = self._latin_hypercube_maximin(self.config.population_size)
        initial_f, initial_g, initial_v = self._evaluate(initial_x)
        initial_physical_cv = np.sum(initial_v, axis=1)
        initial_search_cv = self.objective_tightener.search_total_violation(
            initial_f, initial_physical_cv
        )
        train = _Population(initial_x, initial_f, initial_g, initial_search_cv)
        population = self._update_a1(train, self.config.population_size)

        history = [self._dynamic_history_entry(0, train, population, self.config.wmax)]
        population_history = [
            self._dynamic_population_snapshot(0, len(train.x), population, train)
        ]
        evaluation_metadata: list[dict[str, Any]] = [
            {
                "generation": 0,
                "source": "initial_lhs_maximin",
                "sampling_method": "latin_hypercube_maximin",
                "dynamic_objective_constraint_active": False,
                "dynamic_objective_limit": None,
                "dynamic_constraint_phase": 0,
            }
            for _ in range(len(initial_x))
        ]
        candidate_pools: list[dict[str, Any]] | None = (
            [] if self.config.save_candidate_pools else None
        )

        obj_delete = np.empty((0, self.dimension))
        con_delete_feasible_as_infeasible = np.empty((0, self.dimension))
        con_delete_infeasible_as_feasible = np.empty((0, self.dimension))
        current_wmax = int(self.config.wmax)
        generation = 0

        while len(train.x) < self.config.max_evaluations:
            train_x, train_f, train_g = self._unique_training_data(train)
            joint_model = DSICubicRBF().fit(
                train_x, np.column_stack((train_f, train_g))
            )

            obj_delete = self._unique_rows(obj_delete)
            objective_x, objective_y = self._remove_training_rows(
                train_x, train_f[:, None], obj_delete
            )
            objective_model = DSICubicRBF().fit(objective_x, objective_y)

            con_delete = self._unique_rows(
                np.vstack(
                    (
                        con_delete_feasible_as_infeasible,
                        con_delete_infeasible_as_feasible,
                    )
                )
            )
            constraint_x, constraint_y = self._remove_training_rows(
                train_x, train_g, con_delete
            )
            constraint_model = DSICubicRBF().fit(constraint_x, constraint_y)

            generation += 1
            generated, half, generated_pools = self._generator_c2ode_dynamic(
                population,
                objective_model,
                constraint_model,
                current_wmax,
                generation,
            )
            if candidate_pools is not None:
                candidate_pools.extend(generated_pools)

            first_index, _ = self._best_population_index(generated)
            second_index, _ = self._best_population_index(half)
            offspring01 = generated.x[first_index].copy()
            offspring02 = half.x[second_index].copy()
            selected = np.vstack((offspring01, offspring02))
            selected_source = ["c2ode_final", "c2ode_half"]
            selected_predicted_f = np.array(
                [generated.objective[first_index], half.objective[second_index]]
            )
            generated_physical_cv = self._predicted_physical_cv(generated.constraints)
            half_physical_cv = self._predicted_physical_cv(half.constraints)
            selected_predicted_cv = np.array(
                [generated_physical_cv[first_index], half_physical_cv[second_index]]
            )
            selected_predicted_search_cv = np.array(
                [generated.total_violation[first_index], half.total_violation[second_index]]
            )
            predicted_first_physical_feasible = generated_physical_cv[first_index] == 0.0
            selected_sigma = np.full(2, np.nan)
            use_adaptation = True

            if np.min(pairwise_distance(offspring01[None, :], train.x)) <= 1e-11:
                joint_prediction, sigma = joint_model.predict_with_uncertainty(half.x)
                predicted_f = joint_prediction[:, 0]
                predicted_physical_cv = self._predicted_physical_cv(
                    joint_prediction[:, 1:]
                )
                predicted_search_cv = self.objective_tightener.search_total_violation(
                    predicted_f, predicted_physical_cv
                )
                feasible_indices = np.flatnonzero(predicted_search_cv == 0.0)
                if len(feasible_indices):
                    chosen = int(feasible_indices[np.argmax(sigma[feasible_indices])])
                else:
                    chosen = int(np.argmax(sigma))
                selected = half.x[chosen][None, :]
                selected_source = ["uncertainty_fallback"]
                selected_predicted_f = predicted_f[chosen : chosen + 1]
                selected_predicted_cv = predicted_physical_cv[chosen : chosen + 1]
                selected_predicted_search_cv = predicted_search_cv[chosen : chosen + 1]
                selected_sigma = sigma[chosen : chosen + 1]
                use_adaptation = False

            remaining = self.config.max_evaluations - len(train.x)
            if len(selected) > remaining:
                selected = selected[:remaining]
                selected_source = selected_source[:remaining]
                selected_predicted_f = selected_predicted_f[:remaining]
                selected_predicted_cv = selected_predicted_cv[:remaining]
                selected_predicted_search_cv = selected_predicted_search_cv[:remaining]
                selected_sigma = selected_sigma[:remaining]
                use_adaptation = False

            active_during_selection = self.objective_tightener.threshold_active
            limit_during_selection = self.objective_tightener.active_limit
            phase_during_selection = self.objective_tightener.phase_id
            pair_duplicate = len(selected) == 2 and np.array_equal(selected[0], selected[1])
            evaluation_x = selected[:1] if pair_duplicate else selected
            real_f, real_g, real_v = self._evaluate(evaluation_x)
            real_physical_cv = np.sum(real_v, axis=1)
            real_search_cv = self.objective_tightener.search_total_violation(
                real_f, real_physical_cv
            )
            off = _Population(evaluation_x, real_f, real_g, real_search_cv)

            population = self._update_a1(
                self._concatenate_population(population, off),
                self.config.population_size,
            )
            train = self._concatenate_population(train, off)

            for i in range(len(evaluation_x)):
                evaluation_metadata.append(
                    {
                        "generation": generation,
                        "source": selected_source[i],
                        "sampling_method": "dsi_c2ode",
                        "predicted_objective": float(selected_predicted_f[i]),
                        "predicted_total_violation": float(selected_predicted_cv[i]),
                        "predicted_search_total_violation": float(
                            selected_predicted_search_cv[i]
                        ),
                        "rbf_variance": float(selected_sigma[i]),
                        "wmax": current_wmax,
                        "dynamic_objective_constraint_active": active_during_selection,
                        "dynamic_objective_limit": limit_during_selection,
                        "dynamic_constraint_phase": phase_during_selection,
                    }
                )

            if use_adaptation and len(selected) == 2:
                if pair_duplicate:
                    comparison_f = np.repeat(real_f, 2)
                    comparison_cv = np.repeat(real_search_cv, 2)
                else:
                    comparison_f = real_f
                    comparison_cv = real_search_cv
                if comparison_cv[0] > comparison_cv[1]:
                    current_wmax *= 0.5
                elif comparison_cv[0] == comparison_cv[1]:
                    if comparison_f[0] >= comparison_f[1]:
                        current_wmax *= 0.5
                        worst = int(np.argmax(objective_y[:, 0]))
                        obj_delete = np.vstack((obj_delete, objective_x[worst]))
                    else:
                        current_wmax *= 2.0
                else:
                    current_wmax *= 2.0

                # Preserve DSI's constraint-model correction semantics: only
                # original physical feasibility participates in this feedback.
                actual_first_physical_feasible = real_physical_cv[0] == 0.0
                if (
                    actual_first_physical_feasible
                    != predicted_first_physical_feasible
                ):
                    minimum, maximum = constraint_model.influence_indices(offspring01)
                    if actual_first_physical_feasible:
                        con_delete_feasible_as_infeasible = np.vstack(
                            (
                                con_delete_feasible_as_infeasible,
                                constraint_x[maximum[0]],
                            )
                        )
                        con_delete_infeasible_as_feasible = self._drop_random_row(
                            con_delete_infeasible_as_feasible
                        )
                    else:
                        con_delete_infeasible_as_feasible = np.vstack(
                            (
                                con_delete_infeasible_as_feasible,
                                constraint_x[minimum[0]],
                            )
                        )
                        con_delete_feasible_as_infeasible = self._drop_random_row(
                            con_delete_feasible_as_infeasible
                        )

            current_wmax = int(np.clip(current_wmax, 5, 80))
            tightened = self.objective_tightener.maybe_tighten(
                train.x,
                train.objective,
                train.total_violation,
                population.objective,
                population.total_violation,
                generation,
            )
            if tightened:
                physical_cv = self._physical_cv(train.constraints)
                train.total_violation = self.objective_tightener.search_total_violation(
                    train.objective, physical_cv
                )
                population = self._update_a1(train, self.config.population_size)
                event = self.objective_tightener.history[-1]
                event["selected_population_size"] = int(len(population.x))
                event["selected_search_feasible_count"] = int(
                    np.count_nonzero(population.total_violation == 0.0)
                )

            history.append(
                self._dynamic_history_entry(generation, train, population, current_wmax)
            )
            population_history.append(
                self._dynamic_population_snapshot(
                    generation, len(train.x), population, train
                )
            )

        archive_violation = np.maximum(
            train.constraints - self.config.constraint_tolerance, 0.0
        )
        best = self._final_index(train.objective, archive_violation)
        return OptimizationResult(
            x=train.x[best].copy(),
            objective=float(train.objective[best]),
            constraints=train.constraints[best].copy(),
            violation=archive_violation[best].copy(),
            feasible=bool(np.all(archive_violation[best] <= 0.0)),
            evaluations=len(train.x),
            generations=generation,
            history=history,
            archive_x=train.x,
            archive_objective=train.objective,
            archive_constraints=train.constraints,
            archive_violation=archive_violation,
            evaluation_metadata=evaluation_metadata,
            population_history=population_history,
            candidate_pools=candidate_pools,
            dynamic_constraint_state=self.objective_tightener.state(),
        )

    def _generator_c2ode_dynamic(
        self,
        population: _Population,
        objective_model: DSICubicRBF,
        constraint_model: DSICubicRBF,
        wmax: int,
        outer_generation: int,
    ) -> tuple[_Population, _Population, list[dict[str, Any]]]:
        current = population.copy()
        var0 = float(np.max(current.total_violation))
        cp = ((-np.log(var0) - 6.0) / np.log(0.5)) if var0 > 0.0 else 0.0
        half: _Population | None = None
        pool_trace: list[dict[str, Any]] = []

        for inner_generation in range(1, wmax + 1):
            progress = (inner_generation - 1) / wmax
            epsilon = var0 * (1.0 - progress) ** cp if progress < 0.5 else 0.0

            if np.std(current.total_violation) < 1e-8 and not np.any(
                current.total_violation == 0.0
            ):
                current.x = self.lower + self.rng.random(current.x.shape) * (
                    self.upper - self.lower
                )
                current.objective = objective_model.predict(current.x)[:, 0]
                current.constraints = constraint_model.predict(current.x)
                physical_cv = self._predicted_physical_cv(current.constraints)
                current.total_violation = (
                    self.objective_tightener.search_total_violation(
                        current.objective, physical_cv
                    )
                )

            trial = self._generate_three_trials(current)
            predicted_f = objective_model.predict(trial)[:, 0]
            predicted_g = constraint_model.predict(trial)
            predicted_physical_cv = self._predicted_physical_cv(predicted_g)
            predicted_search_cv = self.objective_tightener.search_total_violation(
                predicted_f, predicted_physical_cv
            )

            if self.config.save_candidate_pools:
                pool_trace.append(
                    {
                        "generation": outer_generation,
                        "inner_generation": inner_generation,
                        "stage": "c2ode_three_trials",
                        "x": trial.copy(),
                        "predicted_objective": predicted_f.copy(),
                        "predicted_total_violation": predicted_physical_cv.copy(),
                        "predicted_search_total_violation": predicted_search_cv.copy(),
                        "dynamic_objective_constraint_active": (
                            self.objective_tightener.threshold_active
                        ),
                        "dynamic_objective_limit": self.objective_tightener.active_limit,
                    }
                )

            chosen = self._preselect_three(predicted_f, predicted_search_cv)
            selected_x = trial[chosen]
            selected_f = predicted_f[chosen]
            selected_g = predicted_g[chosen]
            selected_cv = predicted_search_cv[chosen]
            replace = self._epsilon_replacement_mask(
                current.objective,
                current.total_violation,
                selected_f,
                selected_cv,
                epsilon,
            )
            current.x[replace] = selected_x[replace]
            current.objective[replace] = selected_f[replace]
            current.constraints[replace] = selected_g[replace]
            current.total_violation[replace] = selected_cv[replace]

            if inner_generation == int(np.ceil(wmax / 2.0)):
                half = current.copy()

        if half is None:
            half = current.copy()
        return current, half, pool_trace

    def _physical_cv(self, constraints: FloatArray) -> FloatArray:
        violations = np.maximum(
            np.asarray(constraints, dtype=float) - self.config.constraint_tolerance,
            0.0,
        )
        return np.sum(violations, axis=1)

    @staticmethod
    def _predicted_physical_cv(constraints: FloatArray) -> FloatArray:
        return np.sum(np.maximum(np.asarray(constraints, dtype=float), 0.0), axis=1)

    def _dynamic_history_entry(
        self,
        generation: int,
        train: _Population,
        population: _Population,
        wmax: int,
    ) -> dict[str, Any]:
        archive_physical_cv = self._physical_cv(train.constraints)
        population_physical_cv = self._physical_cv(population.constraints)
        physical_feasible = archive_physical_cv == 0.0
        best = (
            float(np.min(train.objective[physical_feasible]))
            if np.any(physical_feasible)
            else None
        )
        state = self.objective_tightener.state()
        return {
            "generation": generation,
            "evaluations": len(train.x),
            "feasible_count": int(np.count_nonzero(physical_feasible)),
            "population_feasible_count": int(
                np.count_nonzero(population_physical_cv == 0.0)
            ),
            "best_feasible_objective": best,
            "minimum_total_violation": float(np.min(archive_physical_cv)),
            "search_feasible_count": int(
                np.count_nonzero(train.total_violation == 0.0)
            ),
            "population_search_feasible_count": int(
                np.count_nonzero(population.total_violation == 0.0)
            ),
            "sampling_method": "dsi_c2ode" if generation else None,
            "rbf_active": generation > 0,
            "wmax": wmax,
            "dynamic_constraint_tightened": bool(
                state and state["history"] and state["history"][-1]["generation"] == generation
            ),
            "dynamic_constraint_enabled": bool(state and state["enabled"]),
            "dynamic_constraint_active": bool(state and state["threshold_active"]),
            "dynamic_objective_limit": None if state is None else state["active_limit"],
            "dynamic_constraint_phase": 0 if state is None else state["phase_id"],
            "dynamic_tighten_count": 0 if state is None else state["tighten_count"],
        }

    def _dynamic_population_snapshot(
        self,
        generation: int,
        evaluations: int,
        population: _Population,
        train: _Population,
    ) -> dict[str, Any]:
        snapshot = self._population_snapshot(generation, evaluations, population, train)
        snapshot["search_total_violation"] = population.total_violation.copy()
        snapshot["total_violation"] = self._physical_cv(population.constraints)
        return snapshot
