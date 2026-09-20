"""Python port of the local MATLAB DSI_ECOP implementation.

The algorithmic choices follow ``code_local/DSI``: full-domain maximin LHS,
separate cubic RBF models after data selection, the embedded C2oDE generator,
two real evaluations per outer iteration when possible, adaptive ``wmax``, and
the objective/constraint training-set deletion rules from the paper code.
Only numerical safeguards and exact evaluation-budget handling are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.distance import pairwise_distance
from ...core.initialization import latin_hypercube_maximin
from ...core.problem import FloatArray, InequalityProblem, as_2d_points
from ...core.result import OptimizationResult
from .config import DSIConfig
from .surrogate import DSICubicRBF


@dataclass
class _Population:
    x: FloatArray
    objective: FloatArray
    constraints: FloatArray
    total_violation: FloatArray

    def copy(self) -> "_Population":
        return _Population(
            self.x.copy(),
            self.objective.copy(),
            self.constraints.copy(),
            self.total_violation.copy(),
        )


class DSI:
    """Data-selection-based surrogate-assisted constrained optimizer."""

    def __init__(self, problem: InequalityProblem, config: DSIConfig | None = None):
        self.problem = problem
        self.config = config or DSIConfig()
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

    def optimize(self) -> OptimizationResult:
        initial_x = self._latin_hypercube_maximin(self.config.population_size)
        initial_f, initial_g, initial_v = self._evaluate(initial_x)
        train = _Population(initial_x, initial_f, initial_g, np.sum(initial_v, axis=1))
        population = self._update_a1(train, self.config.population_size)

        history = [self._history_entry(0, train, population, self.config.wmax)]
        population_history = [self._population_snapshot(0, len(train.x), population, train)]
        evaluation_metadata: list[dict[str, Any]] = [
            {
                "generation": 0,
                "source": "initial_lhs_maximin",
                "sampling_method": "latin_hypercube_maximin",
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
            generated, half, generated_pools = self._generator_c2ode(
                population,
                objective_model,
                constraint_model,
                current_wmax,
                generation,
            )
            if candidate_pools is not None:
                candidate_pools.extend(generated_pools)

            first_index, predicted_feasible = self._best_population_index(generated)
            second_index, _ = self._best_population_index(half)
            offspring01 = generated.x[first_index].copy()
            offspring02 = half.x[second_index].copy()
            selected = np.vstack((offspring01, offspring02))
            selected_source = ["c2ode_final", "c2ode_half"]
            selected_predicted_f = np.array(
                [generated.objective[first_index], half.objective[second_index]]
            )
            selected_predicted_cv = np.array(
                [generated.total_violation[first_index], half.total_violation[second_index]]
            )
            selected_sigma = np.full(2, np.nan)
            use_adaptation = True

            if np.min(pairwise_distance(offspring01[None, :], train.x)) <= 1e-11:
                joint_prediction, sigma = joint_model.predict_with_uncertainty(half.x)
                predicted_cv = np.sum(np.maximum(joint_prediction[:, 1:], 0.0), axis=1)
                feasible_indices = np.flatnonzero(predicted_cv == 0.0)
                if len(feasible_indices):
                    chosen = int(feasible_indices[np.argmax(sigma[feasible_indices])])
                else:
                    chosen = int(np.argmax(sigma))
                selected = half.x[chosen][None, :]
                selected_source = ["uncertainty_fallback"]
                selected_predicted_f = joint_prediction[chosen, :1]
                selected_predicted_cv = predicted_cv[chosen : chosen + 1]
                selected_sigma = sigma[chosen : chosen + 1]
                use_adaptation = False

            remaining = self.config.max_evaluations - len(train.x)
            if len(selected) > remaining:
                selected = selected[:remaining]
                selected_source = selected_source[:remaining]
                selected_predicted_f = selected_predicted_f[:remaining]
                selected_predicted_cv = selected_predicted_cv[:remaining]
                selected_sigma = selected_sigma[:remaining]
                use_adaptation = False

            pair_duplicate = len(selected) == 2 and np.array_equal(selected[0], selected[1])
            evaluation_x = selected[:1] if pair_duplicate else selected
            real_f, real_g, real_v = self._evaluate(evaluation_x)
            real_cv = np.sum(real_v, axis=1)
            off = _Population(evaluation_x, real_f, real_g, real_cv)

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
                        "rbf_variance": float(selected_sigma[i]),
                        "wmax": current_wmax,
                    }
                )

            if use_adaptation and len(selected) == 2:
                if pair_duplicate:
                    comparison_f = np.repeat(real_f, 2)
                    comparison_cv = np.repeat(real_cv, 2)
                else:
                    comparison_f = real_f
                    comparison_cv = real_cv
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

                actual_first_feasible = comparison_cv[0] == 0.0
                if actual_first_feasible != predicted_feasible:
                    minimum, maximum = constraint_model.influence_indices(offspring01)
                    if actual_first_feasible:
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
            history.append(self._history_entry(generation, train, population, current_wmax))
            population_history.append(
                self._population_snapshot(generation, len(train.x), population, train)
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
        )

    def _latin_hypercube_maximin(self, n_points: int) -> FloatArray:
        return latin_hypercube_maximin(
            n_points, self.lower, self.upper, self.rng, attempts=5
        )

    def _evaluate(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        points = as_2d_points(x, self.dimension)
        objective, constraints = self.problem.evaluate(points)
        objective = np.asarray(objective, dtype=float).reshape(-1)
        constraints = np.asarray(constraints, dtype=float)
        if constraints.ndim == 1:
            constraints = constraints[:, None]
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
        violations = np.maximum(constraints - self.config.constraint_tolerance, 0.0)
        return objective, constraints, violations

    def _generator_c2ode(
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
                current.total_violation = np.sum(
                    np.maximum(current.constraints, 0.0), axis=1
                )

            trial = self._generate_three_trials(current)
            predicted_f = objective_model.predict(trial)[:, 0]
            predicted_g = constraint_model.predict(trial)
            predicted_cv = np.sum(np.maximum(predicted_g, 0.0), axis=1)

            if self.config.save_candidate_pools:
                pool_trace.append(
                    {
                        "generation": outer_generation,
                        "inner_generation": inner_generation,
                        "stage": "c2ode_three_trials",
                        "x": trial.copy(),
                        "predicted_objective": predicted_f.copy(),
                        "predicted_total_violation": predicted_cv.copy(),
                    }
                )

            chosen = self._preselect_three(predicted_f, predicted_cv)
            selected_x = trial[chosen]
            selected_f = predicted_f[chosen]
            selected_g = predicted_g[chosen]
            selected_cv = predicted_cv[chosen]
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

    def _generate_three_trials(self, population: _Population) -> FloatArray:
        popsize, dimension = population.x.shape
        trials = np.empty((3 * popsize, dimension))
        feasible = np.flatnonzero(population.total_violation == 0.0)
        if len(feasible):
            best_solutions = population.x[feasible]
        else:
            best_solutions = population.x[
                [int(np.argmin(population.total_violation))]
            ]
        best_objective = int(np.argmin(population.objective))

        for i in range(popsize):
            others = np.delete(np.arange(popsize), i)

            f = self._draw_f()
            cr = self._draw_cr()
            r1, r2 = self.rng.choice(others, 2, replace=False)
            mutant = (
                population.x[i]
                + f * (population.x[best_objective] - population.x[i])
                + f * (population.x[r1] - population.x[r2])
            )
            mutant = self._repair(mutant)
            trials[3 * i] = self._binomial_crossover(population.x[i], mutant, cr)

            f = self._draw_f()
            cr = self._draw_cr()
            r1, r2, r3, r4 = self.rng.choice(others, 4, replace=False)
            guide = best_solutions[int(self.rng.integers(len(best_solutions)))]
            mutant = (
                population.x[r1]
                + f * (guide - population.x[r2])
                + f * (population.x[r3] - population.x[r4])
            )
            mutant = self._repair(mutant)
            trials[3 * i + 1] = self._binomial_crossover(
                population.x[i], mutant, cr
            )

            f = self._draw_f()
            r1, r2, r3 = self.rng.choice(others, 3, replace=False)
            mutant = (
                population.x[i]
                + self.rng.random() * (population.x[r1] - population.x[i])
                + f * (population.x[r2] - population.x[r3])
            )
            trials[3 * i + 2] = self._repair(mutant)
        return trials

    def _repair(self, x: FloatArray) -> FloatArray:
        repaired = np.asarray(x, dtype=float).copy()
        below = repaired < self.lower
        above = repaired > self.upper
        repaired[below] = np.minimum(
            self.upper[below], 2.0 * self.lower[below] - repaired[below]
        )
        repaired[above] = np.maximum(
            self.lower[above], 2.0 * self.upper[above] - repaired[above]
        )
        return repaired

    def _binomial_crossover(
        self, target: FloatArray, mutant: FloatArray, crossover_rate: float
    ) -> FloatArray:
        mask = self.rng.random(self.dimension) < crossover_rate
        mask[int(self.rng.integers(self.dimension))] = True
        return np.where(mask, mutant, target)

    def _draw_f(self) -> float:
        return float(self.rng.choice((0.6, 0.8, 1.0)))

    def _draw_cr(self) -> float:
        return float(self.rng.choice((0.1, 0.2, 1.0)))

    @staticmethod
    def _preselect_three(objective: FloatArray, cv: FloatArray) -> np.ndarray:
        popsize = len(objective) // 3
        selected = np.empty(popsize, dtype=int)
        for i in range(popsize):
            indices = np.arange(3 * i, 3 * i + 3)
            feasible = indices[cv[indices] == 0.0]
            if len(feasible):
                selected[i] = feasible[np.argmin(objective[feasible])]
            else:
                selected[i] = indices[np.argmin(cv[indices])]
        return selected

    @staticmethod
    def _epsilon_replacement_mask(
        objective: FloatArray,
        cv: FloatArray,
        trial_objective: FloatArray,
        trial_cv: FloatArray,
        epsilon: float,
    ) -> np.ndarray:
        both_below = (trial_cv < epsilon) & (cv < epsilon)
        tied = trial_cv == cv
        return (
            (both_below & (trial_objective < objective))
            | (~both_below & tied & (trial_objective < objective))
            | (~both_below & ~tied & (trial_cv < cv))
        )

    @staticmethod
    def _best_population_index(population: _Population) -> tuple[int, bool]:
        feasible = np.flatnonzero(population.total_violation == 0.0)
        if len(feasible):
            return int(feasible[np.argmin(population.objective[feasible])]), True
        return int(np.argmin(population.total_violation)), False

    @staticmethod
    def _update_a1(population: _Population, popsize: int) -> _Population:
        stable_index = np.arange(len(population.x))
        order = np.lexsort(
            (stable_index, population.objective, population.total_violation)
        )[:popsize]
        return _Population(
            population.x[order].copy(),
            population.objective[order].copy(),
            population.constraints[order].copy(),
            population.total_violation[order].copy(),
        )

    @staticmethod
    def _concatenate_population(first: _Population, second: _Population) -> _Population:
        return _Population(
            np.vstack((first.x, second.x)),
            np.concatenate((first.objective, second.objective)),
            np.vstack((first.constraints, second.constraints)),
            np.concatenate((first.total_violation, second.total_violation)),
        )

    @staticmethod
    def _unique_rows(x: FloatArray) -> FloatArray:
        if len(x) == 0:
            return np.asarray(x, dtype=float).copy()
        return np.unique(np.asarray(x, dtype=float), axis=0)

    @staticmethod
    def _unique_training_data(
        train: _Population,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        _, indices = np.unique(np.round(train.x, 6), axis=0, return_index=True)
        return train.x[indices], train.objective[indices], train.constraints[indices]

    @staticmethod
    def _remove_training_rows(
        x: FloatArray, y: FloatArray, deleted: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        if len(deleted) == 0:
            return x.copy(), y.copy()
        remove = np.any(
            np.all(x[:, None, :] == deleted[None, :, :], axis=2), axis=1
        )
        kept_x = x[~remove]
        kept_y = y[~remove]
        if len(kept_x) < 2:
            return x.copy(), y.copy()
        return kept_x, kept_y

    def _drop_random_row(self, rows: FloatArray) -> FloatArray:
        if len(rows) == 0:
            return rows
        return np.delete(rows, int(self.rng.integers(len(rows))), axis=0)

    @staticmethod
    def _final_index(objective: FloatArray, violations: FloatArray) -> int:
        valid = np.isfinite(objective) & np.all(np.isfinite(violations), axis=1)
        feasible = valid & np.all(violations <= 0.0, axis=1)
        if np.any(feasible):
            indices = np.flatnonzero(feasible)
            return int(indices[np.argmin(objective[indices])])
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise RuntimeError("The evaluator did not return any finite point.")
        total = np.sum(violations[indices], axis=1)
        order = np.lexsort((objective[indices], total))
        return int(indices[order[0]])

    @staticmethod
    def _history_entry(
        generation: int,
        train: _Population,
        population: _Population,
        wmax: int,
    ) -> dict[str, Any]:
        feasible = train.total_violation == 0.0
        best = float(np.min(train.objective[feasible])) if np.any(feasible) else None
        return {
            "generation": generation,
            "evaluations": len(train.x),
            "feasible_count": int(np.count_nonzero(feasible)),
            "population_feasible_count": int(
                np.count_nonzero(population.total_violation == 0.0)
            ),
            "best_feasible_objective": best,
            "minimum_total_violation": float(np.min(train.total_violation)),
            "sampling_method": "dsi_c2ode" if generation else None,
            "rbf_active": generation > 0,
            "wmax": wmax,
        }

    @staticmethod
    def _population_snapshot(
        generation: int,
        evaluations: int,
        population: _Population,
        train: _Population,
    ) -> dict[str, Any]:
        archive_indices = np.full(len(population.x), -1, dtype=int)
        for i, point in enumerate(population.x):
            matches = np.flatnonzero(np.all(train.x == point, axis=1))
            if len(matches):
                archive_indices[i] = int(matches[0])
        return {
            "generation": generation,
            "evaluations": evaluations,
            "archive_indices": archive_indices,
            "x": population.x.copy(),
            "objective": population.objective.copy(),
            "total_violation": population.total_violation.copy(),
        }
