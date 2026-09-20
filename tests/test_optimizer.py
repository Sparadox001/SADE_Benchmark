from __future__ import annotations

import numpy as np

from sade_benchmark import CallableProblem, SADE, SADEConfig
from sade_benchmark.algorithms.sade.tightening import ObjectiveConstraintTightener
from run_benchmark import result_row


def constrained_sphere() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.2) ** 2, axis=1)
        constraints = np.column_stack(
            (np.sum(x, axis=1) - 1.0, -np.sum(x, axis=1) - 1.0)
        )
        return objective, constraints

    return CallableProblem(
        name="constrained_sphere",
        dimension=4,
        n_constraints=2,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_sade_uses_exact_budget_and_is_reproducible() -> None:
    config = SADEConfig(
        max_evaluations=47,
        population_size=12,
        batch_size=7,
        seed=42,
        surrogate_min_samples=8,
    )
    first = SADE(constrained_sphere(), config).optimize()
    second = SADE(constrained_sphere(), config).optimize()

    assert first.evaluations == 47
    assert first.archive_x.shape == (47, 4)
    assert first.archive_constraints.shape == (47, 2)
    assert first.feasible
    np.testing.assert_allclose(first.archive_x, second.archive_x)
    np.testing.assert_allclose(first.archive_objective, second.archive_objective)


def test_final_choice_is_feasible_first() -> None:
    objective = np.array([-100.0, 2.0, 1.0])
    violations = np.array([[0.1], [0.0], [0.0]])
    assert SADE._final_index(objective, violations) == 2


def test_discrete_style_survivor_selection_uses_penalty_first() -> None:
    optimizer = SADE(
        constrained_sphere(),
        SADEConfig(max_evaluations=12, population_size=6, batch_size=2),
    )
    indices = np.arange(7)
    objective = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 0.0])
    violations = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
    survivors = optimizer._select_survivors(indices, objective, violations, generation=1)
    np.testing.assert_array_equal(survivors, np.array([5, 4, 3, 2, 1, 0]))


def test_rbf_threshold_and_discrete_style_local_radius() -> None:
    config = SADEConfig()
    assert config.surrogate_min_samples == 30
    optimizer = SADE(constrained_sphere(), config)
    total_generations = int(
        np.ceil((config.max_evaluations - config.population_size) / config.batch_size)
    )
    assert optimizer._local_radius(0) == 0.30
    assert np.isclose(optimizer._local_radius(total_generations), 0.10)


def test_result_objective_uses_scientific_notation() -> None:
    class Result:
        objective = 1234.5
        violation = np.array([0.0])
        feasible = True
        evaluations = 10
        generations = 1
        x = np.array([0.0])
        constraints = np.array([-1.0])

    class Problem:
        suite = "test"
        name = "p01"
        dimension = 1

    row = result_row(Problem(), 1, 1, Result())
    assert row["objective"] == "1.2345000000000000e+03"


def test_sampling_history_switches_to_ei_at_configured_80_archive_samples() -> None:
    result = SADE(
        constrained_sphere(),
        SADEConfig(
            max_evaluations=90,
            population_size=30,
            batch_size=10,
            seed=7,
            surrogate_min_samples=80,
        ),
    ).optimize()
    methods_by_evaluations = {
        item["evaluations"]: item["sampling_method"] for item in result.history
    }
    assert methods_by_evaluations[80] == "random"
    assert methods_by_evaluations[90] == "expected_improvement"


def test_first_survivor_update_uses_original_generation_zero_penalty() -> None:
    class TrackingSADE(SADE):
        survivor_generations: list[int]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.survivor_generations = []

        def _select_survivors(self, indices, objective, violations, generation):
            self.survivor_generations.append(generation)
            return super()._select_survivors(
                indices, objective, violations, generation
            )

    optimizer = TrackingSADE(
        constrained_sphere(),
        SADEConfig(max_evaluations=18, population_size=12, batch_size=6),
    )
    optimizer.optimize()
    assert optimizer.survivor_generations == [0]


def test_objective_constraint_tightener_uses_current_feasible_range() -> None:
    config = SADEConfig(
        max_evaluations=6,
        population_size=6,
        dynamic_constraint_tightening=True,
        tightening_trigger_feasible_count=6,
        tightening_keep_ratio=0.5,
        tightening_patience=1,
    )
    tightener = ObjectiveConstraintTightener(config)
    objective = np.arange(1.0, 7.0)
    original_violations = np.zeros((6, 1))
    population = np.arange(6)

    assert not tightener.maybe_tighten(
        objective, original_violations, population, generation=0
    )
    assert tightener.maybe_tighten(
        objective, original_violations, population, generation=1
    )
    assert tightener.active_limit == 3.5
    assert tightener.phase_id == 1
    assert tightener.tighten_count == 1

    search_violations = tightener.search_violations(
        objective, original_violations
    )
    assert search_violations.shape == (6, 2)
    np.testing.assert_allclose(
        search_violations[:, -1],
        np.array([0.0, 0.0, 0.0, 1.0 / 7.0, 3.0 / 7.0, 5.0 / 7.0]),
    )


def test_tightening_progress_resets_when_population_loses_feasibility() -> None:
    config = SADEConfig(
        max_evaluations=6,
        population_size=6,
        dynamic_constraint_tightening=True,
        tightening_patience=2,
    )
    tightener = ObjectiveConstraintTightener(config)
    objective = np.arange(1.0, 7.0)
    violations = np.zeros((6, 1))
    population = np.arange(6)

    assert not tightener.maybe_tighten(objective, violations, population, 0)
    assert len(tightener.phase_progress) == 1
    violations[-1, 0] = 1.0
    assert not tightener.maybe_tighten(objective, violations, population, 1)
    assert tightener.phase_progress == []


def test_tightening_stagnation_tracks_phase_running_best_not_range_limit() -> None:
    config = SADEConfig(
        max_evaluations=6,
        population_size=6,
        dynamic_constraint_tightening=True,
        tightening_trigger_feasible_count=6,
        tightening_keep_ratio=0.5,
        tightening_patience=1,
        tightening_improvement_tol=0.01,
    )
    tightener = ObjectiveConstraintTightener(config)
    violations = np.zeros((6, 1))
    population = np.arange(6)

    first = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    improved = np.array([0.5, 2.0, 3.0, 4.0, 5.0, 100.0])
    assert not tightener.maybe_tighten(first, violations, population, 0)
    assert not tightener.maybe_tighten(improved, violations, population, 1)
    assert tightener.active_limit is None
    assert tightener.maybe_tighten(improved, violations, population, 2)
    assert tightener.active_limit == 50.25


def test_sade_optional_objective_constraint_keeps_original_result_semantics() -> None:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum(x**2, axis=1)
        constraints = -np.ones((len(x), 1))
        return objective, constraints

    problem = CallableProblem(
        name="always_feasible_sphere",
        dimension=2,
        n_constraints=1,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )
    result = SADE(
        problem,
        SADEConfig(
            max_evaluations=18,
            population_size=6,
            batch_size=6,
            seed=3,
            dynamic_constraint_tightening=True,
            tightening_trigger_feasible_count=6,
            tightening_keep_ratio=0.5,
            tightening_patience=1,
            tightening_improvement_tol=1e9,
        ),
    ).optimize()

    assert result.feasible
    assert result.archive_violation.shape == (18, 1)
    assert result.dynamic_constraint_state is not None
    assert result.dynamic_constraint_state["threshold_active"]
    assert result.dynamic_constraint_state["tighten_count"] == 1
    assert result.history[-1]["dynamic_constraint_active"]
