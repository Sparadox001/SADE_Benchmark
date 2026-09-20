from __future__ import annotations

import numpy as np

from sade_benchmark import CallableProblem, SADEV2_11, SADEV2_11Config
from sade_benchmark.experiments import make_optimizer


def _problem(*, feasible: bool = False) -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum(x**2, axis=1)
        constraint = -np.ones(len(x)) if feasible else np.ones(len(x))
        return objective, constraint[:, None]

    return CallableProblem(
        name="v2_11_test",
        dimension=2,
        n_constraints=1,
        lower_bounds=0.0,
        upper_bounds=10.0,
        evaluator=evaluate,
    )


def test_infeasible_survival_replaces_only_the_nearest_parent() -> None:
    optimizer = SADEV2_11(
        _problem(),
        SADEV2_11Config(
            max_evaluations=30,
            population_size=6,
            surrogate_min_samples=6,
        ),
    )
    optimizer.constraint_scales_ = np.ones(1)
    archive_x = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [8.0, 0.0],
            [9.0, 0.0],
            [10.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ]
    )
    objective = np.zeros(8)
    violations = np.asarray(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [0.5], [0.4]]
    )

    survivors = optimizer._update_population(
        np.arange(6),
        np.asarray([6, 7]),
        archive_x,
        objective,
        violations,
        generation=0,
    )

    assert set(survivors) == {1, 2, 3, 4, 5, 7}
    assert optimizer._population_survival_mode == (
        "nearest_neighbor_infeasible"
    )
    assert optimizer._nearest_candidates_processed == 2
    assert optimizer._nearest_replacements == 1


def test_first_feasible_point_restores_global_survivor_truncation() -> None:
    optimizer = SADEV2_11(
        _problem(),
        SADEV2_11Config(
            max_evaluations=30,
            population_size=6,
            surrogate_min_samples=6,
        ),
    )
    optimizer.constraint_scales_ = np.ones(1)
    archive_x = np.arange(16, dtype=float).reshape(8, 2)
    objective = np.asarray([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 10.0, 9.0])
    violations = np.ones((8, 1))
    violations[7, 0] = 0.0
    expected = optimizer._select_survivors(
        np.arange(8), objective, violations, generation=0
    )

    survivors = optimizer._update_population(
        np.arange(6),
        np.asarray([6, 7]),
        archive_x,
        objective,
        violations,
        generation=0,
    )

    np.testing.assert_array_equal(survivors, expected)
    assert 7 in survivors
    assert optimizer._population_survival_mode == "global_feasible_first"
    assert optimizer._nearest_candidates_processed == 0
    assert optimizer._nearest_replacements == 0


def test_v2_11_history_records_the_active_survival_rule() -> None:
    infeasible = SADEV2_11(
        _problem(), SADEV2_11Config(max_evaluations=32, seed=4)
    ).optimize()
    feasible = SADEV2_11(
        _problem(feasible=True),
        SADEV2_11Config(max_evaluations=32, seed=4),
    ).optimize()

    assert infeasible.history[-1]["population_survival"] == (
        "nearest_neighbor_infeasible"
    )
    assert infeasible.history[-1]["nearest_candidates_processed"] == 2
    assert feasible.history[-1]["population_survival"] == (
        "global_feasible_first"
    )
    assert feasible.history[-1]["nearest_candidates_processed"] == 0


def test_registry_constructs_v2_11() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_11",
        _problem(),
        common_parameters={
            "max_evaluations": 30,
            "population_size": 30,
            "constraint_tolerance": 0.0,
        },
        algorithm_parameters={
            "batch_size": 2,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
            "boundary_stagnation_batches": 3,
        },
        seed=1,
        save_candidate_pools=False,
    )

    assert isinstance(optimizer, SADEV2_11)
    assert isinstance(config, SADEV2_11Config)
