from __future__ import annotations

import numpy as np

from sade_benchmark import CallableProblem, SADEV2_9, SADEV2_9Config
from sade_benchmark.experiments import make_optimizer


def _problem(*, feasible: bool) -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum(x**2, axis=1)
        sign = -1.0 if feasible else 1.0
        constraints = np.full((len(x), 1), sign, dtype=float)
        return objective, constraints

    return CallableProblem(
        name="v2_9_test",
        dimension=3,
        n_constraints=1,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_v2_9_switches_between_raw_and_robust_predicted_cv() -> None:
    optimizer = SADEV2_9(_problem(feasible=False))
    scores = {
        "predicted_total_violation": np.array([1.0, 2.0]),
        "robust_predicted_total_violation": np.array([3.0, 4.0]),
    }

    optimizer._raw_feasibility_acquisition = True
    assert np.array_equal(
        optimizer._acquisition_cv(scores), np.array([1.0, 2.0])
    )
    optimizer._raw_feasibility_acquisition = False
    assert np.array_equal(
        optimizer._acquisition_cv(scores), np.array([3.0, 4.0])
    )


def test_v2_9_logs_raw_constraint_mode_before_first_feasible_point() -> None:
    result = SADEV2_9(
        _problem(feasible=False),
        SADEV2_9Config(max_evaluations=32, seed=7),
    ).optimize()
    sampled = result.evaluation_metadata[30:]

    assert len(sampled) == 2
    assert all(
        row["constraint_acquisition_mode"] == "raw_predicted_cv"
        for row in sampled
    )
    assert result.history[-1]["constraint_acquisition"] == (
        "raw_predicted_cv_before_feasible_then_p90"
    )


def test_v2_9_retains_robust_constraint_mode_after_feasibility() -> None:
    result = SADEV2_9(
        _problem(feasible=True),
        SADEV2_9Config(max_evaluations=32, seed=9),
    ).optimize()
    sampled = result.evaluation_metadata[30:]

    assert all(
        row["constraint_acquisition_mode"] == "robust_p90_cv"
        for row in sampled
    )


def test_registry_constructs_v2_9() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_9",
        _problem(feasible=False),
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
    assert isinstance(optimizer, SADEV2_9)
    assert isinstance(config, SADEV2_9Config)
