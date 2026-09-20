from __future__ import annotations

import numpy as np

from sade_benchmark import CallableProblem, SADEV2_13, SADEV2_13Config
from sade_benchmark.experiments import make_optimizer


def _problem() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.sum(x**2, axis=1), -np.ones((len(x), 1))

    return CallableProblem(
        name="v2_13_test",
        dimension=2,
        n_constraints=1,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def _scores() -> dict[str, np.ndarray]:
    return {
        "predicted_objective": np.asarray([8.0, 1.0, 3.0, 2.0]),
        "predicted_feasible": np.ones(4, dtype=bool),
        "predicted_total_violation": np.zeros(4),
        "robust_predicted_feasible": np.zeros(4, dtype=bool),
        "empirical_joint_feasibility_probability": np.asarray(
            [0.2, 0.8, 0.8, 0.1]
        ),
        "surrogate_uncertainty": np.asarray([0.2, 0.1, 0.4, 0.3]),
    }


def test_probability_fallback_is_lexicographic() -> None:
    optimizer = SADEV2_13(_problem())
    index, rule = optimizer._objective_candidate_index(
        _scores(), np.ones(4)
    )

    assert index == 1
    assert rule == "probability_feasible_fallback"


def test_available_p90_gate_remains_minimum_predicted_objective() -> None:
    optimizer = SADEV2_13(_problem())
    scores = _scores()
    scores["robust_predicted_feasible"] = np.asarray(
        [True, False, True, False]
    )
    index, rule = optimizer._objective_candidate_index(
        scores, np.ones(4)
    )

    assert index == 2
    assert rule == "hard_feasible"


def test_registry_constructs_v2_13() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_13",
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

    assert isinstance(optimizer, SADEV2_13)
    assert isinstance(config, SADEV2_13Config)
