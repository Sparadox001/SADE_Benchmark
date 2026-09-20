from __future__ import annotations

import numpy as np

from sade_benchmark import CallableProblem, SADEV2_5, SADEV2_5Config
from sade_benchmark.experiments import make_optimizer


def _problem() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.1) ** 2, axis=1)
        constraints = np.column_stack(
            (np.sum(x, axis=1) - 0.25, -np.sum(x, axis=1) - 0.75)
        )
        return objective, constraints

    return CallableProblem(
        name="v2_5_toy",
        dimension=3,
        n_constraints=2,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def _scores() -> dict[str, np.ndarray]:
    return {
        "robust_predicted_feasible": np.array([True, True, False, False]),
        "predicted_objective": np.array([5.0, 1.0, -100.0, 2.0]),
        "objective_ei": np.array([100.0, 1.0, 50.0, 2.0]),
        "constrained_expected_improvement": np.array([0.1, 0.2, 0.8, 0.3]),
        "continuous_feasibility_weight": np.array([0.1, 0.4, 0.5, 0.6]),
        "surrogate_uncertainty": np.array([0.1, 0.2, 0.3, 0.4]),
    }


def test_v2_5_uses_minimum_predicted_objective_in_robust_set() -> None:
    optimizer = SADEV2_5(
        _problem(), SADEV2_5Config(max_evaluations=30)
    )
    index, rule = optimizer._objective_candidate_index(
        _scores(), np.zeros(4)
    )
    assert index == 1
    assert rule == "hard_feasible"


def test_v2_5_retains_continuous_cei_fallback() -> None:
    optimizer = SADEV2_5(
        _problem(),
        SADEV2_5Config(
            max_evaluations=30, hard_feasible_min_fraction=0.75
        ),
    )
    index, rule = optimizer._objective_candidate_index(
        _scores(), np.zeros(4)
    )
    assert index == 2
    assert rule == "continuous_cei"


def test_v2_5_trace_labels_match_the_actual_acquisition() -> None:
    assert SADEV2_5._objective_acquisition_role(
        "global_de", "unused", "hard_feasible"
    ) == "global_robust_feasible_min_predicted_objective"
    assert SADEV2_5._objective_acquisition_role(
        "local_search", "unused", "continuous_cei"
    ) == "local_continuous_cei_fallback"

    result = SADEV2_5(
        _problem(), SADEV2_5Config(max_evaluations=33, seed=4)
    ).optimize()
    assert result.history[-1]["objective_acquisition"] == (
        "min_predicted_objective_with_continuous_cei_fallback"
    )


def test_registry_constructs_v2_5() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_5",
        _problem(),
        common_parameters={
            "max_evaluations": 30,
            "population_size": 30,
            "constraint_tolerance": 0.0,
        },
        algorithm_parameters={
            "batch_size": 3,
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
        },
        seed=1,
        save_candidate_pools=False,
    )
    assert isinstance(optimizer, SADEV2_5)
    assert isinstance(config, SADEV2_5Config)

