from __future__ import annotations

import csv
import gzip

import numpy as np
import pytest

from sade_benchmark import CallableProblem, SADEV2_4, SADEV2_4Config
from sade_benchmark.experiments import make_optimizer, save_run


def _problem() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.1) ** 2, axis=1)
        constraints = np.column_stack(
            (np.sum(x, axis=1) - 0.25, -np.sum(x, axis=1) - 0.75)
        )
        return objective, constraints

    return CallableProblem(
        name="v2_4_toy",
        dimension=3,
        n_constraints=2,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_v2_4_config_validates_hard_feasible_fraction() -> None:
    assert SADEV2_4Config().hard_feasible_min_fraction == 0.05
    with pytest.raises(ValueError, match="hard_feasible_min_fraction"):
        SADEV2_4Config(hard_feasible_min_fraction=1.01)


def test_v2_4_accumulates_signed_joint_residual_vectors() -> None:
    optimizer = SADEV2_4(_problem(), SADEV2_4Config(max_evaluations=30))
    optimizer._after_true_evaluation(
        [{"predicted_constraints": [1.0, 2.0]}],
        np.array([[2.0, 0.0]]),
        np.array([2.0, 4.0]),
    )
    np.testing.assert_allclose(
        optimizer._signed_constraint_residuals[0], [0.5, -0.5]
    )
    np.testing.assert_allclose(
        optimizer._constraint_underprediction_errors[0], [0.5, 0.0]
    )


def test_v2_4_continuous_weight_uses_joint_residual_scenarios() -> None:
    config = SADEV2_4Config(
        max_evaluations=30,
        constraint_error_min_samples=2,
        constraint_error_window=2,
    )
    optimizer = SADEV2_4(_problem(), config)
    optimizer._signed_constraint_residuals = [
        np.array([1.0, -1.0]),
        np.array([-1.0, 1.0]),
    ]
    scores = optimizer._continuous_feasibility_scores(
        predicted_constraints=np.array([[-0.5, -0.5], [-2.0, -2.0]]),
        predicted_cv=np.zeros(2),
        constraint_scales=np.ones(2),
        objective_ei=np.array([2.0, 2.0]),
    )
    np.testing.assert_allclose(
        scores["empirical_joint_feasibility_probability"], [0.25, 0.75]
    )
    np.testing.assert_allclose(
        scores["expected_constraint_violation"], [0.5, 0.0]
    )
    np.testing.assert_allclose(
        scores["continuous_feasibility_weight"], [1.0 / 6.0, 0.75]
    )
    np.testing.assert_allclose(
        scores["constrained_expected_improvement"], [1.0 / 3.0, 1.5]
    )


def test_v2_4_uses_continuous_cei_when_hard_set_is_sparse() -> None:
    optimizer = SADEV2_4(
        _problem(),
        SADEV2_4Config(
            max_evaluations=30, hard_feasible_min_fraction=0.5
        ),
    )
    scores = {
        "robust_predicted_feasible": np.array([True, False, False, False]),
        "objective_ei": np.array([10.0, 1.0, 1.0, 1.0]),
        "constrained_expected_improvement": np.array([0.1, 0.2, 0.8, 0.3]),
        "continuous_feasibility_weight": np.array([0.1, 0.4, 0.5, 0.6]),
        "surrogate_uncertainty": np.array([0.1, 0.2, 0.3, 0.4]),
    }
    index, rule = optimizer._objective_candidate_index(
        scores, np.zeros(4)
    )
    assert index == 2
    assert rule == "continuous_cei"

    scores["robust_predicted_feasible"] = np.array(
        [True, True, False, False]
    )
    index, rule = optimizer._objective_candidate_index(scores, np.zeros(4))
    assert index == 0
    assert rule == "hard_feasible"


def test_v2_4_persists_continuous_acquisition_diagnostics(tmp_path) -> None:
    problem = _problem()
    problem.suite = "test"
    config = SADEV2_4Config(
        max_evaluations=45, seed=5, save_candidate_pools=True
    )
    result = SADEV2_4(problem, config).optimize()
    save_run(
        tmp_path,
        algorithm="sade_v2_4",
        problem=problem,
        run=1,
        seed=5,
        config=config,
        result=result,
    )
    with gzip.open(
        tmp_path / "evaluations.csv.gz", "rt", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[42]["continuous_feasibility_active"] == "True"
    assert rows[42]["constraint_residual_samples"] == "12"
    assert rows[42]["continuous_feasibility_weight"] != ""
    assert rows[42]["constrained_expected_improvement"] != ""
    assert result.history[-1]["constraint_acquisition"] == (
        "hybrid_p90_continuous_cei"
    )

    pools = np.load(tmp_path / "candidate_pools.npz")
    for field in (
        "empirical_joint_feasibility_probability",
        "expected_constraint_violation",
        "continuous_feasibility_weight",
        "constrained_expected_improvement",
        "constraint_residual_samples",
        "continuous_feasibility_active",
    ):
        assert field in pools


def test_registry_constructs_v2_4() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_4",
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
    assert isinstance(optimizer, SADEV2_4)
    assert isinstance(config, SADEV2_4Config)
