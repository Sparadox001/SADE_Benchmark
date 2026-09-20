from __future__ import annotations

import csv
import gzip

import numpy as np

from sade_benchmark import CallableProblem, SADEV2_3, SADEV2_3Config
from sade_benchmark.experiments import make_optimizer, save_run


def _problem() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.1) ** 2, axis=1)
        constraints = np.column_stack(
            (np.sum(x, axis=1) - 0.25, -np.sum(x, axis=1) - 0.75)
        )
        return objective, constraints

    return CallableProblem(
        name="v2_3_toy",
        dimension=3,
        n_constraints=2,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_v2_3_accumulates_only_positive_normalized_underprediction() -> None:
    optimizer = SADEV2_3(
        _problem(), SADEV2_3Config(max_evaluations=30)
    )
    optimizer._after_true_evaluation(
        [{"predicted_constraints": [1.0, 2.0]}],
        np.array([[2.0, 0.0]]),
        np.array([2.0, 4.0]),
    )
    np.testing.assert_allclose(
        optimizer._constraint_underprediction_errors[0], [0.5, 0.0]
    )
    optimizer._after_true_evaluation(
        [{"sampling_method": "random"}],
        np.array([[2.0, 0.0]]),
        np.array([2.0, 4.0]),
    )
    assert len(optimizer._constraint_underprediction_errors) == 1


def test_v2_3_robust_feasibility_uses_recent_error_quantile() -> None:
    config = SADEV2_3Config(
        max_evaluations=30,
        constraint_error_min_samples=3,
        constraint_error_window=3,
        constraint_error_quantile=0.9,
    )
    optimizer = SADEV2_3(_problem(), config)
    optimizer._constraint_underprediction_errors = [
        np.array([0.2, 0.4]),
        np.array([0.2, 0.4]),
        np.array([0.2, 0.4]),
    ]
    robust_cv, robust_feasible, margin, active = (
        optimizer._robust_constraint_scores(
            np.array([[-0.1, -0.5], [-0.3, -0.5]]), np.ones(2)
        )
    )
    assert active
    np.testing.assert_allclose(margin, [0.2, 0.4])
    np.testing.assert_allclose(robust_cv, [0.1, 0.0])
    np.testing.assert_array_equal(robust_feasible, [False, True])


def test_v2_3_calibration_activates_after_out_of_sample_minimum() -> None:
    result = SADEV2_3(
        _problem(), SADEV2_3Config(max_evaluations=45, seed=7)
    ).optimize()
    assert result.evaluations == 45
    assert not result.evaluation_metadata[39]["constraint_calibration_active"]
    assert result.evaluation_metadata[42]["constraint_calibration_active"]
    assert "robust_predicted_feasible" in result.evaluation_metadata[42]
    assert result.history[-1]["constraint_error_samples_used"] == 12
    assert result.history[-1]["constraint_acquisition"] == "online_one_sided_p90"


def test_v2_3_persistence_includes_calibration_diagnostics(tmp_path) -> None:
    problem = _problem()
    problem.suite = "test"
    config = SADEV2_3(
        problem,
        SADEV2_3Config(
            max_evaluations=45, seed=5, save_candidate_pools=True
        ),
    ).config
    result = SADEV2_3(problem, config).optimize()
    save_run(
        tmp_path,
        algorithm="sade_v2_3",
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
    assert rows[42]["robust_predicted_total_violation"] != ""
    assert rows[42]["constraint_error_quantiles"].startswith("[")
    pools = np.load(tmp_path / "candidate_pools.npz")
    for field in (
        "robust_predicted_total_violation",
        "robust_predicted_feasible",
        "constraint_error_quantiles",
        "constraint_calibration_active",
    ):
        assert field in pools


def test_registry_constructs_v2_3() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_3",
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
        },
        seed=1,
        save_candidate_pools=False,
    )
    assert isinstance(optimizer, SADEV2_3)
    assert isinstance(config, SADEV2_3Config)
