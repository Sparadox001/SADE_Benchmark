from __future__ import annotations

import numpy as np

from sade_benchmark import CallableProblem, SADEV2_10, SADEV2_10Config
from sade_benchmark.experiments import make_optimizer


def _problem(dimension: int = 10) -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum(x**2, axis=1)
        constraints = np.full((len(x), 1), -1.0, dtype=float)
        return objective, constraints

    return CallableProblem(
        name="v2_10_test",
        dimension=dimension,
        n_constraints=1,
        lower_bounds=-100.0,
        upper_bounds=100.0,
        evaluator=evaluate,
    )


def test_v2_10_local_pool_is_evenly_split_across_fixed_scales() -> None:
    optimizer = SADEV2_10(_problem(), SADEV2_10Config(seed=3))
    center = np.zeros(10)

    pool = optimizer.local_search(center, 0, n_local=60)

    assert pool.shape == (60, 10)
    assert optimizer._last_coarse_local_pool.shape == (30, 10)
    assert optimizer._last_fine_local_pool.shape == (30, 10)
    coarse_changed = np.count_nonzero(
        optimizer._last_coarse_local_pool != center, axis=1
    )
    fine_changed = np.count_nonzero(
        optimizer._last_fine_local_pool != center, axis=1
    )
    assert np.all(coarse_changed == 3)
    assert np.all(fine_changed == 1)
    assert np.max(np.abs(optimizer._last_fine_local_pool)) <= 6.0


def test_v2_10_logs_selected_scale_and_split_candidate_pools() -> None:
    result = SADEV2_10(
        _problem(3),
        SADEV2_10Config(
            max_evaluations=32,
            seed=7,
            save_candidate_pools=True,
        ),
    ).optimize()
    sampled = result.evaluation_metadata[30:]
    local = [row for row in sampled if row["source"] == "local_search"]

    assert len(local) == 1
    assert local[0]["local_search_scale"] in {"coarse", "fine"}
    assert local[0]["local_search_changed_dimensions"] >= 1
    assert result.candidate_pools is not None
    stages = [row["stage"] for row in result.candidate_pools]
    assert stages == [
        "global_de",
        "local_search_coarse",
        "local_search_fine",
    ]
    sizes = [len(row["x"]) for row in result.candidate_pools]
    assert sizes == [180, 30, 30]


def test_v2_10_retains_v2_9_acquisition_and_adds_history_label() -> None:
    result = SADEV2_10(
        _problem(3), SADEV2_10Config(max_evaluations=32, seed=9)
    ).optimize()

    assert result.history[-1]["constraint_acquisition"] == (
        "raw_predicted_cv_before_feasible_then_p90"
    )
    assert result.history[-1]["local_search_design"] == (
        "half_coarse_half_fine"
    )
    assert result.history[-1]["local_fine_radius"] < (
        result.history[-1]["local_radius"]
    )


def test_registry_constructs_v2_10() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_10",
        _problem(3),
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
    assert isinstance(optimizer, SADEV2_10)
    assert isinstance(config, SADEV2_10Config)
