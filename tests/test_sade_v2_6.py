from __future__ import annotations

from collections import Counter

import numpy as np

from sade_benchmark import CallableProblem, SADEV2_6, SADEV2_6Config
from sade_benchmark.experiments import make_optimizer


def _problem() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.1) ** 2, axis=1)
        constraints = np.column_stack(
            (np.sum(x, axis=1) - 0.25, -np.sum(x, axis=1) - 0.75)
        )
        return objective, constraints

    return CallableProblem(
        name="v2_6_toy",
        dimension=3,
        n_constraints=2,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_v2_6_defaults_to_two_point_batches() -> None:
    config = SADEV2_6Config(max_evaluations=36)
    assert config.batch_size == 2

    result = SADEV2_6(_problem(), config).optimize()
    sampled = result.evaluation_metadata[config.population_size :]
    assert len(sampled) == 6
    assert "boundary_exploration" not in {
        row["acquisition_role"] for row in sampled
    }

    for generation in {row["generation"] for row in sampled}:
        batch = [row for row in sampled if row["generation"] == generation]
        assert Counter(row["source"] for row in batch) == {
            "global_de": 1,
            "local_search": 1,
        }


def test_registry_constructs_v2_6() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_6",
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
        },
        seed=1,
        save_candidate_pools=False,
    )
    assert isinstance(optimizer, SADEV2_6)
    assert isinstance(config, SADEV2_6Config)
