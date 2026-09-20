from __future__ import annotations

from typing import Any

import numpy as np

from sade_benchmark import CallableProblem, SADEV2_7, SADEV2_7Config
from sade_benchmark.experiments import make_optimizer


def _problem() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.zeros(len(x), dtype=float)
        constraints = np.full((len(x), 1), -1.0, dtype=float)
        return objective, constraints

    return CallableProblem(
        name="v2_7_constant",
        dimension=3,
        n_constraints=1,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_v2_7_dynamic_boundary_schedule_uses_true_stagnation() -> None:
    result = SADEV2_7(
        _problem(),
        SADEV2_7Config(
            max_evaluations=46,
            seed=4,
            boundary_stagnation_batches=3,
        ),
    ).optimize()
    sampled = result.evaluation_metadata[30:]
    roles = {
        generation: [
            row["acquisition_role"]
            for row in sampled
            if row["generation"] == generation
        ]
        for generation in range(1, 9)
    }
    assert all("boundary_exploration" not in roles[g] for g in (1, 2, 3))
    assert "boundary_exploration" in roles[4]
    assert all("boundary_exploration" not in roles[g] for g in (5, 6, 7))
    assert "boundary_exploration" in roles[8]


def test_true_improvement_resets_v2_7_stagnation() -> None:
    optimizer = SADEV2_7(
        _problem(), SADEV2_7Config(max_evaluations=30)
    )
    optimizer._stagnant_batches = 3
    metadata: list[dict[str, Any]] = [
        {"acquisition_role": "boundary_exploration"}
    ]
    optimizer._after_expensive_batch(
        np.array([10.0]),
        np.array([[0.0]]),
        np.array([9.0, 11.0]),
        np.zeros((2, 1)),
        np.ones(1),
        metadata,
    )
    assert optimizer._stagnant_batches == 0


def test_boundary_batch_resets_v2_7_stagnation_without_improvement() -> None:
    optimizer = SADEV2_7(
        _problem(), SADEV2_7Config(max_evaluations=30)
    )
    optimizer._stagnant_batches = 3
    optimizer._after_expensive_batch(
        np.array([10.0]),
        np.array([[0.0]]),
        np.array([11.0, 12.0]),
        np.zeros((2, 1)),
        np.ones(1),
        [{"acquisition_role": "boundary_exploration"}],
    )
    assert optimizer._stagnant_batches == 0


def test_registry_constructs_v2_7() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_7",
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
    assert isinstance(optimizer, SADEV2_7)
    assert isinstance(config, SADEV2_7Config)
