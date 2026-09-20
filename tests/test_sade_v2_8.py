from __future__ import annotations

import numpy as np
import pytest

from sade_benchmark import CallableProblem, SADEV2_8, SADEV2_8Config
from sade_benchmark.experiments import make_optimizer


def _problem(dimension: int = 3) -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum(x**2, axis=1)
        constraints = np.full((len(x), 1), -1.0, dtype=float)
        return objective, constraints

    return CallableProblem(
        name="v2_8_test",
        dimension=dimension,
        n_constraints=1,
        lower_bounds=0.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_v2_8_knn_reranks_global_rbf_shortlist() -> None:
    optimizer = SADEV2_8(
        _problem(1),
        SADEV2_8Config(
            max_evaluations=30,
            objective_shortlist_fraction=1.0,
            objective_shortlist_min_size=1,
            objective_knn_neighbors=1,
        ),
    )
    optimizer._objective_archive_x = np.array([[0.0], [1.0]])
    optimizer._objective_archive_f = np.array([100.0, 0.0])
    optimizer._scored_points = np.array([[0.1], [0.9], [0.5]])
    scores = {
        "robust_predicted_feasible": np.ones(3, dtype=bool),
        "predicted_objective": np.array([0.0, 1.0, 2.0]),
    }

    index, rule = optimizer._objective_candidate_index(
        scores, np.zeros(3)
    )

    assert index == 1
    assert rule == "hard_feasible_knn_rerank"


def test_v2_8_integration_logs_knn_role_and_local_estimate() -> None:
    result = SADEV2_8(
        _problem(), SADEV2_8Config(max_evaluations=32, seed=7)
    ).optimize()
    sampled = result.evaluation_metadata[30:]

    assert len(sampled) == 2
    assert any("knn_rerank" in row["acquisition_role"] for row in sampled)
    assert all("local_neighbor_objective" in row for row in sampled)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("objective_shortlist_fraction", 0.0),
        ("objective_shortlist_min_size", 0),
        ("objective_knn_neighbors", 0),
    ],
)
def test_v2_8_rejects_invalid_reranking_parameters(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError):
        SADEV2_8Config(**{field: value})


def test_registry_constructs_v2_8() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_8",
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
            "objective_shortlist_fraction": 0.20,
            "objective_shortlist_min_size": 10,
            "objective_knn_neighbors": 5,
        },
        seed=1,
        save_candidate_pools=False,
    )
    assert isinstance(optimizer, SADEV2_8)
    assert isinstance(config, SADEV2_8Config)
