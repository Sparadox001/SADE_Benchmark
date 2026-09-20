from __future__ import annotations

import numpy as np

from sade_benchmark import CallableProblem, SADEV2_2, SADEV2_2Config
from sade_benchmark.experiments import make_optimizer


def _problem(dimension: int = 3) -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.1) ** 2, axis=1)
        constraints = np.column_stack(
            (np.sum(x, axis=1) - 0.5, -np.sum(x, axis=1) - 0.5)
        )
        return objective, constraints

    return CallableProblem(
        name="v2_2_toy",
        dimension=dimension,
        n_constraints=2,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_v2_2_feasible_first_order_uses_fixed_scaled_cv() -> None:
    optimizer = SADEV2_2(
        _problem(),
        SADEV2_2Config(max_evaluations=6, population_size=6),
    )
    optimizer.constraint_scales_ = np.array([1.0, 100.0])
    objective = np.array([-100.0, 10.0, 1.0, -10.0, 5.0, 2.0, 3.0])
    violations = np.array(
        [
            [0.10, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.20, 0.0],
            [0.0, 2.0],
            [0.03, 0.0],
            [0.0, 4.0],
        ]
    )
    order = optimizer._feasible_first_order(objective, violations)
    # Feasible objective 1 before feasible objective 10; then CV=.02,.03,.04,.10,.20.
    np.testing.assert_array_equal(order, np.array([2, 1, 4, 5, 6, 0, 3]))


def test_v2_2_survivors_and_guide_scores_share_the_same_order() -> None:
    optimizer = SADEV2_2(
        _problem(),
        SADEV2_2Config(max_evaluations=6, population_size=6),
    )
    optimizer.constraint_scales_ = np.array([1.0, 10.0])
    objective = np.array([8.0, 1.0, 2.0, 0.0, 4.0, 3.0, 5.0])
    violations = np.array(
        [[0, 0], [0.2, 0], [0, 0], [1, 0], [0, 1], [0.05, 0], [0, 2]],
        dtype=float,
    )
    expected = optimizer._feasible_first_order(objective, violations)
    survivors = optimizer._select_survivors(
        np.arange(7), objective, violations, generation=99
    )
    rank_score, fixed_cv, weights = optimizer._population_guide_scores(
        objective, violations, generation=99
    )
    np.testing.assert_array_equal(survivors, expected[:6])
    np.testing.assert_array_equal(np.argsort(rank_score), expected)
    np.testing.assert_allclose(fixed_cv, np.sum(violations / [1.0, 10.0], axis=1))
    np.testing.assert_allclose(weights, np.array([1.0, 0.1]))


def test_v2_2_final_index_uses_the_same_fixed_constraint_scales() -> None:
    optimizer = SADEV2_2(
        _problem(),
        SADEV2_2Config(max_evaluations=6, population_size=6),
    )
    optimizer.constraint_scales_ = np.array([1.0, 1000.0])
    objective = np.array([0.0, 10.0])
    violations = np.array([[0.1, 0.0], [0.0, 10.0]])
    # Fixed CV prefers the second point (.01 < .10), even though its objective
    # is worse. The old archive-max rescaling tied both violations.
    assert optimizer._final_index(objective, violations) == 1


def test_v2_2_shortage_fallback_preserves_distance_threshold() -> None:
    config = SADEV2_2Config(
        max_evaluations=6,
        population_size=6,
        distance_threshold=0.2,
        seed=3,
    )
    optimizer = SADEV2_2(_problem(), config)
    archive = np.zeros((1, 3))
    pool = np.array(
        [
            [0.01, 0.00, 0.00],
            [0.80, 0.80, 0.80],
            [-0.80, 0.80, 0.80],
            [0.80, -0.80, 0.80],
            [0.80, 0.80, -0.80],
        ]
    )
    completed, sources = optimizer._complete_batch_v2(
        np.empty((0, 3)), [], pool, pool[:1], 3, archive
    )
    normalized = (np.vstack((archive, completed)) + 1.0) / 2.0
    distances = np.linalg.norm(
        normalized[:, None, :] - normalized[None, :, :], axis=2
    ) / np.sqrt(3)
    distances[np.eye(len(distances), dtype=bool)] = np.inf
    assert np.min(distances) > config.distance_threshold
    assert sources == ["batch_shortage_distance_fallback"] * 3


def test_v2_2_local_center_is_best_true_feasible_point() -> None:
    class TrackingV22(SADEV2_2):
        local_center: np.ndarray | None = None

        def local_search(self, best_vec, current_gen, *, n_local):
            self.local_center = np.asarray(best_vec).copy()
            return np.repeat(np.asarray(best_vec)[None, :], n_local, axis=0)

    optimizer = TrackingV22(
        _problem(),
        SADEV2_2Config(max_evaluations=6, population_size=6, seed=4),
    )
    optimizer.constraint_scales_ = np.ones(2)
    population = np.arange(18, dtype=float).reshape(6, 3) / 20.0
    objective = np.array([10.0, 8.0, 6.0, 0.5, 2.0, 1.0])
    violations = np.array(
        [[1, 0], [0.5, 0], [0.1, 0], [2, 0], [0, 0], [0, 0]], dtype=float
    )
    fitness, penalty, _ = optimizer._population_guide_scores(
        objective, violations, generation=0
    )
    optimizer._generate_candidate_pool(
        population, objective, violations, fitness, penalty, generation=0
    )
    np.testing.assert_array_equal(optimizer.local_center, population[5])


def test_v2_2_end_to_end_budget_reproducibility_and_history() -> None:
    config = SADEV2_2Config(
        max_evaluations=36,
        population_size=30,
        surrogate_min_samples=30,
        seed=12,
    )
    first = SADEV2_2(_problem(), config).optimize()
    second = SADEV2_2(_problem(), config).optimize()
    assert first.evaluations == 36
    np.testing.assert_allclose(first.archive_x, second.archive_x)
    assert first.history[-1]["population_ranking"] == "fixed_scale_feasible_first"
    assert first.evaluation_metadata[30]["sampling_method"] == "two_stage_rbf"


def test_registry_constructs_v2_2() -> None:
    optimizer, config = make_optimizer(
        "sade_v2_2",
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
        },
        seed=1,
        save_candidate_pools=False,
    )
    assert isinstance(optimizer, SADEV2_2)
    assert isinstance(config, SADEV2_2Config)
