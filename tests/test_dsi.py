from __future__ import annotations

import numpy as np

from sade_benchmark import CallableProblem, DSI, DSIConfig


def constrained_sphere() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.1) ** 2, axis=1)
        constraints = (np.sum(x, axis=1) - 0.5)[:, None]
        return objective, constraints

    return CallableProblem(
        name="dsi_test",
        dimension=3,
        n_constraints=1,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_dsi_is_reproducible_and_respects_exact_budget() -> None:
    config = DSIConfig(
        max_evaluations=17,
        population_size=6,
        wmax=2,
        seed=12,
    )
    first = DSI(constrained_sphere(), config).optimize()
    second = DSI(constrained_sphere(), config).optimize()

    assert first.evaluations == 17
    assert first.archive_x.shape == (17, 3)
    assert first.archive_constraints.shape == (17, 1)
    assert len(first.evaluation_metadata) == 17
    assert len(first.population_history) == first.generations + 1
    np.testing.assert_allclose(first.archive_x, second.archive_x)
    np.testing.assert_allclose(first.archive_objective, second.archive_objective)


def test_dsi_candidate_pool_trace_is_opt_in() -> None:
    without_trace = DSI(
        constrained_sphere(),
        DSIConfig(max_evaluations=8, population_size=6, wmax=1),
    ).optimize()
    with_trace = DSI(
        constrained_sphere(),
        DSIConfig(
            max_evaluations=8,
            population_size=6,
            wmax=1,
            save_candidate_pools=True,
        ),
    ).optimize()
    assert without_trace.candidate_pools is None
    assert with_trace.candidate_pools

