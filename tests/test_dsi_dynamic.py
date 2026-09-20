from __future__ import annotations

import numpy as np

from sade_benchmark import (
    CallableProblem,
    DSI,
    DSIConfig,
    DSIDynamic,
    DSIDynamicConfig,
)
from sade_benchmark.experiments import load_configuration, make_algorithm_config


def always_feasible_sphere() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.2) ** 2, axis=1)
        constraints = -np.ones((len(x), 1))
        return objective, constraints

    return CallableProblem(
        name="dsi_dynamic_test",
        dimension=3,
        n_constraints=1,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def constrained_sphere() -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.1) ** 2, axis=1)
        constraints = (np.sum(x, axis=1) - 0.5)[:, None]
        return objective, constraints

    return CallableProblem(
        name="dsi_dynamic_equivalence_test",
        dimension=3,
        n_constraints=1,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )


def test_disabled_dynamic_variant_matches_original_dsi_trajectory() -> None:
    problem = constrained_sphere()
    base = DSI(
        problem,
        DSIConfig(max_evaluations=17, population_size=6, wmax=2, seed=9),
    ).optimize()
    variant = DSIDynamic(
        problem,
        DSIDynamicConfig(
            max_evaluations=17,
            population_size=6,
            wmax=2,
            seed=9,
            dynamic_constraint_tightening=False,
        ),
    ).optimize()

    np.testing.assert_allclose(variant.archive_x, base.archive_x)
    np.testing.assert_allclose(variant.archive_objective, base.archive_objective)
    np.testing.assert_allclose(variant.archive_constraints, base.archive_constraints)
    assert variant.objective == base.objective
    assert variant.dynamic_constraint_state is None


def test_dynamic_variant_tightens_and_keeps_physical_result_semantics() -> None:
    result = DSIDynamic(
        always_feasible_sphere(),
        DSIDynamicConfig(
            max_evaluations=18,
            population_size=6,
            wmax=1,
            seed=4,
            tightening_patience=1,
            tightening_improvement_tol=1e9,
        ),
    ).optimize()

    state = result.dynamic_constraint_state
    assert state is not None
    assert state["tighten_count"] >= 1
    assert state["threshold_active"]
    assert result.feasible
    assert result.archive_violation.shape == (18, 1)
    assert any(entry["dynamic_constraint_tightened"] for entry in result.history)
    assert all(
        entry["search_feasible_count"] <= entry["feasible_count"]
        for entry in result.history
    )


def test_dsi_dynamic_is_registered_with_independent_defaults() -> None:
    configuration = load_configuration()
    config = make_algorithm_config(
        "dsi_dynamic",
        common_parameters=configuration["common"],
        algorithm_parameters=configuration["algorithms"]["dsi_dynamic"],
        seed=1,
        save_candidate_pools=False,
    )
    assert isinstance(config, DSIDynamicConfig)
    assert config.dynamic_constraint_tightening
    assert config.tightening_trigger_feasible_count is None
