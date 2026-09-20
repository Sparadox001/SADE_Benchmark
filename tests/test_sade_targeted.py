from __future__ import annotations

import numpy as np
import pytest

from sade_benchmark.algorithms.sade_targeted import SADETargeted, SADETargetedConfig
from sade_benchmark.algorithms.sade_targeted.tightening import (
    TargetedObjectiveConstraintTightener,
)
from sade_benchmark.benchmarks import make_problem


def test_targeted_search_replaces_existing_constraint_without_extra_column() -> None:
    config = SADETargetedConfig(population_size=6)
    tightener = TargetedObjectiveConstraintTightener(
        config, base_limit=100.0, constraint_index=1
    )
    objective = np.array([10.0, 80.0, 120.0])
    physical = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 20.0]])
    np.testing.assert_array_equal(tightener.search_violations(objective, physical), physical)
    tightener.active_limit = 60.0
    search = tightener.search_violations(objective, physical)
    assert search.shape == physical.shape
    np.testing.assert_array_equal(search[:, 0], physical[:, 0])
    np.testing.assert_array_equal(search[:, 1], [0.0, 20.0, 60.0])
    np.testing.assert_array_equal(physical[:, 1], [0.0, 0.0, 20.0])


def test_targeted_tightening_starts_at_the_fixed_problem_cap() -> None:
    config = SADETargetedConfig(
        population_size=6,
        tightening_trigger_feasible_count=6,
        tightening_patience=1,
    )
    tightener = TargetedObjectiveConstraintTightener(
        config, base_limit=100.0, constraint_index=1
    )
    objective = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    physical = np.zeros((6, 2))
    population = np.arange(6)
    assert not tightener.maybe_tighten(objective, physical, population, 1)
    assert tightener.maybe_tighten(objective, physical, population, 2)
    assert tightener.active_limit == pytest.approx(25.0)
    state = tightener.state()
    assert state["base_limit"] == 100.0
    assert state["target_constraint_index"] == 1
    assert state["constraint_mode"] == "replace_existing_objective_constraint"
    assert state["history"][0]["old_limit"] == 100.0


def test_targeted_optimizer_requires_the_objective_constraint() -> None:
    problem = make_problem("cec2017_objcap", "c04", 10)
    optimizer = SADETargeted(problem, SADETargetedConfig())
    assert optimizer.objective_tightener.constraint_index == problem.n_constraints - 1
    with pytest.raises(ValueError, match="not the objective-derived"):
        SADETargeted(problem, SADETargetedConfig(tightening_constraint_index=0))
    with pytest.raises(ValueError, match="outside"):
        SADETargeted(problem, SADETargetedConfig(tightening_constraint_index=20))
    with pytest.raises(ValueError, match="objective_cap"):
        SADETargeted(make_problem("cec2017", "c04", 10))
