from __future__ import annotations

import numpy as np
import pytest

from sade_benchmark.benchmarks import BENCHMARKS, make_problem


def test_retained_problem_inventory() -> None:
    assert BENCHMARKS["cec2006"]["problems"] == (
        "g01", "g02", "g04", "g06", "g07", "g08", "g09",
        "g10", "g12", "g16", "g18", "g19", "g24",
    )
    assert BENCHMARKS["cec2010"]["problems"] == (
        "c01", "c07", "c08", "c13", "c14", "c15",
    )
    assert BENCHMARKS["cec2017"]["problems"] == (
        "c01", "c02", "c04", "c05", "c13", "c19", "c20", "c22", "c28",
    )
    assert BENCHMARKS["cec2017_objcap"]["problems"] == (
        "c01", "c02", "c04", "c05", "c13", "c20", "c22",
    )
    assert BENCHMARKS["cec2017_objcap_first_mean"]["problems"] == (
        "c01", "c02", "c04", "c05", "c13", "c20", "c22",
    )


@pytest.mark.parametrize("name", BENCHMARKS["cec2006"]["problems"])
def test_cec2006_batch_shapes(name: str) -> None:
    problem = make_problem("cec2006", name)
    rng = np.random.default_rng(10)
    x = problem.lower_bounds + rng.random((3, problem.dimension)) * (
        problem.upper_bounds - problem.lower_bounds
    )
    objective, constraints = problem.evaluate(x)
    assert objective.shape == (3,)
    assert constraints.shape == (3, problem.n_constraints)
    assert np.all(np.isfinite(objective))
    assert np.all(np.isfinite(constraints))


@pytest.mark.parametrize("name", BENCHMARKS["cec2010"]["problems"])
@pytest.mark.parametrize("dimension", (10, 30))
def test_cec2010_batch_shapes(name: str, dimension: int) -> None:
    problem = make_problem("cec2010", name, dimension)
    rng = np.random.default_rng(11)
    x = problem.lower_bounds + rng.random((3, dimension)) * (
        problem.upper_bounds - problem.lower_bounds
    )
    objective, constraints = problem.evaluate(x)
    assert objective.shape == (3,)
    assert constraints.shape == (3, problem.n_constraints)
    assert np.all(np.isfinite(objective))
    assert np.all(np.isfinite(constraints))


@pytest.mark.parametrize("name", BENCHMARKS["cec2017"]["problems"])
@pytest.mark.parametrize("dimension", (10, 30, 50, 100))
def test_cec2017_batch_shapes(name: str, dimension: int) -> None:
    problem = make_problem("cec2017", name, dimension)
    rng = np.random.default_rng(12)
    x = problem.lower_bounds + rng.random((3, dimension)) * (
        problem.upper_bounds - problem.lower_bounds
    )
    objective, constraints = problem.evaluate(x)
    assert objective.shape == (3,)
    assert constraints.shape == (3, problem.n_constraints)
    assert np.all(np.isfinite(objective))
    assert np.all(np.isfinite(constraints))


@pytest.mark.parametrize("name", ("c12", "c21"))
def test_cec2017_explicitly_excludes_c12_and_c21(name: str) -> None:
    with pytest.raises(ValueError, match="retained subset"):
        make_problem("cec2017", name, 10)


@pytest.mark.parametrize(
    ("name", "dimension"),
    [
        ("c01", 10), ("c01", 30), ("c02", 10), ("c02", 30),
        ("c04", 10), ("c04", 30), ("c05", 10), ("c05", 30),
        ("c13", 10), ("c20", 10), ("c20", 30), ("c22", 10),
    ],
)
def test_objective_cap_adds_exactly_the_frozen_inequality(
    name: str, dimension: int
) -> None:
    original = make_problem("cec2017", name, dimension)
    capped = make_problem("cec2017_objcap", name, dimension)
    rng = np.random.default_rng(2026)
    points = original.lower_bounds + rng.random((3, dimension)) * (
        original.upper_bounds - original.lower_bounds
    )
    objective, constraints = original.evaluate(points)
    capped_objective, capped_constraints = capped.evaluate(points)
    np.testing.assert_array_equal(capped_objective, objective)
    np.testing.assert_array_equal(capped_constraints[:, :-1], constraints)
    np.testing.assert_allclose(
        capped_constraints[:, -1], objective - capped.objective_cap, rtol=0, atol=0
    )
    assert capped.n_constraints == original.n_constraints + 1


def test_objective_cap_evaluates_base_once(monkeypatch) -> None:
    capped = make_problem("cec2017_objcap", "c01", 10)
    original_evaluate = capped._base.evaluate
    calls = 0

    def counted(points):
        nonlocal calls
        calls += 1
        return original_evaluate(points)

    monkeypatch.setattr(capped._base, "evaluate", counted)
    capped.evaluate(np.zeros((2, 10)))
    assert calls == 1


def test_objective_cap_rejects_unfrozen_instances() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        make_problem("cec2017_objcap", "c13", 30)
    with pytest.raises(ValueError, match="no frozen"):
        make_problem("cec2017_objcap", "c19", 10)


@pytest.mark.parametrize(
    ("name", "dimension"),
    [
        ("c01", 10), ("c01", 30), ("c02", 10), ("c02", 30),
        ("c04", 10), ("c04", 30), ("c05", 10), ("c05", 30),
        ("c13", 10), ("c20", 10), ("c20", 30), ("c22", 10),
    ],
)
def test_first_mean_variant_only_changes_the_cap(name: str, dimension: int) -> None:
    median_problem = make_problem("cec2017_objcap", name, dimension)
    mean_problem = make_problem("cec2017_objcap_first_mean", name, dimension)
    points = np.zeros((2, dimension))
    median_f, median_g = median_problem.evaluate(points)
    mean_f, mean_g = mean_problem.evaluate(points)
    np.testing.assert_array_equal(mean_f, median_f)
    np.testing.assert_array_equal(mean_g[:, :-1], median_g[:, :-1])
    np.testing.assert_allclose(
        mean_g[:, -1], mean_f - mean_problem.objective_cap, rtol=0, atol=0
    )
    assert mean_problem.objective_cap >= median_problem.objective_cap
    assert mean_problem.objective_constraint_index == mean_problem.n_constraints - 1
