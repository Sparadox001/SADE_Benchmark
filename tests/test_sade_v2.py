from __future__ import annotations

import numpy as np
import gzip
import csv

from sade_benchmark import CallableProblem, DSI, DSIConfig, SADEV2, SADEV2Config
from sade_benchmark.algorithms.sade_v2.surrogate import (
    CubicRBFMultiOutput,
    standardized_expected_improvement,
)
from sade_benchmark.algorithms.sade_v2.optimizer import (
    _low_cv_shortlist,
    _minimum_cv_indices,
)
from sade_benchmark.experiments import make_optimizer, save_run


def _problem(*, feasible: bool) -> CallableProblem:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objective = np.sum((x - 0.15) ** 2, axis=1)
        sign = -1.0 if feasible else 1.0
        constraints = np.column_stack(
            (np.full(len(x), sign), np.full(len(x), 2.0 * sign))
        )
        return objective, constraints

    return CallableProblem(
        name="v2_toy", dimension=3, n_constraints=2,
        lower_bounds=-1.0, upper_bounds=1.0, evaluator=evaluate,
    )


def test_multioutput_rbf_shapes_and_training_interpolation() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(-1.0, 1.0, (12, 2))
    y = np.column_stack((x[:, 0] ** 2 + x[:, 1], 3.0 * x[:, 0] - x[:, 1]))
    model = CubicRBFMultiOutput(np.full(2, -1.0), np.ones(2)).fit(x, y)

    prediction = model.predict(x)
    assert prediction.shape == y.shape
    np.testing.assert_allclose(prediction, y, atol=2e-7, rtol=2e-7)


def test_v2_and_dsi_use_identical_initial_design_for_the_same_seed() -> None:
    problem = _problem(feasible=True)
    v2 = SADEV2(problem, SADEV2Config(max_evaluations=30, seed=19)).optimize()
    dsi = DSI(
        problem, DSIConfig(max_evaluations=30, population_size=30, seed=19)
    ).optimize()
    np.testing.assert_array_equal(v2.archive_x, dsi.archive_x)
    assert v2.evaluation_metadata[0]["sampling_method"] == "latin_hypercube_maximin"


def test_standardized_ei_ranking_is_invariant_to_objective_scale() -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 1.0, (20, 2))
    candidates = rng.uniform(0.0, 1.0, (40, 2))
    y = (x[:, 0] - 0.2) ** 2 + 2.0 * (x[:, 1] - 0.7) ** 2
    rankings = []
    for scale, shift in ((1.0, 0.0), (1000.0, -37.0)):
        scaled_y = scale * y + shift
        model = CubicRBFMultiOutput(np.zeros(2), np.ones(2)).fit(x, scaled_y)
        mean = model.predict_normalized(candidates)[:, 0]
        incumbent = model.normalize_outputs(np.asarray([np.min(scaled_y)]))[0, 0]
        ei = standardized_expected_improvement(
            mean, incumbent, model.geometric_uncertainty(candidates)
        )
        assert np.all(np.isfinite(ei))
        rankings.append(np.argsort(-ei))
    np.testing.assert_array_equal(rankings[0], rankings[1])


def test_low_cv_selection_keeps_all_candidates_tied_at_cutoff() -> None:
    cv = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(_low_cv_shortlist(cv, 0.2), np.array([0, 1, 2]))
    np.testing.assert_array_equal(_minimum_cv_indices(cv), np.array([0, 1, 2]))


def test_fixed_constraint_scales_use_positive_violation_p90() -> None:
    optimizer = SADEV2(_problem(feasible=False), SADEV2Config(max_evaluations=30))
    constraints = np.column_stack(
        (np.arange(-5.0, 25.0), -np.arange(1.0, 31.0))
    )
    scales = optimizer._initial_constraint_scales(constraints)
    assert np.isclose(scales[0], np.percentile(np.arange(1.0, 25.0), 90.0))
    assert np.isclose(scales[1], np.percentile(np.arange(1.0, 31.0), 90.0))


def test_population_and_surrogate_threshold_are_configurable_defaults() -> None:
    config = SADEV2Config(
        max_evaluations=18, population_size=12, surrogate_min_samples=15, seed=6
    )
    result = SADEV2(_problem(feasible=True), config).optimize()
    assert result.evaluations == 18
    assert result.evaluation_metadata[12]["sampling_method"] == "random"
    assert result.evaluation_metadata[15]["sampling_method"] == "two_stage_rbf"


def test_rbf_waits_for_dimension_plus_one_unique_sites() -> None:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.sum(x**2, axis=1), -np.ones((len(x), 1))

    problem = CallableProblem(
        name="high_dimensional_geometry", dimension=30, n_constraints=1,
        lower_bounds=-1.0, upper_bounds=1.0, evaluator=evaluate,
    )
    result = SADEV2(
        problem,
        SADEV2Config(
            max_evaluations=36,
            population_size=30,
            surrogate_min_samples=30,
            seed=9,
        ),
    ).optimize()
    assert result.evaluation_metadata[30]["sampling_method"] == "random"
    assert result.evaluation_metadata[33]["sampling_method"] == "two_stage_rbf"
    assert result.history[1]["rbf_inactive_reason"] == "insufficient_unique_sites"
    assert result.history[2]["objective_rbf_rank"] == 64
    assert result.history[2]["rbf_system_size"] == 64


def test_v2_uses_exact_budget_is_reproducible_and_records_roles() -> None:
    config = SADEV2Config(max_evaluations=36, seed=8, save_candidate_pools=True)
    first = SADEV2(_problem(feasible=True), config).optimize()
    second = SADEV2(_problem(feasible=True), config).optimize()

    assert first.evaluations == 36
    np.testing.assert_allclose(first.archive_x, second.archive_x)
    sampled = first.evaluation_metadata[30:33]
    assert [row["acquisition_role"] for row in sampled] == [
        "global_feasible_max_ei", "boundary_exploration", "local_feasible_max_ei"
    ]
    assert all(row["acquisition_phase"] == "objective" for row in sampled)
    assert all(row["sampling_method"] == "two_stage_rbf" for row in sampled)
    assert all(len(row["predicted_constraints"]) == 2 for row in sampled)
    assert len(np.unique(first.archive_x[30:33], axis=0)) == 3
    normalized = (first.archive_x[30:33] + 1.0) / 2.0
    pairwise = np.linalg.norm(normalized[:, None] - normalized[None, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    assert np.min(pairwise) / np.sqrt(3) > config.distance_threshold
    assert first.candidate_pools is not None


def test_v2_stays_in_feasibility_stage_without_true_feasible_point() -> None:
    result = SADEV2(
        _problem(feasible=False), SADEV2Config(max_evaluations=33, seed=5)
    ).optimize()
    sampled = result.evaluation_metadata[30:]
    assert [row["acquisition_role"] for row in sampled] == [
        "global_min_cv", "boundary_exploration", "local_min_cv"
    ]
    assert all(row["acquisition_phase"] == "feasibility" for row in sampled)


def test_v2_switches_phase_only_after_a_true_feasible_evaluation() -> None:
    class TransitionSADE(SADEV2):
        evaluation_calls = 0

        def _evaluate(self, x):
            self.evaluation_calls += 1
            points = np.atleast_2d(np.asarray(x, dtype=float))
            objective = np.sum(points**2, axis=1)
            raw = 1.0 if self.evaluation_calls == 1 else -1.0
            constraints = np.full((len(points), 2), raw)
            return objective, constraints, np.maximum(constraints, 0.0)

    result = TransitionSADE(
        _problem(feasible=False), SADEV2Config(max_evaluations=36, seed=3)
    ).optimize()
    assert all(
        row["acquisition_phase"] == "feasibility"
        for row in result.evaluation_metadata[30:33]
    )
    assert all(
        row["acquisition_phase"] == "objective"
        for row in result.evaluation_metadata[33:36]
    )


def test_registry_constructs_sade_v2() -> None:
    optimizer, config = make_optimizer(
        "sade_v2", _problem(feasible=True),
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
    assert isinstance(optimizer, SADEV2)
    assert isinstance(config, SADEV2Config)


def test_v2_persistence_includes_acquisition_diagnostics(tmp_path) -> None:
    problem = _problem(feasible=True)
    problem.suite = "test"
    config = SADEV2Config(max_evaluations=33, save_candidate_pools=True)
    result = SADEV2(problem, config).optimize()
    save_run(
        tmp_path, algorithm="sade_v2", problem=problem, run=1, seed=1,
        config=config, result=result,
    )

    with gzip.open(tmp_path / "evaluations.csv.gz", "rt", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[30]["acquisition_role"] == "global_feasible_max_ei"
    assert rows[30]["predicted_constraints"].startswith("[")
    pools = np.load(tmp_path / "candidate_pools.npz")
    for field in (
        "predicted_objective", "predicted_constraints",
        "predicted_total_violation", "predicted_feasible", "objective_ei",
        "surrogate_uncertainty", "acquisition_phase",
    ):
        assert field in pools
