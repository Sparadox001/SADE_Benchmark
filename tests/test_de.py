from __future__ import annotations

import numpy as np

from sade_benchmark.algorithms.sade.de import generate_trial_candidates_group, getF_CR


def test_original_f_cr_ranges_are_retained() -> None:
    rng = np.random.default_rng(9)
    draws = np.asarray([getF_CR(rng) for _ in range(1000)])
    assert np.all((0.2 <= draws[:, 0]) & (draws[:, 0] <= 1.0))
    assert np.all((0.1 <= draws[:, 1]) & (draws[:, 1] <= 1.0))


def test_six_trials_are_generated_for_each_target() -> None:
    rng = np.random.default_rng(10)
    population = rng.uniform(-1.0, 1.0, size=(10, 4))
    objective = np.sum(population**2, axis=1)
    violations = np.maximum(np.column_stack((population[:, 0], population[:, 1])), 0.0)
    penalty = np.sum(violations, axis=1)
    fitness = objective + penalty

    trials = generate_trial_candidates_group(
        population,
        target_index=0,
        objective=objective,
        penalty=penalty,
        fitness=fitness,
        violations=violations,
        lower_bounds=np.full(4, -1.0),
        upper_bounds=np.full(4, 1.0),
        rng=rng,
        num_trials=6,
        p_pbest=0.2,
    )

    assert trials.shape == (6, 4)
    assert np.all(trials >= -1.0)
    assert np.all(trials <= 1.0)
