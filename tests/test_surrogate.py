from __future__ import annotations

import numpy as np

from sade_benchmark.algorithms.sade.surrogate import (
    CubicRBF,
    expected_improvement,
    select_nsamples,
)


def test_expected_improvement_is_explicit_and_finite() -> None:
    x_train = np.linspace(0.0, 1.0, 8)[:, None]
    y_train = (x_train[:, 0] - 0.25) ** 2
    model = CubicRBF(
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([1.0]),
    ).fit(x_train, y_train)
    candidates = np.array([[0.1], [0.3], [0.8]])

    acquisition = expected_improvement(candidates, model, float(np.min(y_train)))
    selected, indices = select_nsamples(
        candidates,
        model,
        float(np.min(y_train)),
        n_samples=2,
    )

    assert acquisition.shape == (3,)
    assert np.all(np.isfinite(acquisition))
    np.testing.assert_array_equal(selected, candidates[indices])
    np.testing.assert_array_equal(indices, np.argsort(acquisition)[:2])
