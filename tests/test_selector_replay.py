from __future__ import annotations

import numpy as np

from sade_benchmark.diagnostics.selector_replay import _selector_index


def _scores() -> dict[str, np.ndarray]:
    return {
        "predicted_objective": np.asarray([8.0, 1.0, 3.0, 2.0]),
        "predicted_constraints": np.asarray(
            [[-1.0], [-1.0], [-1.0], [-1.0]]
        ),
        "surrogate_uncertainty": np.asarray([0.2, 0.1, 0.4, 0.3]),
        "robust_predicted_total_violation": np.ones(4),
        "robust_predicted_feasible": np.zeros(4, dtype=bool),
        "empirical_joint_feasibility_probability": np.asarray(
            [0.2, 0.8, 0.8, 0.1]
        ),
        "expected_constraint_violation": np.asarray([1.0, 0.1, 0.1, 2.0]),
        "continuous_feasibility_weight": np.ones(4),
        "constrained_expected_improvement": np.asarray(
            [10.0, 1.0, 2.0, 0.5]
        ),
    }


def test_probability_fallback_uses_probability_then_objective() -> None:
    index = _selector_index(
        "p90_then_probability",
        "objective",
        "global_robust_feasible_min_predicted_objective",
        _scores(),
        np.ones(1),
        0.0,
        0.05,
        0.2,
    )

    assert index == 1


def test_probability_fallback_preserves_available_p90_gate() -> None:
    scores = _scores()
    scores["robust_predicted_feasible"] = np.asarray(
        [True, False, True, False]
    )
    index = _selector_index(
        "p90_then_probability",
        "objective",
        "global_robust_feasible_min_predicted_objective",
        scores,
        np.ones(1),
        0.0,
        0.05,
        0.2,
    )

    assert index == 2
