"""Adaptive penalty used by the original continuous SADE search."""

from __future__ import annotations

import numpy as np

from ...core.problem import FloatArray


def adaptive_penalty_fitness(
    objective: FloatArray,
    violations: FloatArray,
    generation: int,
    *,
    alpha: float = 0.05,
    weight_min: float = 0.1,
    weight_max: float = 100.0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return fitness, normalized objective, penalty, and constraint weights.

    This retains the useful part of ``SADE_TED/utils/penalty_fitness.py``:
    objectives and violations are normalized over the current archive, while a
    constraint gets a larger weight when few archived points satisfy it.
    """

    objective = np.asarray(objective, dtype=float).reshape(-1)
    violations = np.asarray(violations, dtype=float)
    if violations.ndim == 1:
        violations = violations.reshape(-1, 1)
    if len(objective) != len(violations):
        raise ValueError("Objective and violation arrays must have the same length.")

    n_points, n_constraints = violations.shape
    fitness = np.full(n_points, np.inf)
    objective_norm = np.full(n_points, np.inf)
    penalty = np.full(n_points, np.inf)
    weights = np.full(n_constraints, weight_max, dtype=float)

    valid = np.isfinite(objective) & np.all(np.isfinite(violations), axis=1)
    if not np.any(valid):
        return fitness, objective_norm, penalty, weights

    valid_objective = objective[valid]
    valid_violations = violations[valid]
    feasible_ratio = np.mean(valid_violations <= 0.0, axis=0)
    weights = np.clip(
        (1.0 + alpha * generation) / (feasible_ratio + 1e-6),
        weight_min,
        weight_max,
    )

    objective_span = np.ptp(valid_objective)
    if objective_span <= 1e-15:
        scaled_objective = np.zeros_like(valid_objective)
    else:
        scaled_objective = (
            valid_objective - np.min(valid_objective)
        ) / objective_span

    violation_scale = np.maximum(np.max(valid_violations, axis=0), 1e-12)
    scaled_penalty = np.sum(
        valid_violations / violation_scale * weights,
        axis=1,
    )

    objective_norm[valid] = scaled_objective
    penalty[valid] = scaled_penalty
    fitness[valid] = scaled_objective + scaled_penalty
    return fitness, objective_norm, penalty, weights
