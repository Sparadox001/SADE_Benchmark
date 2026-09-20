"""Shared initial experimental designs for benchmark algorithms."""

from __future__ import annotations

import numpy as np

from .distance import pairwise_distance
from .problem import FloatArray


def latin_hypercube_maximin(
    n_points: int,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    rng: np.random.Generator,
    *,
    attempts: int = 5,
) -> FloatArray:
    """Return DSI's best-of-five maximin Latin hypercube design."""

    lower = np.asarray(lower_bounds, dtype=float).reshape(-1)
    upper = np.asarray(upper_bounds, dtype=float).reshape(-1)
    if n_points < 2:
        raise ValueError("n_points must be at least 2 for a maximin design.")
    if attempts < 1:
        raise ValueError("attempts must be positive.")
    if lower.shape != upper.shape or np.any(lower >= upper):
        raise ValueError("Initial-design bounds are inconsistent.")

    dimension = len(lower)
    best: FloatArray | None = None
    best_score = -np.inf
    for _ in range(attempts):
        sample = np.empty((n_points, dimension), dtype=float)
        for column in range(dimension):
            ranks = rng.permutation(n_points) + 1
            sample[:, column] = (ranks - rng.random(n_points)) / n_points
        distances = pairwise_distance(sample, sample)
        np.fill_diagonal(distances, np.inf)
        score = float(np.min(distances))
        if score > best_score:
            best = sample
            best_score = score
    assert best is not None
    return lower + best * (upper - lower)

