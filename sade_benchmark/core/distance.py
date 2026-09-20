"""Small numerical helpers shared by optimization algorithms."""

from __future__ import annotations

import numpy as np

from .problem import FloatArray


def pairwise_distance(a: FloatArray, b: FloatArray) -> FloatArray:
    """Return the Euclidean distance matrix between two point sets."""

    squared = (
        np.sum(a * a, axis=1)[:, None]
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * a @ b.T
    )
    return np.sqrt(np.maximum(squared, 0.0))

