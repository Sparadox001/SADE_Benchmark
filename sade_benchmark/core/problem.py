"""Common interface for continuous inequality-constrained problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@runtime_checkable
class InequalityProblem(Protocol):
    """A minimization problem whose constraints use the convention ``g(x) <= 0``."""

    name: str
    dimension: int
    n_constraints: int
    lower_bounds: FloatArray
    upper_bounds: FloatArray

    def evaluate(self, x: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return objective values ``f`` and raw inequalities ``g`` for a batch."""


def as_2d_points(x: ArrayLike, dimension: int) -> FloatArray:
    """Convert a point or a batch of points to a validated ``(N, D)`` array."""

    points = np.asarray(x, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError(
            f"Expected points with shape (N, {dimension}), got {points.shape}."
        )
    return points


@dataclass
class CallableProblem:
    """Wrap an arbitrary vectorized Python evaluator in the common interface."""

    name: str
    dimension: int
    n_constraints: int
    lower_bounds: ArrayLike
    upper_bounds: ArrayLike
    evaluator: Callable[[FloatArray], tuple[ArrayLike, ArrayLike]]

    def __post_init__(self) -> None:
        self.lower_bounds = np.broadcast_to(
            np.asarray(self.lower_bounds, dtype=float), (self.dimension,)
        ).copy()
        self.upper_bounds = np.broadcast_to(
            np.asarray(self.upper_bounds, dtype=float), (self.dimension,)
        ).copy()
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError("Each lower bound must be strictly below its upper bound.")
        if self.n_constraints < 1:
            raise ValueError("SADE_Benchmark currently requires at least one inequality.")

    def evaluate(self, x: ArrayLike) -> tuple[FloatArray, FloatArray]:
        points = as_2d_points(x, self.dimension)
        objective, constraints = self.evaluator(points)
        objective = np.asarray(objective, dtype=float).reshape(-1)
        constraints = np.asarray(constraints, dtype=float)
        if constraints.ndim == 1:
            constraints = constraints.reshape(-1, 1)
        if objective.shape != (len(points),):
            raise ValueError(f"Objective must have shape ({len(points)},).")
        expected = (len(points), self.n_constraints)
        if constraints.shape != expected:
            raise ValueError(f"Constraints must have shape {expected}.")
        return objective, constraints
