"""Small cubic radial-basis surrogate for expensive objective evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ...core.distance import pairwise_distance
from ...core.problem import FloatArray


@dataclass
class CubicRBF:
    """Cubic RBF interpolation with a linear polynomial tail."""

    lower_bounds: FloatArray | None = None
    upper_bounds: FloatArray | None = None
    regularization: float = 1e-10

    def _transform(self, x: FloatArray) -> FloatArray:
        x = np.asarray(x, dtype=float)
        if self.lower_bounds is None and self.upper_bounds is None:
            return x
        if self.lower_bounds is None or self.upper_bounds is None:
            raise ValueError("Both lower_bounds and upper_bounds are required.")
        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)
        return 2.0 * (x - lower) / (upper - lower) - 1.0

    def fit(self, x: FloatArray, y: FloatArray) -> "CubicRBF":
        x = self._transform(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("RBF inputs must have shapes (N, D) and (N,).")
        if len(x) < 2:
            raise ValueError("At least two samples are required for an RBF model.")

        # Exact duplicate sites make the interpolation system unnecessarily ill
        # conditioned. Keep the first occurrence; candidates are already filtered.
        _, unique_index = np.unique(np.round(x, decimals=14), axis=0, return_index=True)
        unique_index.sort()
        self.x_train_ = x[unique_index]
        y = y[unique_index]

        self.y_min_ = float(np.min(y))
        self.y_max_ = float(np.max(y))
        self.y_center_ = 0.5 * (self.y_min_ + self.y_max_)
        self.y_half_range_ = 0.5 * (self.y_max_ - self.y_min_)
        if self.y_half_range_ <= 1e-15:
            self.y_half_range_ = 1.0
            target = np.zeros_like(y)
        else:
            target = (y - self.y_center_) / self.y_half_range_

        n_samples, dimension = self.x_train_.shape
        phi = pairwise_distance(self.x_train_, self.x_train_) ** 3
        self.kernel_matrix_ = phi.copy()
        phi.flat[:: n_samples + 1] += self.regularization
        polynomial = np.column_stack((np.ones(n_samples), self.x_train_))
        system = np.block(
            [
                [phi, polynomial],
                [polynomial.T, np.zeros((dimension + 1, dimension + 1))],
            ]
        )
        rhs = np.concatenate((target, np.zeros(dimension + 1)))
        coefficients, *_ = np.linalg.lstsq(system, rhs, rcond=1e-12)
        self.radial_weights_ = coefficients[:n_samples]
        self.polynomial_weights_ = coefficients[n_samples:]
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        x = self._transform(x)
        radial = pairwise_distance(x, self.x_train_) ** 3
        polynomial = np.column_stack((np.ones(len(x)), x))
        prediction = (
            radial @ self.radial_weights_
            + polynomial @ self.polynomial_weights_
        )
        return prediction * self.y_half_range_ + self.y_center_

    def predict_with_uncertainty(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return RBF prediction and the distance-based variance used by SADE EI.

        The original SADE code derives a heuristic uncertainty from the cubic
        kernel matrix rather than from a probabilistic Gaussian process. We
        retain that acquisition behavior, but solve it with ``lstsq`` so a
        nearly singular archive does not terminate an optimization run.
        """

        raw_x = np.atleast_2d(np.asarray(x, dtype=float))
        transformed_x = self._transform(raw_x)
        radial = pairwise_distance(transformed_x, self.x_train_) ** 3
        prediction = self.predict(raw_x)
        kernel_solution, *_ = np.linalg.lstsq(
            self.kernel_matrix_ + self.regularization * np.eye(len(self.kernel_matrix_)),
            radial.T,
            rcond=1e-12,
        )
        normalized_variance = np.abs(
            np.sum(radial * kernel_solution.T, axis=1)
        )
        return prediction, normalized_variance


def expected_improvement(
    x: FloatArray,
    para_rbf: CubicRBF,
    y_min: float,
) -> FloatArray:
    """Return negative expected improvement, matching SADE_TED's minimizer API."""

    prediction, variance = para_rbf.predict_with_uncertainty(np.atleast_2d(x))
    sigma = np.sqrt(np.maximum(variance, 0.0))
    sigma = np.maximum(sigma, 1e-10)
    improvement = y_min - prediction
    standardized = improvement / sigma
    ei = improvement * norm.cdf(standardized) + sigma * norm.pdf(standardized)
    return -ei


def select_nsamples(
    x: FloatArray,
    para_rbf: CubicRBF,
    y_min: float,
    n_samples: int = 10,
) -> tuple[FloatArray, np.ndarray]:
    """Select candidate points with the largest expected improvement."""

    acquisition = expected_improvement(x, para_rbf, y_min)
    selected_indices = np.argsort(acquisition)[:n_samples]
    return np.asarray(x, dtype=float)[selected_indices], selected_indices
