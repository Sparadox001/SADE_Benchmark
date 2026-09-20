"""Multi-output cubic RBFs and scale-consistent objective EI for SADE V2.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ...core.distance import pairwise_distance
from ...core.problem import FloatArray


@dataclass
class CubicRBFMultiOutput:
    """Cubic RBF with a linear tail and independent normalized outputs.

    One interpolation system is shared by all outputs. Inputs are always scaled
    with the fixed problem bounds; output columns are independently mapped to
    approximately ``[-1, 1]``.
    """

    lower_bounds: FloatArray
    upper_bounds: FloatArray
    regularization: float = 1e-10

    def _transform(self, x: FloatArray) -> FloatArray:
        values = np.atleast_2d(np.asarray(x, dtype=float))
        return 2.0 * (values - self.lower_bounds) / (
            self.upper_bounds - self.lower_bounds
        ) - 1.0

    def fit(self, x: FloatArray, y: FloatArray) -> "CubicRBFMultiOutput":
        sites = self._transform(x)
        targets = np.asarray(y, dtype=float)
        if targets.ndim == 1:
            targets = targets[:, None]
        if sites.ndim != 2 or targets.ndim != 2 or len(sites) != len(targets):
            raise ValueError("RBF inputs must have shapes (N, D) and (N, M).")
        if len(sites) < 2 or not np.all(np.isfinite(targets)):
            raise ValueError("RBF fitting requires at least two finite samples.")

        _, unique = np.unique(np.round(sites, decimals=14), axis=0, return_index=True)
        unique.sort()
        self.x_train_ = sites[unique]
        targets = targets[unique]
        self.y_min_ = np.min(targets, axis=0)
        self.y_max_ = np.max(targets, axis=0)
        self.y_center_ = 0.5 * (self.y_min_ + self.y_max_)
        raw_half_range = 0.5 * (self.y_max_ - self.y_min_)
        self.constant_output_ = raw_half_range <= 1e-15
        self.y_half_range_ = np.where(self.constant_output_, 1.0, raw_half_range)
        normalized = (targets - self.y_center_) / self.y_half_range_
        normalized[:, self.constant_output_] = 0.0

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
        rhs = np.vstack((normalized, np.zeros((dimension + 1, targets.shape[1]))))
        coefficients, _, self.rank_, singular_values = np.linalg.lstsq(
            system, rhs, rcond=1e-12
        )
        self.condition_number_ = (
            float(singular_values[0] / singular_values[-1])
            if len(singular_values) and singular_values[-1] > 0.0
            else np.inf
        )
        self.radial_weights_ = coefficients[:n_samples]
        self.polynomial_weights_ = coefficients[n_samples:]
        return self

    def predict_normalized(self, x: FloatArray) -> FloatArray:
        transformed = self._transform(x)
        radial = pairwise_distance(transformed, self.x_train_) ** 3
        polynomial = np.column_stack((np.ones(len(transformed)), transformed))
        prediction = radial @ self.radial_weights_ + polynomial @ self.polynomial_weights_
        prediction[:, self.constant_output_] = 0.0
        return prediction

    def predict(self, x: FloatArray) -> FloatArray:
        normalized = self.predict_normalized(x)
        prediction = normalized * self.y_half_range_ + self.y_center_
        prediction[:, self.constant_output_] = self.y_center_[self.constant_output_]
        return prediction

    def normalize_outputs(self, y: FloatArray) -> FloatArray:
        values = np.asarray(y, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        normalized = (values - self.y_center_) / self.y_half_range_
        normalized[:, self.constant_output_] = 0.0
        return normalized

    def geometric_uncertainty(self, x: FloatArray) -> FloatArray:
        """Return the original SADE cubic-kernel geometric variance proxy."""

        transformed = self._transform(x)
        radial = pairwise_distance(transformed, self.x_train_) ** 3
        kernel = self.kernel_matrix_ + self.regularization * np.eye(
            len(self.kernel_matrix_)
        )
        solution, *_ = np.linalg.lstsq(kernel, radial.T, rcond=1e-12)
        return np.abs(np.sum(radial * solution.T, axis=1))


def standardized_expected_improvement(
    predicted_normalized: FloatArray,
    incumbent_normalized: float,
    geometric_variance: FloatArray,
) -> FloatArray:
    """Expected improvement entirely in the objective's normalized output space."""

    mean = np.asarray(predicted_normalized, dtype=float).reshape(-1)
    variance = np.asarray(geometric_variance, dtype=float).reshape(-1)
    sigma = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1e-10)
    improvement = float(incumbent_normalized) - mean
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


@dataclass
class SADEV2Surrogates:
    """The one objective RBF and one multi-output constraint RBF used by V2.1."""

    lower_bounds: FloatArray
    upper_bounds: FloatArray
    regularization: float = 1e-10

    def fit(
        self, x: FloatArray, objective: FloatArray, constraints: FloatArray
    ) -> "SADEV2Surrogates":
        self.objective = CubicRBFMultiOutput(
            self.lower_bounds, self.upper_bounds, self.regularization
        ).fit(x, np.asarray(objective, dtype=float)[:, None])
        self.constraints = CubicRBFMultiOutput(
            self.lower_bounds, self.upper_bounds, self.regularization
        ).fit(x, constraints)
        return self

