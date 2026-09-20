"""Cubic RBF model used by the MATLAB DSI implementation."""

from __future__ import annotations

import numpy as np

from ...core.distance import pairwise_distance
from ...core.problem import FloatArray


class DSICubicRBF:
    """Multi-output cubic RBF with DSI's sample-based normalization."""

    def fit(self, x: FloatArray, y: FloatArray) -> "DSICubicRBF":
        raw_x = np.asarray(x, dtype=float)
        raw_y = np.asarray(y, dtype=float)
        if raw_y.ndim == 1:
            raw_y = raw_y[:, None]
        if raw_x.ndim != 2 or raw_y.ndim != 2 or len(raw_x) != len(raw_y):
            raise ValueError("RBF inputs must have shapes (N, D) and (N, M).")

        self.x_min_ = np.min(raw_x, axis=0)
        self.x_max_ = np.max(raw_x, axis=0)
        self.y_min_ = np.min(raw_y, axis=0)
        self.y_max_ = np.max(raw_y, axis=0)
        self.nodes_ = self._normalize(raw_x, self.x_min_, self.x_max_)
        target = self._normalize(raw_y, self.y_min_, self.y_max_)

        n_samples, dimension = self.nodes_.shape
        phi = pairwise_distance(self.nodes_, self.nodes_) ** 3
        polynomial = np.column_stack((np.ones(n_samples), self.nodes_))
        system = np.block(
            [
                [phi, polynomial],
                [polynomial.T, np.zeros((dimension + 1, dimension + 1))],
            ]
        )
        rhs = np.vstack((target, np.zeros((dimension + 1, target.shape[1]))))
        coefficients, *_ = np.linalg.lstsq(system, rhs, rcond=None)
        self.alpha_ = coefficients[:n_samples]
        self.beta_ = coefficients[n_samples:]
        self.kernel_matrix_ = phi
        return self

    @staticmethod
    def _normalize(values: FloatArray, low: FloatArray, high: FloatArray) -> FloatArray:
        normalized = np.asarray(values, dtype=float).copy()
        changing = high != low
        normalized[:, changing] = (
            2.0 * (normalized[:, changing] - low[changing])
            / (high[changing] - low[changing])
            - 1.0
        )
        return normalized

    def predict(self, x: FloatArray) -> FloatArray:
        normalized = self._normalize(
            np.atleast_2d(np.asarray(x, dtype=float)), self.x_min_, self.x_max_
        )
        radial = pairwise_distance(normalized, self.nodes_) ** 3
        polynomial = np.column_stack((np.ones(len(normalized)), normalized))
        prediction = radial @ self.alpha_ + polynomial @ self.beta_
        changing = self.y_max_ != self.y_min_
        prediction[:, changing] = (
            (self.y_max_[changing] - self.y_min_[changing])
            * (prediction[:, changing] + 1.0)
            / 2.0
            + self.y_min_[changing]
        )
        return prediction

    def predict_with_uncertainty(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        points = np.atleast_2d(np.asarray(x, dtype=float))
        normalized = self._normalize(points, self.x_min_, self.x_max_)
        radial = pairwise_distance(normalized, self.nodes_) ** 3
        solution, *_ = np.linalg.lstsq(self.kernel_matrix_, radial.T, rcond=None)
        sigma = np.abs(np.diag(-(radial @ solution)))
        return self.predict(points), sigma

    def influence_indices(self, x: FloatArray) -> tuple[np.ndarray, np.ndarray]:
        """Return DSI's minimum/maximum radial-contribution node indices."""

        points = np.atleast_2d(np.asarray(x, dtype=float))
        normalized = self._normalize(points, self.x_min_, self.x_max_)
        radial = pairwise_distance(normalized, self.nodes_) ** 3
        minimum = np.empty(len(points), dtype=int)
        maximum = np.empty(len(points), dtype=int)
        for i in range(len(points)):
            contribution = np.sum(radial[i, :, None] * self.alpha_, axis=1)
            minimum[i] = int(np.argmin(contribution))
            maximum[i] = int(np.argmax(contribution))
        return minimum, maximum
