"""Distance filtering, EI sampling, and discrete-version shortage fallback."""

from __future__ import annotations

import numpy as np

from ...core.distance import pairwise_distance
from ...core.problem import FloatArray
from .surrogate import CubicRBF, select_nsamples


def physical_scale(
    x: FloatArray,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
) -> FloatArray:
    """Normalize physical coordinates to [0, 1] for distance calculations."""

    return (np.asarray(x, dtype=float) - lower_bounds) / (upper_bounds - lower_bounds)


def calculate_distance(
    x_train: FloatArray,
    x_test: FloatArray,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    threshold: float = 1e-3,
) -> tuple[np.ndarray, FloatArray]:
    """Keep candidates whose normalized RMS distance exceeds ``threshold``."""

    train_normalized = physical_scale(x_train, lower_bounds, upper_bounds)
    test_normalized = physical_scale(x_test, lower_bounds, upper_bounds)
    distances = pairwise_distance(train_normalized, test_normalized)
    normalized_distances = distances / np.sqrt(train_normalized.shape[1])
    minimum_distance = np.min(normalized_distances, axis=0)
    valid_indices = np.flatnonzero(minimum_distance > threshold)
    return valid_indices, np.asarray(x_test, dtype=float)[valid_indices]


def distance_filter_log(
    archive_x: FloatArray,
    new_pop_values: FloatArray,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    threshold: float = 1e-3,
) -> tuple[FloatArray, np.ndarray]:
    """Deduplicate a candidate pool, then apply the original distance filter."""

    _, unique_indices = np.unique(new_pop_values, axis=0, return_index=True)
    unique_values = np.asarray(new_pop_values, dtype=float)[np.sort(unique_indices)]
    valid_indices, filtered = calculate_distance(
        archive_x,
        unique_values,
        lower_bounds,
        upper_bounds,
        threshold,
    )
    return filtered, valid_indices


def supplement_inds(
    selected_inds: FloatArray | None,
    candidate_pool: FloatArray,
    min_samples: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Retain filtered candidates and randomly fill a short quota.

    This is the continuous analogue of ``SADE_Discrete.supplement_inds``.
    Exact duplicates already present in ``selected_inds`` are avoided when the
    original pool contains enough alternatives.
    """

    pool = np.asarray(candidate_pool, dtype=float)
    if selected_inds is None:
        selected = np.empty((0, pool.shape[1]), dtype=float)
    else:
        selected = np.asarray(selected_inds, dtype=float).reshape(-1, pool.shape[1])
    if len(selected) >= min_samples:
        return selected[:min_samples]

    if len(selected):
        duplicate = np.any(np.all(pool[:, None, :] == selected[None, :, :], axis=2), axis=1)
        available = pool[~duplicate]
    else:
        available = pool
    lack = min_samples - len(selected)
    if len(available) == 0:
        return selected
    chosen = rng.choice(len(available), size=min(lack, len(available)), replace=False)
    supplement = available[chosen]
    return supplement if len(selected) == 0 else np.vstack((selected, supplement))


def sample_fun(
    new_pop_values: FloatArray,
    archive_fitness: FloatArray,
    para_rbf: CubicRBF | None,
    rng: np.random.Generator,
    n_samples: int = 10,
) -> FloatArray | None:
    """Randomly sample before RBF activation; otherwise maximize EI."""

    if new_pop_values is None or len(new_pop_values) < n_samples:
        return None
    if para_rbf is None:
        indices = rng.choice(len(new_pop_values), n_samples, replace=False)
        return new_pop_values[indices]
    y_min = float(np.min(archive_fitness[np.isfinite(archive_fitness)]))
    selected_samples, _ = select_nsamples(new_pop_values, para_rbf, y_min, n_samples)
    return selected_samples
