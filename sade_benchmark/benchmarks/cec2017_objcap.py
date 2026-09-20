"""Frozen objective-cap variants of selected CEC2017 inequality problems."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ..core.problem import FloatArray
from .cec2017 import CEC2017Problem


_SPEC_PATH = Path(__file__).with_name("data") / "cec2017_objcap_median_t0.json"


@lru_cache(maxsize=1)
def _thresholds() -> dict[str, dict[int, float]]:
    with _SPEC_PATH.open("r", encoding="utf-8") as handle:
        specification = json.load(handle)
    if specification["version"] != "median_t0_v1":
        raise ValueError("Unsupported CEC2017 objective-cap specification version.")
    return {
        name: {int(dimension): float(value) for dimension, value in dimensions.items()}
        for name, dimensions in specification["thresholds"].items()
    }


CEC2017_OBJECTIVE_CAP_DIMENSIONS = {
    name: tuple(dimensions) for name, dimensions in _thresholds().items()
}
CEC2017_OBJECTIVE_CAP_PROBLEMS = tuple(CEC2017_OBJECTIVE_CAP_DIMENSIONS)


class CEC2017ObjectiveCapProblem:
    """Append the inequality ``f(x) - T0 <= 0`` without reevaluating ``f``."""

    def __init__(
        self,
        name: str,
        dimension: int,
        *,
        _threshold_map: dict[str, dict[int, float]] | None = None,
        _suite: str = "cec2017_objcap",
    ):
        threshold_map = _thresholds() if _threshold_map is None else _threshold_map
        normalized = str(name).strip().lower()
        if normalized not in threshold_map:
            raise ValueError(f"{name!r} has no frozen objective-cap specification.")
        if dimension not in threshold_map[normalized]:
            raise ValueError(
                f"{normalized} objective-cap dimensions are "
                f"{tuple(threshold_map[normalized])}."
            )
        self._base = CEC2017Problem(normalized, dimension)
        self.name = self._base.name
        self.suite = _suite
        self.dimension = self._base.dimension
        self.n_constraints = self._base.n_constraints + 1
        self.lower_bounds = self._base.lower_bounds.copy()
        self.upper_bounds = self._base.upper_bounds.copy()
        self.objective_cap = threshold_map[normalized][dimension]
        self.objective_constraint_index = self.n_constraints - 1

    def evaluate(self, x: ArrayLike) -> tuple[FloatArray, FloatArray]:
        objective, original_constraints = self._base.evaluate(x)
        cap_constraint = objective - self.objective_cap
        return objective, np.column_stack((original_constraints, cap_constraint))
