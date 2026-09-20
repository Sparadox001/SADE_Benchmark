"""The six pure-inequality CEC2010 constrained problems."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ..core.problem import FloatArray, as_2d_points


CEC2010_PROBLEMS = ("c01", "c07", "c08", "c13", "c14", "c15")
CEC2010_DIMENSIONS = (10, 30)

_N_CONSTRAINTS = {"c01": 2, "c07": 1, "c08": 1, "c13": 3, "c14": 3, "c15": 3}
_BOUND = {"c01": (0.0, 10.0), "c07": (-140.0, 140.0), "c08": (-140.0, 140.0),
          "c13": (-500.0, 500.0), "c14": (-1000.0, 1000.0), "c15": (-1000.0, 1000.0)}
_DATA_FILE = Path(__file__).with_name("data") / "cec2010.npz"


def _normalize_name(name: str | int) -> str:
    if isinstance(name, int):
        normalized = f"c{name:02d}"
    else:
        text = str(name).strip().lower()
        if text.startswith("c"):
            text = text[1:]
        if not text.isdigit():
            raise ValueError(f"Invalid CEC2010 problem name: {name!r}.")
        normalized = f"c{int(text):02d}"
    if normalized not in CEC2010_PROBLEMS:
        allowed = ", ".join(CEC2010_PROBLEMS)
        raise ValueError(f"{normalized} is not in the pure-inequality subset: {allowed}.")
    return normalized


@lru_cache(maxsize=1)
def _data() -> dict[str, FloatArray]:
    with np.load(_DATA_FILE) as loaded:
        return {key: np.asarray(loaded[key], dtype=float) for key in loaded.files}


class CEC2010Problem:
    def __init__(self, name: str | int, dimension: int):
        self.name = _normalize_name(name)
        if dimension not in CEC2010_DIMENSIONS:
            raise ValueError(f"CEC2010 dimension must be one of {CEC2010_DIMENSIONS}.")
        self.suite = "cec2010"
        self.dimension = int(dimension)
        self.n_constraints = _N_CONSTRAINTS[self.name]
        lower, upper = _BOUND[self.name]
        self.lower_bounds = np.full(self.dimension, lower)
        self.upper_bounds = np.full(self.dimension, upper)
        self._shift = _data()[f"o{int(self.name[1:]):02d}"][: self.dimension]
        self._rotation = None
        if self.name in {"c08", "c15"}:
            self._rotation = _data()[f"rotation_{self.dimension}"]

    def evaluate(self, x: ArrayLike) -> tuple[FloatArray, FloatArray]:
        points = as_2d_points(x, self.dimension)
        y = points - self._shift

        if self.name == "c01":
            constraints = np.column_stack(
                (0.75 - np.prod(y, axis=1), np.sum(y, axis=1) - 7.5 * self.dimension)
            )
            numerator = np.abs(
                np.sum(np.cos(y) ** 4, axis=1)
                - 2.0 * np.prod(np.cos(y) ** 2, axis=1)
            )
            denominator = np.sqrt(
                np.sum(np.arange(1, self.dimension + 1) * y**2, axis=1)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                objective = -numerator / denominator

        elif self.name in {"c07", "c08"}:
            rosenbrock_input = y + 1.0
            objective = _rosenbrock(rosenbrock_input)
            constraint_input = y if self._rotation is None else y @ self._rotation
            constraint = (
                0.5
                - np.exp(-0.1 * np.sqrt(np.mean(constraint_input**2, axis=1)))
                - 3.0 * np.exp(np.mean(np.cos(0.1 * constraint_input), axis=1))
                + np.e
            )
            constraints = constraint[:, None]

        elif self.name == "c13":
            objective = np.mean(-y * np.sin(np.sqrt(np.abs(y))), axis=1)
            griewank = (
                np.sum(y**2, axis=1) / 4000.0
                - np.prod(np.cos(y / np.sqrt(np.arange(1, self.dimension + 1))), axis=1)
                + 1.0
            )
            constraints = np.column_stack(
                (
                    -50.0 + np.sum(y**2, axis=1) / (100.0 * self.dimension),
                    50.0 / self.dimension * np.sum(np.sin(0.02 * np.pi * y), axis=1),
                    75.0 - 50.0 * griewank,
                )
            )

        elif self.name in {"c14", "c15"}:
            rosenbrock_input = y + 1.0
            objective = _rosenbrock(rosenbrock_input)
            constraint_input = y if self._rotation is None else y @ self._rotation
            root = np.sqrt(np.abs(constraint_input))
            constraints = np.column_stack(
                (
                    np.sum(-constraint_input * np.cos(root), axis=1) - self.dimension,
                    np.sum(constraint_input * np.cos(root), axis=1) - self.dimension,
                    np.sum(constraint_input * np.sin(root), axis=1) - 10.0 * self.dimension,
                )
            )
        else:  # pragma: no cover - guarded by constructor
            raise AssertionError(self.name)

        return np.asarray(objective, dtype=float), np.asarray(constraints, dtype=float)


def _rosenbrock(x: FloatArray) -> FloatArray:
    return np.sum(
        100.0 * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1.0) ** 2,
        axis=1,
    )
