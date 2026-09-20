"""Nine retained pure-inequality CEC2017 constrained problems.

CEC2017 C12 and C21 are deliberately excluded by project decision. The shared
``Function2.mat`` data are also the shift/rotation data used by C13 onward in
the reference MATLAB implementation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.io import loadmat

from ..core.problem import FloatArray, as_2d_points


CEC2017_PROBLEMS = (
    "c01",
    "c02",
    "c04",
    "c05",
    "c13",
    "c19",
    "c20",
    "c22",
    "c28",
)
CEC2017_DIMENSIONS = (10, 30, 50, 100)

_N_CONSTRAINTS = {
    "c01": 1,
    "c02": 1,
    "c04": 2,
    "c05": 2,
    "c13": 3,
    "c19": 2,
    "c20": 2,
    "c22": 3,
    "c28": 2,
}
_BOUND = {
    "c01": (-100.0, 100.0),
    "c02": (-100.0, 100.0),
    "c04": (-10.0, 10.0),
    "c05": (-10.0, 10.0),
    "c13": (-100.0, 100.0),
    "c19": (-50.0, 50.0),
    "c20": (-100.0, 100.0),
    "c22": (-100.0, 100.0),
    "c28": (-50.0, 50.0),
}
_MAT_FILE = {
    "c01": "Function1.mat",
    "c02": "Function2.mat",
    "c04": "Function4.mat",
    "c05": "Function5.mat",
    "c13": "Function2.mat",
    "c19": "Function2.mat",
    "c20": "Function2.mat",
    "c22": "Function2.mat",
    "c28": "Function2.mat",
}
_DATA_DIR = Path(__file__).with_name("data")


def _normalize_name(name: str | int) -> str:
    if isinstance(name, int):
        normalized = f"c{name:02d}"
    else:
        text = str(name).strip().lower()
        if text.startswith("c"):
            text = text[1:]
        if not text.isdigit():
            raise ValueError(f"Invalid CEC2017 problem name: {name!r}.")
        normalized = f"c{int(text):02d}"
    if normalized not in CEC2017_PROBLEMS:
        allowed = ", ".join(CEC2017_PROBLEMS)
        raise ValueError(f"{normalized} is not in the retained subset: {allowed}.")
    return normalized


@lru_cache(maxsize=None)
def _mat_data(filename: str) -> dict[str, FloatArray]:
    raw = loadmat(_DATA_DIR / filename)
    return {
        key: np.asarray(value, dtype=float)
        for key, value in raw.items()
        if not key.startswith("__")
    }


class CEC2017Problem:
    def __init__(self, name: str | int, dimension: int):
        self.name = _normalize_name(name)
        if dimension not in CEC2017_DIMENSIONS:
            raise ValueError(f"CEC2017 dimension must be one of {CEC2017_DIMENSIONS}.")
        self.suite = "cec2017"
        self.dimension = int(dimension)
        self.n_constraints = _N_CONSTRAINTS[self.name]
        lower, upper = _BOUND[self.name]
        self.lower_bounds = np.full(self.dimension, lower)
        self.upper_bounds = np.full(self.dimension, upper)

        data = _mat_data(_MAT_FILE[self.name])
        self._shift = data["o"].reshape(-1)[: self.dimension]
        self._rotation = None
        self._rotation_2 = None
        if self.name in {"c02", "c22", "c28"}:
            self._rotation = data[f"M_{self.dimension}"]
        elif self.name == "c05":
            self._rotation = data[f"M1_{self.dimension}"]
            self._rotation_2 = data[f"M2_{self.dimension}"]

    def evaluate(self, x: ArrayLike) -> tuple[FloatArray, FloatArray]:
        points = as_2d_points(x, self.dimension)
        y = points - self._shift

        if self.name in {"c01", "c02"}:
            objective = np.sum(np.cumsum(y, axis=1) ** 2, axis=1)
            constraint_input = y if self._rotation is None else y @ self._rotation.T
            constraint = np.sum(
                constraint_input**2
                - 5000.0 * np.cos(0.1 * np.pi * constraint_input)
                - 4000.0,
                axis=1,
            )
            constraints = constraint[:, None]

        elif self.name == "c04":
            objective = _rastrigin(y)
            constraints = np.column_stack(
                (np.sum(-y * np.sin(2.0 * y), axis=1), np.sum(y * np.sin(y), axis=1))
            )

        elif self.name == "c05":
            objective = _rosenbrock(y)
            z1 = y @ self._rotation.T
            z2 = y @ self._rotation_2.T
            constraints = np.column_stack(
                (
                    np.sum(z1**2 - 50.0 * np.cos(2.0 * np.pi * z1) - 40.0, axis=1),
                    np.sum(z2**2 - 50.0 * np.cos(2.0 * np.pi * z2) - 40.0, axis=1),
                )
            )

        elif self.name in {"c13", "c22"}:
            z = y if self._rotation is None else y @ self._rotation.T
            objective = _rosenbrock(z)
            constraints = np.column_stack(
                (
                    _rastrigin(z) - 100.0,
                    np.sum(z, axis=1) - 2.0 * self.dimension,
                    -np.sum(z, axis=1) + 5.0,
                )
            )

        elif self.name in {"c19", "c28"}:
            z = y if self._rotation is None else y @ self._rotation.T
            objective, constraints = _cec19_components(z)

        elif self.name == "c20":
            following = np.roll(y, shift=-1, axis=1)
            radius = np.sqrt(y**2 + following**2)
            objective = np.sum(
                0.5
                + (np.sin(radius) ** 2 - 0.5) / (1.0 + 0.001 * radius) ** 2,
                axis=1,
            )
            summed = np.sum(y, axis=1)
            constraints = np.column_stack(
                (
                    np.cos(summed) ** 2 - 0.25 * np.cos(summed) - 0.125,
                    np.exp(np.cos(summed)) - np.exp(0.25),
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


def _rastrigin(x: FloatArray) -> FloatArray:
    return np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)


def _cec19_components(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    objective = np.sum(np.abs(x) ** 0.5 + 2.0 * np.sin(x**3), axis=1)
    adjacent_radius = np.sqrt(x[:, :-1] ** 2 + x[:, 1:] ** 2)
    g1 = (
        np.sum(-10.0 * np.exp(-0.2 * adjacent_radius), axis=1)
        + (x.shape[1] - 1) * 10.0 / np.exp(-5.0)
    )
    g2 = np.sum(np.sin(2.0 * x) ** 2, axis=1) - 0.5 * x.shape[1]
    return objective, np.column_stack((g1, g2))
