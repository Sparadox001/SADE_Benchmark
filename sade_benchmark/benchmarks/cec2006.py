"""Selected pure-inequality CEC2006 problems via pymoo's tested definitions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from pymoo.problems import get_problem

from ..core.problem import FloatArray, as_2d_points


CEC2006_PROBLEMS = (
    "g01",
    "g02",
    "g04",
    "g06",
    "g07",
    "g08",
    "g09",
    "g10",
    "g12",
    "g16",
    "g18",
    "g19",
    "g24",
)

_KNOWN_OPTIMUM = {
    "g01": -15.0,
    "g02": -0.8036191042559,
    "g04": -30665.538671783332,
    "g06": -6961.81387558015,
    "g07": 24.30620906818,
    "g08": -0.0958250414180359,
    "g09": 680.630057374402,
    "g10": 7049.24802052867,
    "g12": -1.0,
    "g16": -1.90515525853479,
    "g18": -0.866025403784439,
    "g19": 32.6555929502463,
    "g24": -5.50801327159536,
}


def _normalize_name(name: str | int) -> str:
    if isinstance(name, int):
        normalized = f"g{name:02d}"
    else:
        text = str(name).strip().lower()
        if text.startswith("g"):
            text = text[1:]
        if not text.isdigit():
            raise ValueError(f"Invalid CEC2006 problem name: {name!r}.")
        normalized = f"g{int(text):02d}"
    if normalized not in CEC2006_PROBLEMS:
        allowed = ", ".join(CEC2006_PROBLEMS)
        raise ValueError(f"{normalized} is not in the pure-inequality subset: {allowed}.")
    return normalized


class CEC2006Problem:
    """Adapter that exposes a selected pymoo G problem through our interface."""

    def __init__(self, name: str | int):
        self.name = _normalize_name(name)
        self.suite = "cec2006"
        self.known_optimum = _KNOWN_OPTIMUM[self.name]
        self._problem = get_problem(f"g{int(self.name[1:])}")
        self.dimension = int(self._problem.n_var)
        self.n_constraints = int(self._problem.n_ieq_constr)
        self.lower_bounds = np.asarray(self._problem.xl, dtype=float).copy()
        self.upper_bounds = np.asarray(self._problem.xu, dtype=float).copy()

    def evaluate(self, x: ArrayLike) -> tuple[FloatArray, FloatArray]:
        points = as_2d_points(x, self.dimension)
        objective, constraints = self._problem.evaluate(
            points,
            return_values_of=["F", "G"],
        )
        objective = np.asarray(objective, dtype=float).reshape(-1)
        constraints = np.asarray(constraints, dtype=float)
        if constraints.ndim == 1:
            constraints = constraints.reshape(-1, 1)
        # pymoo uses the same six G04 inequalities in a different column order.
        # Keep DSI/CEC ordering so numerical cross-checks and logs line up exactly.
        if self.name == "g04":
            constraints = constraints[:, [1, 0, 3, 2, 5, 4]]
        return objective, constraints
