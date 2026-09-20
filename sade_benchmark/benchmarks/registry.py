"""Factory and inventory for the selected benchmark subsets."""

from __future__ import annotations

from ..core.problem import InequalityProblem
from .cec2006 import CEC2006_PROBLEMS, CEC2006Problem
from .cec2010 import CEC2010_DIMENSIONS, CEC2010_PROBLEMS, CEC2010Problem
from .cec2017 import CEC2017_DIMENSIONS, CEC2017_PROBLEMS, CEC2017Problem
from .cec2017_objcap import (
    CEC2017_OBJECTIVE_CAP_DIMENSIONS,
    CEC2017_OBJECTIVE_CAP_PROBLEMS,
    CEC2017ObjectiveCapProblem,
)
from .cec2017_objcap_first_mean import (
    CEC2017_FIRST_MEAN_DIMENSIONS,
    CEC2017_FIRST_MEAN_PROBLEMS,
    CEC2017FirstMeanProblem,
)


BENCHMARKS = {
    "cec2006": {"problems": CEC2006_PROBLEMS, "dimensions": "fixed"},
    "cec2010": {"problems": CEC2010_PROBLEMS, "dimensions": CEC2010_DIMENSIONS},
    "cec2017": {"problems": CEC2017_PROBLEMS, "dimensions": CEC2017_DIMENSIONS},
    "cec2017_objcap": {
        "problems": CEC2017_OBJECTIVE_CAP_PROBLEMS,
        "dimensions": (10, 30),
        "dimensions_by_problem": CEC2017_OBJECTIVE_CAP_DIMENSIONS,
        "included_in_all": False,
    },
    "cec2017_objcap_first_mean": {
        "problems": CEC2017_FIRST_MEAN_PROBLEMS,
        "dimensions": (10, 30),
        "dimensions_by_problem": CEC2017_FIRST_MEAN_DIMENSIONS,
        "included_in_all": False,
    },
}


def make_problem(
    suite: str,
    problem: str | int,
    dimension: int | None = None,
) -> InequalityProblem:
    suite = suite.strip().lower()
    if suite == "cec2006":
        if dimension is not None:
            raise ValueError("CEC2006 dimensions are fixed; omit dimension.")
        return CEC2006Problem(problem)
    if suite == "cec2010":
        if dimension is None:
            raise ValueError("CEC2010 requires dimension=10 or dimension=30.")
        return CEC2010Problem(problem, dimension)
    if suite == "cec2017":
        if dimension is None:
            raise ValueError("CEC2017 requires dimension=10, 30, 50, or 100.")
        return CEC2017Problem(problem, dimension)
    if suite == "cec2017_objcap":
        if dimension is None:
            raise ValueError("CEC2017 objective-cap problems require a dimension.")
        return CEC2017ObjectiveCapProblem(str(problem), dimension)
    if suite == "cec2017_objcap_first_mean":
        if dimension is None:
            raise ValueError("CEC2017 first-mean objective-cap problems require a dimension.")
        return CEC2017FirstMeanProblem(str(problem), dimension)
    raise ValueError(f"Unknown suite {suite!r}; choose from {tuple(BENCHMARKS)}.")
