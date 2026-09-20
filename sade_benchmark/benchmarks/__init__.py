"""Selected CEC inequality-constrained benchmark suites."""

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
from .registry import BENCHMARKS, make_problem

__all__ = [
    "BENCHMARKS",
    "CEC2006_PROBLEMS",
    "CEC2006Problem",
    "CEC2010_DIMENSIONS",
    "CEC2010_PROBLEMS",
    "CEC2010Problem",
    "CEC2017_DIMENSIONS",
    "CEC2017_PROBLEMS",
    "CEC2017Problem",
    "CEC2017_OBJECTIVE_CAP_DIMENSIONS",
    "CEC2017_OBJECTIVE_CAP_PROBLEMS",
    "CEC2017ObjectiveCapProblem",
    "CEC2017_FIRST_MEAN_DIMENSIONS",
    "CEC2017_FIRST_MEAN_PROBLEMS",
    "CEC2017FirstMeanProblem",
    "make_problem",
]
