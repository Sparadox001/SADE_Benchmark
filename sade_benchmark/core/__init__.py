"""Algorithm-independent problem and result contracts."""

from .problem import CallableProblem, InequalityProblem
from .result import OptimizationResult

__all__ = ["CallableProblem", "InequalityProblem", "OptimizationResult"]

