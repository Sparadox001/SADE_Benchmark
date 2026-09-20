"""Common optimization result and trace containers.

Both SADE and DSI return this object so the experiment runner can persist and
compare them without knowing either algorithm's internal implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .problem import FloatArray


@dataclass
class OptimizationResult:
    """Final result plus the complete trace of one seeded run."""

    x: FloatArray
    objective: float
    constraints: FloatArray
    violation: FloatArray
    feasible: bool
    evaluations: int
    generations: int
    history: list[dict[str, Any]] = field(default_factory=list)
    archive_x: FloatArray | None = None
    archive_objective: FloatArray | None = None
    archive_constraints: FloatArray | None = None
    archive_violation: FloatArray | None = None
    evaluation_metadata: list[dict[str, Any]] = field(default_factory=list)
    population_history: list[dict[str, Any]] = field(default_factory=list)
    candidate_pools: list[dict[str, Any]] | None = None
    dynamic_constraint_state: dict[str, Any] | None = None
