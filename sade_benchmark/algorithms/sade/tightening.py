"""Optional dynamic objective constraint for the benchmark SADE baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.problem import FloatArray
from .config import SADEConfig


class ObjectiveConstraintTightener:
    """Turn the objective into a progressively tightened virtual constraint.

    The real benchmark constraints remain unchanged.  Once enough members of
    the current search population satisfy both the real and active virtual
    constraints, the phase-wise running best objective is tracked. Stagnation
    activates or tightens ``f(x) <= active_limit`` using a fraction of the
    current feasible objective range.
    """

    def __init__(self, config: SADEConfig):
        self.enabled = bool(config.dynamic_constraint_tightening)
        self.trigger_feasible_count = (
            config.population_size
            if config.tightening_trigger_feasible_count is None
            else int(config.tightening_trigger_feasible_count)
        )
        self.keep_ratio = float(config.tightening_keep_ratio)
        self.patience = int(config.tightening_patience)
        self.improvement_tol = float(config.tightening_improvement_tol)
        self.max_tightens = int(config.tightening_max_tightens)
        self.min_tighten_ratio = float(config.tightening_min_ratio)
        self.dedupe_decimals = 15
        self.eps = 1e-12

        self.active_limit: float | None = None
        self.phase_id = 0
        self.tighten_count = 0
        self.phase_progress: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []

    @property
    def threshold_active(self) -> bool:
        return self.active_limit is not None

    def search_violations(
        self,
        objective: FloatArray,
        original_violations: FloatArray,
    ) -> FloatArray:
        """Append the active objective-bound violation used only by the search."""

        original = np.asarray(original_violations, dtype=float)
        if not self.enabled:
            return original

        objective = np.asarray(objective, dtype=float).reshape(-1)
        dynamic = np.zeros(len(objective), dtype=float)
        invalid = ~np.isfinite(objective)
        if self.active_limit is not None:
            limit = float(self.active_limit)
            dynamic = np.maximum(objective - limit, 0.0) / max(abs(limit), self.eps)
        dynamic[invalid] = np.inf
        return np.column_stack((original, dynamic))

    def maybe_tighten(
        self,
        objective: FloatArray,
        original_violations: FloatArray,
        population: np.ndarray,
        generation: int,
        archive_x: FloatArray | None = None,
    ) -> bool:
        """Update the objective limit after phase-running-best stagnation."""

        if not self.enabled:
            return False

        objective = np.asarray(objective, dtype=float).reshape(-1)
        population = np.asarray(population, dtype=int).reshape(-1)
        search_violations = self.search_violations(objective, original_violations)
        archive_feasible = np.all(search_violations <= 0.0, axis=1)
        archive_feasible &= np.isfinite(objective)
        if archive_x is None:
            archive_unique_feasible_count = int(np.count_nonzero(archive_feasible))
        else:
            feasible_x = np.asarray(archive_x, dtype=float)[archive_feasible]
            archive_unique_feasible_count = int(
                len(np.unique(np.round(feasible_x, self.dedupe_decimals), axis=0))
            )
        population_objective = objective[population]
        feasible = np.all(search_violations[population] <= 0.0, axis=1)
        feasible &= np.isfinite(population_objective)
        values = population_objective[feasible]

        if len(values) < self.trigger_feasible_count:
            self.phase_progress = []
            return False

        sorted_values = np.sort(values)
        current_best = float(sorted_values[0])
        current_worst = float(sorted_values[-1])
        range_limit = float(
            current_best + self.keep_ratio * (current_worst - current_best)
        )
        range_keep_count = int(np.count_nonzero(values <= range_limit))
        if self.phase_progress:
            previous_best = self.phase_progress[-1]["best_feasible_objective"]
            phase_best = min(float(previous_best), current_best)
        else:
            phase_best = current_best
        self.phase_progress.append(
            {
                "generation": int(generation),
                "phase_id": int(self.phase_id),
                "active_limit": self.active_limit,
                "current_feasible_count": int(len(values)),
                "current_best_objective": current_best,
                "current_worst_objective": current_worst,
                "current_range_limit": range_limit,
                "current_range_keep_count": range_keep_count,
                "best_feasible_objective": phase_best,
            }
        )

        if len(self.phase_progress) <= self.patience:
            return False
        old_best = float(
            self.phase_progress[-self.patience - 1]["best_feasible_objective"]
        )
        current_phase_best = float(
            self.phase_progress[-1]["best_feasible_objective"]
        )
        relative_improvement = (
            old_best - current_phase_best
        ) / max(abs(old_best), self.eps)
        if relative_improvement > self.improvement_tol:
            return False
        if self.tighten_count >= self.max_tightens:
            return False

        new_limit = range_limit
        if not np.isfinite(new_limit):
            return False
        old_limit = self.active_limit
        first_activation = old_limit is None
        if first_activation:
            limit_change = None
            tighten_ratio = None
        else:
            limit_change = float(old_limit) - new_limit
            tighten_ratio = limit_change / max(abs(float(old_limit)), self.eps)
            if limit_change <= 0.0 or tighten_ratio < self.min_tighten_ratio:
                return False

        old_phase_id = self.phase_id
        self.active_limit = new_limit
        self.phase_id += 1
        self.tighten_count += 1
        self.history.append(
            {
                "tighten_count": int(self.tighten_count),
                "old_phase_id": int(old_phase_id),
                "new_phase_id": int(self.phase_id),
                "generation": int(generation),
                "first_activation": bool(first_activation),
                "old_limit": old_limit,
                "new_limit": new_limit,
                "limit_change_abs": limit_change,
                "limit_change_ratio": tighten_ratio,
                "unique_feasible_count_before": archive_unique_feasible_count,
                "current_feasible_count_before": int(len(values)),
                "trigger_feasible_count": int(self.trigger_feasible_count),
                "keep_ratio": float(self.keep_ratio),
                "keep_count": int(range_keep_count),
                "limit_strategy": "current_feasible_objective_range_ratio",
                "best_feasible_objective_before": current_best,
                "current_objective_min_before": current_best,
                "current_objective_max_before": current_worst,
                "current_range_limit_before": new_limit,
                "relative_best_improvement": float(relative_improvement),
                "patience": int(self.patience),
                "improvement_tol": float(self.improvement_tol),
            }
        )
        self.phase_progress = []
        return True

    def state(self) -> dict[str, Any] | None:
        """Return a serializable trace, or ``None`` when the feature is disabled."""

        if not self.enabled:
            return None
        return {
            "enabled": True,
            "source": "objective",
            "direction": "minimize",
            "virtual_constraint": "objective <= active_limit",
            "threshold_active": self.threshold_active,
            "active_limit": self.active_limit,
            "phase_id": int(self.phase_id),
            "tighten_count": int(self.tighten_count),
            "max_tightens": int(self.max_tightens),
            "trigger_scope": "current_population",
            "progress_metric": "phase_running_best_objective",
            "limit_strategy": "current_feasible_objective_range_ratio",
            "trigger_feasible_count": int(self.trigger_feasible_count),
            "keep_ratio": float(self.keep_ratio),
            "patience": int(self.patience),
            "improvement_tol": float(self.improvement_tol),
            "min_tighten_ratio": float(self.min_tighten_ratio),
            "dedupe_decimals": int(self.dedupe_decimals),
            "history": list(self.history),
            "current_phase_progress": list(self.phase_progress),
        }
