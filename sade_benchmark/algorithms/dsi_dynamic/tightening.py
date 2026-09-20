"""Dynamic objective constraint used only by the DSI search state."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.problem import FloatArray
from .config import DSIDynamicConfig


class DSIObjectiveConstraintTightener:
    """Maintain ``f(x) <= active_limit`` using the SADE tightening rule."""

    def __init__(self, config: DSIDynamicConfig):
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
        self.active_limit: float | None = None
        self.phase_id = 0
        self.tighten_count = 0
        self.phase_progress: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.eps = 1e-12
        self.dedupe_decimals = 15

    @property
    def threshold_active(self) -> bool:
        return self.active_limit is not None

    def objective_violation(self, objective: FloatArray) -> FloatArray:
        """Return the normalized violation of the active objective limit."""

        values = np.asarray(objective, dtype=float).reshape(-1)
        violation = np.zeros(len(values), dtype=float)
        if self.active_limit is not None:
            limit = float(self.active_limit)
            violation = np.maximum(values - limit, 0.0) / max(abs(limit), self.eps)
        violation[~np.isfinite(values)] = np.inf
        return violation

    def search_total_violation(
        self,
        objective: FloatArray,
        physical_total_violation: FloatArray,
    ) -> FloatArray:
        """Append the objective violation to DSI's physical total CV."""

        physical = np.asarray(physical_total_violation, dtype=float).reshape(-1)
        if not self.enabled:
            return physical.copy()
        return physical + self.objective_violation(objective)

    def maybe_tighten(
        self,
        archive_x: FloatArray,
        archive_objective: FloatArray,
        archive_search_cv: FloatArray,
        population_objective: FloatArray,
        population_search_cv: FloatArray,
        generation: int,
    ) -> bool:
        """Tighten after the current DSI search population stagnates."""

        if not self.enabled:
            return False

        objective = np.asarray(population_objective, dtype=float).reshape(-1)
        search_cv = np.asarray(population_search_cv, dtype=float).reshape(-1)
        feasible = (search_cv == 0.0) & np.isfinite(objective)
        values = objective[feasible]
        if len(values) < self.trigger_feasible_count:
            self.phase_progress = []
            return False

        current_best = float(np.min(values))
        current_worst = float(np.max(values))
        range_limit = float(
            current_best + self.keep_ratio * (current_worst - current_best)
        )
        keep_count = int(np.count_nonzero(values <= range_limit))
        phase_best = (
            min(
                float(self.phase_progress[-1]["best_feasible_objective"]),
                current_best,
            )
            if self.phase_progress
            else current_best
        )
        self.phase_progress.append(
            {
                "generation": int(generation),
                "phase_id": int(self.phase_id),
                "active_limit": self.active_limit,
                "current_feasible_count": int(len(values)),
                "current_best_objective": current_best,
                "current_worst_objective": current_worst,
                "current_range_limit": range_limit,
                "current_range_keep_count": keep_count,
                "best_feasible_objective": phase_best,
            }
        )

        if len(self.phase_progress) <= self.patience:
            return False
        old_best = float(
            self.phase_progress[-self.patience - 1]["best_feasible_objective"]
        )
        relative_improvement = (old_best - phase_best) / max(abs(old_best), self.eps)
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

        archive_feasible = np.asarray(archive_search_cv, dtype=float) == 0.0
        archive_feasible &= np.isfinite(archive_objective)
        unique_feasible_count = int(
            len(
                np.unique(
                    np.round(np.asarray(archive_x)[archive_feasible], self.dedupe_decimals),
                    axis=0,
                )
            )
        )
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
                "unique_feasible_count_before": unique_feasible_count,
                "current_feasible_count_before": int(len(values)),
                "trigger_feasible_count": int(self.trigger_feasible_count),
                "keep_ratio": self.keep_ratio,
                "keep_count": keep_count,
                "limit_strategy": "current_feasible_objective_range_ratio",
                "best_feasible_objective_before": current_best,
                "current_objective_min_before": current_best,
                "current_objective_max_before": current_worst,
                "current_range_limit_before": new_limit,
                "relative_best_improvement": float(relative_improvement),
                "patience": self.patience,
            }
        )
        self.phase_progress = []
        return True

    def state(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "enabled": self.enabled,
            "source": "objective",
            "direction": "minimize",
            "virtual_constraint": "objective <= active_limit",
            "threshold_active": self.threshold_active,
            "active_limit": self.active_limit,
            "phase_id": int(self.phase_id),
            "tighten_count": int(self.tighten_count),
            "max_tightens": self.max_tightens,
            "trigger_scope": "current_population",
            "progress_metric": "phase_running_best_objective",
            "limit_strategy": "current_feasible_objective_range_ratio",
            "trigger_feasible_count": self.trigger_feasible_count,
            "keep_ratio": self.keep_ratio,
            "patience": self.patience,
            "improvement_tol": self.improvement_tol,
            "min_tighten_ratio": self.min_tighten_ratio,
            "objective_violation_scale": "max(abs(active_limit), 1e-12)",
            "history": list(self.history),
            "current_phase_progress": list(self.phase_progress),
        }
