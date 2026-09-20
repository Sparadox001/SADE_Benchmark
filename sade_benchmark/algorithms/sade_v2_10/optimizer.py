"""SADE V2.10 with a fixed coarse/fine local-pool split."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sade_v2_9.optimizer import SADEV2_9
from ...core.problem import FloatArray, InequalityProblem
from ...core.result import OptimizationResult
from .config import SADEV2_10Config


class SADEV2_10(SADEV2_9):
    """Split the unchanged local pool evenly across two spatial scales.

    Half of the local candidates retain V2.9's radius and changed-coordinate
    count.  The other half uses one tenth of that radius, floored at 0.01 of
    the variable span, and changes exactly one coordinate.  Every acquisition
    and population rule remains inherited from V2.9.
    """

    _FINE_RADIUS_RATIO = 0.1
    _FINE_RADIUS_FLOOR = 0.01

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_10Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_10Config())
        self.config: SADEV2_10Config
        self._last_coarse_local_pool = np.empty((0, self.dimension))
        self._last_fine_local_pool = np.empty((0, self.dimension))

    def optimize(self) -> OptimizationResult:
        self._last_coarse_local_pool = np.empty((0, self.dimension))
        self._last_fine_local_pool = np.empty((0, self.dimension))
        return super().optimize()

    def _fine_local_radius(self, generation: int) -> float:
        return max(
            self._FINE_RADIUS_FLOOR,
            self._FINE_RADIUS_RATIO * self._local_radius(generation),
        )

    def local_search(
        self,
        best_vec: FloatArray,
        current_gen: int,
        *,
        n_local: int,
    ) -> FloatArray:
        """Return the fixed 50/50 coarse/fine local candidate pool."""

        n_fine = n_local // 2
        n_coarse = n_local - n_fine
        coarse = super().local_search(
            best_vec, current_gen, n_local=n_coarse
        )

        fine_radius = self._fine_local_radius(current_gen)
        fine = np.repeat(
            np.asarray(best_vec, dtype=float)[None, :], n_fine, axis=0
        )
        for candidate in fine:
            changed = int(self.rng.integers(self.dimension))
            noise = self.rng.uniform(-1.0, 1.0) * (
                fine_radius * self.span[changed]
            )
            candidate[changed] = np.clip(
                candidate[changed] + noise,
                self.lower[changed],
                self.upper[changed],
            )

        self._last_coarse_local_pool = coarse.copy()
        self._last_fine_local_pool = fine.copy()
        return np.vstack((coarse, fine))

    @staticmethod
    def _point_in_pool(point: FloatArray, pool: FloatArray) -> bool:
        if not len(pool):
            return False
        return bool(np.any(np.all(pool == point, axis=1)))

    @staticmethod
    def _slice_pool_record(
        record: dict[str, Any], start: int, end: int, stage: str
    ) -> dict[str, Any]:
        """Slice every candidate-aligned trace field without changing scores."""

        total = len(record["x"])
        sliced: dict[str, Any] = {}
        for key, value in record.items():
            if key == "stage":
                sliced[key] = stage
                continue
            array = np.asarray(value) if isinstance(value, (list, np.ndarray)) else None
            if array is not None and array.ndim >= 1 and len(array) == total:
                sliced[key] = array[start:end].copy()
            else:
                sliced[key] = value
        return sliced

    def _select_candidates_v2(
        self,
        pools: tuple[FloatArray, FloatArray],
        batch_size: int,
        archive_x: FloatArray,
        objective: FloatArray,
        constraints: FloatArray,
        violations: FloatArray,
        constraint_scales: FloatArray,
        evaluation_generation: int,
    ) -> tuple[
        FloatArray,
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
        str,
        dict[str, Any],
    ]:
        """Attach scale labels while retaining V2.9's selected points."""

        result = super()._select_candidates_v2(
            pools,
            batch_size,
            archive_x,
            objective,
            constraints,
            violations,
            constraint_scales,
            evaluation_generation,
        )
        selected, method, metadata, pools_trace, phase, diagnostics = result
        coarse_radius = self._local_radius(evaluation_generation)
        fine_radius = self._fine_local_radius(evaluation_generation)

        for point, row in zip(selected, metadata, strict=True):
            if row.get("source") != "local_search":
                continue
            if self._point_in_pool(point, self._last_fine_local_pool):
                row.update(
                    local_search_scale="fine",
                    local_search_radius=fine_radius,
                    local_search_changed_dimensions=1,
                )
            elif self._point_in_pool(point, self._last_coarse_local_pool):
                row.update(
                    local_search_scale="coarse",
                    local_search_radius=coarse_radius,
                    local_search_changed_dimensions=max(
                        1, int(coarse_radius * self.dimension)
                    ),
                )

        split_trace: list[dict[str, Any]] = []
        coarse_count = len(self._last_coarse_local_pool)
        for record in pools_trace:
            if record.get("stage") != "local_search":
                split_trace.append(record)
                continue
            split_trace.append(
                self._slice_pool_record(
                    record, 0, coarse_count, "local_search_coarse"
                )
            )
            split_trace.append(
                self._slice_pool_record(
                    record,
                    coarse_count,
                    len(record["x"]),
                    "local_search_fine",
                )
            )

        diagnostics.update(
            local_search_design="half_coarse_half_fine",
            local_coarse_candidates=len(self._last_coarse_local_pool),
            local_fine_candidates=len(self._last_fine_local_pool),
            local_coarse_radius=coarse_radius,
            local_fine_radius=fine_radius,
        )
        return selected, method, metadata, split_trace, phase, diagnostics

    @staticmethod
    def _v2_history_entry(
        generation: int,
        objective: FloatArray,
        violations: FloatArray,
        weights: FloatArray,
        evaluations: int,
        sampling_method: str | None,
        population: np.ndarray,
        local_radius: float | None,
        constraint_scales: FloatArray,
        rbf_active: bool,
        acquisition_phase: str,
        surrogate_diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        entry = SADEV2_9._v2_history_entry(
            generation,
            objective,
            violations,
            weights,
            evaluations,
            sampling_method,
            population,
            local_radius,
            constraint_scales,
            rbf_active,
            acquisition_phase,
            surrogate_diagnostics,
        )
        entry["local_search_design"] = "half_coarse_half_fine"
        entry["local_fine_radius"] = (
            None
            if local_radius is None
            else max(
                SADEV2_10._FINE_RADIUS_FLOOR,
                SADEV2_10._FINE_RADIUS_RATIO * local_radius,
            )
        )
        return entry
