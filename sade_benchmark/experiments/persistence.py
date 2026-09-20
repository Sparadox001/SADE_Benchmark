"""Per-run persistence for reproducible benchmark experiments."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.result import OptimizationResult


def scientific(value: float | None) -> str | None:
    """Format finite scalar values in paper-friendly scientific notation."""

    if value is None or not np.isfinite(float(value)):
        return None
    return f"{float(value):.16e}"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_value(key: str, value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(_jsonable(value), separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, (float, np.floating)):
        if "objective" in key or "violation" in key or "variance" in key:
            return scientific(float(value))
        return None if not np.isfinite(value) else float(value)
    return value


def _history_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: _csv_value(key, value) for key, value in entry.items()}
        for entry in history
    ]


def _evaluation_rows(result: OptimizationResult) -> list[dict[str, Any]]:
    if result.archive_x is None or result.archive_objective is None:
        return []
    constraints = result.archive_constraints
    violations = result.archive_violation
    if constraints is None:
        constraints = np.empty((len(result.archive_x), 0))
    if violations is None:
        violations = np.maximum(constraints, 0.0)

    rows: list[dict[str, Any]] = []
    for index in range(len(result.archive_x)):
        metadata = (
            result.evaluation_metadata[index]
            if index < len(result.evaluation_metadata)
            else {}
        )
        row: dict[str, Any] = {
            "evaluation": index + 1,
            "generation": metadata.get("generation"),
            "source": metadata.get("source"),
            "sampling_method": metadata.get("sampling_method"),
            "objective": scientific(float(result.archive_objective[index])),
            "total_violation": scientific(float(np.sum(violations[index]))),
            "feasible": bool(np.all(violations[index] <= 0.0)),
            "x": json.dumps(result.archive_x[index].tolist(), separators=(",", ":")),
            "constraints": json.dumps(constraints[index].tolist(), separators=(",", ":")),
            "violations": json.dumps(violations[index].tolist(), separators=(",", ":")),
        }
        for key, value in metadata.items():
            if key not in row:
                row[key] = _csv_value(key, value)
        rows.append(row)
    return rows


def _write_evaluations(path: Path, result: OptimizationResult) -> None:
    rows = _evaluation_rows(result)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with gzip.open(path, "wt", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_populations(path: Path, snapshots: list[dict[str, Any]]) -> None:
    if not snapshots:
        return
    payload = {
        "generation": np.asarray(
            [item["generation"] for item in snapshots], dtype=int
        ),
        "evaluations": np.asarray(
            [item["evaluations"] for item in snapshots], dtype=int
        ),
        "archive_indices": np.stack(
            [item["archive_indices"] for item in snapshots]
        ),
        "x": np.stack([item["x"] for item in snapshots]),
        "objective": np.stack([item["objective"] for item in snapshots]),
        "total_violation": np.stack(
            [item["total_violation"] for item in snapshots]
        ),
    }
    if all("search_total_violation" in item for item in snapshots):
        payload["search_total_violation"] = np.stack(
            [item["search_total_violation"] for item in snapshots]
        )
    np.savez_compressed(path, **payload)


def _write_candidate_pools(path: Path, pools: list[dict[str, Any]]) -> None:
    if not pools:
        return
    lengths = np.asarray([len(item["x"]) for item in pools], dtype=int)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    total = int(offsets[-1])
    payload: dict[str, Any] = {
        "offsets": offsets,
        "generation": np.asarray([item["generation"] for item in pools], dtype=int),
        "inner_generation": np.asarray(
            [item.get("inner_generation", -1) for item in pools], dtype=int
        ),
        "stage": np.asarray([item["stage"] for item in pools]),
        "acquisition_phase": np.asarray(
            [item.get("acquisition_phase", "") for item in pools]
        ),
        "x": np.vstack([item["x"] for item in pools]),
    }
    for field in (
        "predicted_objective",
        "predicted_fitness",
        "predicted_total_violation",
        "predicted_search_total_violation",
        "rbf_variance",
        "expected_improvement",
        "objective_ei",
        "surrogate_uncertainty",
        "predicted_feasible",
        "robust_predicted_total_violation",
        "robust_predicted_feasible",
        "constraint_calibration_active",
        "empirical_joint_feasibility_probability",
        "expected_constraint_violation",
        "continuous_feasibility_weight",
        "constrained_expected_improvement",
        "constraint_residual_samples",
        "continuous_feasibility_active",
    ):
        values = np.full(total, np.nan)
        found = False
        for pool_index, item in enumerate(pools):
            if field in item:
                start, end = offsets[pool_index : pool_index + 2]
                values[start:end] = np.asarray(item[field], dtype=float).reshape(-1)
                found = True
        if found:
            payload[field] = values
    constraint_width = next(
        (
            np.asarray(item["predicted_constraints"]).shape[1]
            for item in pools
            if "predicted_constraints" in item
        ),
        None,
    )
    if constraint_width is not None:
        predicted_constraints = np.full((total, constraint_width), np.nan)
        for pool_index, item in enumerate(pools):
            if "predicted_constraints" in item:
                start, end = offsets[pool_index : pool_index + 2]
                predicted_constraints[start:end] = np.asarray(
                    item["predicted_constraints"], dtype=float
                )
        payload["predicted_constraints"] = predicted_constraints
    error_width = next(
        (
            len(np.asarray(item["constraint_error_quantiles"]).reshape(-1))
            for item in pools
            if "constraint_error_quantiles" in item
        ),
        None,
    )
    if error_width is not None:
        error_quantiles = np.full((total, error_width), np.nan)
        for pool_index, item in enumerate(pools):
            if "constraint_error_quantiles" in item:
                start, end = offsets[pool_index : pool_index + 2]
                values = np.asarray(
                    item["constraint_error_quantiles"], dtype=float
                )
                if values.ndim == 1:
                    error_quantiles[start:end] = values
                else:
                    error_quantiles[start:end] = values
        payload["constraint_error_quantiles"] = error_quantiles
    np.savez_compressed(path, **payload)


def save_run(
    run_directory: Path,
    *,
    algorithm: str,
    problem: Any,
    run: int,
    seed: int,
    config: Any,
    result: OptimizationResult,
    save_evaluations: bool = True,
    save_populations: bool = True,
) -> dict[str, Any]:
    """Persist one completed run and return its root-summary row."""

    run_directory.mkdir(parents=True, exist_ok=True)
    known_optimum = getattr(problem, "known_optimum", None)
    objective_error = (
        float(result.objective - known_optimum) if known_optimum is not None else None
    )
    config_data = {
        "algorithm": algorithm,
        "suite": problem.suite,
        "problem": problem.name,
        "dimension": problem.dimension,
        "run": run,
        "seed": seed,
        "algorithm_config": config,
    }
    objective_cap = getattr(problem, "objective_cap", None)
    if objective_cap is not None:
        config_data["objective_cap"] = scientific(objective_cap)
    write_json(run_directory / "config.json", config_data)
    final_data = {
        "objective": scientific(result.objective),
        "objective_error": scientific(objective_error),
        "known_optimum": scientific(known_optimum),
        "total_violation": scientific(float(np.sum(result.violation))),
        "feasible": result.feasible,
        "evaluations": result.evaluations,
        "generations": result.generations,
        "x": result.x,
        "constraints": result.constraints,
        "violations": result.violation,
        "dynamic_constraint": result.dynamic_constraint_state,
    }
    if objective_cap is not None:
        final_data["objective_cap"] = scientific(objective_cap)
    write_json(run_directory / "final_result.json", final_data)
    if result.dynamic_constraint_state is not None:
        write_json(
            run_directory / "dynamic_constraint_history.json",
            result.dynamic_constraint_state,
        )
    write_rows(run_directory / "history.csv", _history_rows(result.history))
    if save_evaluations:
        _write_evaluations(run_directory / "evaluations.csv.gz", result)
    if save_populations:
        _write_populations(run_directory / "populations.npz", result.population_history)
    if result.candidate_pools is not None:
        _write_candidate_pools(run_directory / "candidate_pools.npz", result.candidate_pools)

    row = {
        "status": "ok",
        "algorithm": algorithm,
        "suite": problem.suite,
        "problem": problem.name,
        "dimension": problem.dimension,
        "run": run,
        "seed": seed,
        "objective": scientific(result.objective),
        "objective_numeric": float(result.objective),
        "objective_error": scientific(objective_error),
        "objective_error_numeric": objective_error,
        "total_violation": scientific(float(np.sum(result.violation))),
        "feasible": result.feasible,
        "evaluations": result.evaluations,
        "generations": result.generations,
        "run_directory": str(run_directory),
    }
    if objective_cap is not None:
        row["objective_cap"] = scientific(objective_cap)
    return row
