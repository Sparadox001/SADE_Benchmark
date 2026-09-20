"""True-evaluation oracle diagnostics for saved SADE candidate pools."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ..algorithms.sade.sampling import calculate_distance
from ..benchmarks import make_problem
from ..core.problem import FloatArray


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_evaluations(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                **row,
                "evaluation": int(row["evaluation"]),
                "generation": int(row["generation"]),
                "objective_value": float(row["objective"]),
                "x_value": np.asarray(json.loads(row["x"]), dtype=float),
                "constraints_value": np.asarray(
                    json.loads(row["constraints"]), dtype=float
                ),
            }
        )
    return parsed


def _eligible_pool(
    raw_x: FloatArray,
    reference: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    threshold: float,
) -> tuple[FloatArray, np.ndarray]:
    """Return eligible points and their indices in the saved raw pool."""

    _, first_indices = np.unique(raw_x, axis=0, return_index=True)
    raw_unique_indices = np.sort(first_indices)
    unique_x = raw_x[raw_unique_indices]
    valid_indices, eligible_x = calculate_distance(
        reference, unique_x, lower, upper, threshold
    )
    return eligible_x, raw_unique_indices[valid_indices]


def _true_values(
    problem: Any,
    x: FloatArray,
    scales: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    objective, constraints = problem.evaluate(x)
    violation = np.maximum(constraints - tolerance, 0.0)
    normalized_cv = np.sum(violation / scales, axis=1)
    feasible = np.all(violation <= 0.0, axis=1)
    return objective, normalized_cv, feasible


def _selected_rank_percentile(
    pool_objective: FloatArray,
    pool_cv: FloatArray,
    pool_feasible: np.ndarray,
    selected_objective: float,
    selected_cv: float,
    selected_feasible: bool,
) -> float:
    if selected_feasible:
        better = pool_feasible & (pool_objective < selected_objective)
    else:
        better = pool_feasible | (
            (~pool_feasible) & (pool_cv < selected_cv)
        )
    rank = 1 + int(np.count_nonzero(better))
    return min(1.0, rank / max(1, len(pool_objective)))


def _finite_spearman(predicted: FloatArray, actual: FloatArray) -> float:
    valid = np.isfinite(predicted) & np.isfinite(actual)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    if np.ptp(predicted[valid]) == 0.0 or np.ptp(actual[valid]) == 0.0:
        return float("nan")
    return float(spearmanr(predicted[valid], actual[valid]).statistic)


def _opportunity_metrics(
    phase: str,
    incumbent_objective: float | None,
    incumbent_cv: float,
    pool_objective: FloatArray,
    pool_cv: FloatArray,
    pool_feasible: np.ndarray,
    selected_objective: FloatArray,
    selected_cv: FloatArray,
    selected_feasible: np.ndarray,
) -> dict[str, Any]:
    if phase == "objective" and incumbent_objective is not None:
        pool_values = pool_objective[pool_feasible]
        selected_values = selected_objective[selected_feasible]
        oracle = float(np.min(pool_values)) if len(pool_values) else None
        selected_best = (
            float(np.min(selected_values)) if len(selected_values) else None
        )
        pool_improves = bool(
            oracle is not None and oracle < incumbent_objective
        )
        selected_improves = bool(
            selected_best is not None and selected_best < incumbent_objective
        )
        regret = (
            None
            if oracle is None or selected_best is None
            else selected_best - oracle
        )
    else:
        oracle = float(np.min(pool_cv))
        selected_best = float(np.min(selected_cv))
        pool_improves = bool(oracle < incumbent_cv)
        selected_improves = bool(selected_best < incumbent_cv)
        regret = selected_best - oracle
    return {
        "oracle_value": oracle,
        "selected_best_value": selected_best,
        "oracle_regret": regret,
        "pool_improves_incumbent": pool_improves,
        "selected_improves_incumbent": selected_improves,
        "missed_improvement": pool_improves and not selected_improves,
        "pool_has_feasible": bool(np.any(pool_feasible)),
        "selected_has_feasible": bool(np.any(selected_feasible)),
        "pool_feasible_count": int(np.count_nonzero(pool_feasible)),
        "pool_feasible_fraction": float(np.mean(pool_feasible)),
        "selected_feasible_count": int(np.count_nonzero(selected_feasible)),
    }


def _pool_records(pool_path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    with np.load(pool_path) as saved:
        arrays = {name: saved[name] for name in saved.files}
    records: dict[tuple[int, str], dict[str, Any]] = {}
    fields = set(arrays) - {
        "offsets",
        "generation",
        "inner_generation",
        "stage",
        "acquisition_phase",
        "x",
    }
    offsets = arrays["offsets"]
    for index, (generation, stage) in enumerate(
        zip(arrays["generation"], arrays["stage"], strict=True)
    ):
        start, end = offsets[index : index + 2]
        record = {"x": arrays["x"][start:end]}
        for field in fields:
            record[field] = arrays[field][start:end]
        records[(int(generation), str(stage))] = record
    return records


def analyze_candidate_pool_run(
    run_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze one saved run without changing its algorithmic trace."""

    config = json.loads((run_directory / "config.json").read_text("utf-8"))
    problem = make_problem(
        config["suite"], config["problem"], int(config["dimension"])
    )
    algorithm_config = config["algorithm_config"]
    tolerance = float(algorithm_config["constraint_tolerance"])
    threshold = float(algorithm_config["distance_threshold"])
    evaluations = _read_evaluations(run_directory / "evaluations.csv.gz")
    history = _read_csv(run_directory / "history.csv")
    scales = np.asarray(json.loads(history[0]["constraint_scales"]), dtype=float)
    pools = _pool_records(run_directory / "candidate_pools.npz")
    seed = int(config["seed"])

    slot_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    sampled_generations = sorted(
        {row["generation"] for row in evaluations if row["generation"] > 0}
    )
    for generation in sampled_generations:
        selected_rows = [
            row for row in evaluations if row["generation"] == generation
        ]
        prior_rows = [
            row for row in evaluations if row["generation"] < generation
        ]
        prior_x = np.vstack([row["x_value"] for row in prior_rows])
        prior_objective = np.asarray(
            [row["objective_value"] for row in prior_rows]
        )
        prior_constraints = np.vstack(
            [row["constraints_value"] for row in prior_rows]
        )
        prior_violation = np.maximum(prior_constraints - tolerance, 0.0)
        prior_cv = np.sum(prior_violation / scales, axis=1)
        prior_feasible = np.all(prior_violation <= 0.0, axis=1)
        incumbent_cv = float(np.min(prior_cv))
        incumbent_objective = (
            float(np.min(prior_objective[prior_feasible]))
            if np.any(prior_feasible)
            else None
        )
        phase = selected_rows[0]["acquisition_phase"]
        previous_selected: list[FloatArray] = []
        batch_pool_x: list[FloatArray] = []
        batch_pool_objective: list[FloatArray] = []
        batch_pool_cv: list[FloatArray] = []
        batch_pool_feasible: list[np.ndarray] = []
        selected_objective_values: list[float] = []
        selected_cv_values: list[float] = []
        selected_feasible_values: list[bool] = []

        for slot, selected in enumerate(selected_rows, start=1):
            source = selected["source"]
            record = pools[(generation, source)]
            reference = (
                prior_x
                if not previous_selected
                else np.vstack((prior_x, previous_selected))
            )
            eligible_x, raw_indices = _eligible_pool(
                record["x"],
                reference,
                problem.lower_bounds,
                problem.upper_bounds,
                threshold,
            )
            pool_objective, pool_cv, pool_feasible = _true_values(
                problem, eligible_x, scales, tolerance
            )
            selected_x = selected["x_value"]
            selected_f, selected_cv, selected_feasible = _true_values(
                problem, selected_x[None, :], scales, tolerance
            )
            selected_percentile = _selected_rank_percentile(
                pool_objective,
                pool_cv,
                pool_feasible,
                float(selected_f[0]),
                float(selected_cv[0]),
                bool(selected_feasible[0]),
            )
            if phase == "feasibility":
                predicted = record.get(
                    "robust_predicted_total_violation",
                    record.get("predicted_total_violation"),
                )
                rank_correlation = _finite_spearman(
                    np.asarray(predicted)[raw_indices], pool_cv
                )
            else:
                true_feasible = pool_feasible
                predicted_objective = np.asarray(
                    record["predicted_objective"]
                )[raw_indices]
                rank_correlation = _finite_spearman(
                    predicted_objective[true_feasible],
                    pool_objective[true_feasible],
                )
            robust_feasible = np.asarray(
                record.get(
                    "robust_predicted_feasible",
                    record.get("predicted_feasible"),
                )
            )[raw_indices].astype(bool)
            robust_recall = (
                float(np.mean(robust_feasible[pool_feasible]))
                if np.any(pool_feasible)
                else float("nan")
            )
            robust_precision = (
                float(np.mean(pool_feasible[robust_feasible]))
                if np.any(robust_feasible)
                else float("nan")
            )
            opportunity = _opportunity_metrics(
                phase,
                incumbent_objective,
                incumbent_cv,
                pool_objective,
                pool_cv,
                pool_feasible,
                selected_f,
                selected_cv,
                selected_feasible,
            )
            slot_rows.append(
                {
                    "suite": config["suite"],
                    "problem": config["problem"],
                    "dimension": int(config["dimension"]),
                    "seed": seed,
                    "generation": generation,
                    "evaluations_before_batch": len(prior_rows),
                    "slot": slot,
                    "phase": phase,
                    "source": source,
                    "acquisition_role": selected["acquisition_role"],
                    "raw_pool_size": len(record["x"]),
                    "eligible_pool_size": len(eligible_x),
                    "selected_objective": float(selected_f[0]),
                    "selected_normalized_cv": float(selected_cv[0]),
                    "selected_feasible": bool(selected_feasible[0]),
                    "selected_true_rank_percentile": selected_percentile,
                    "selected_in_true_top_10pct": selected_percentile <= 0.10,
                    "prediction_true_spearman": rank_correlation,
                    "robust_feasible_recall": robust_recall,
                    "robust_feasible_precision": robust_precision,
                    **opportunity,
                }
            )
            previous_selected.append(selected_x)
            batch_pool_x.append(eligible_x)
            batch_pool_objective.append(pool_objective)
            batch_pool_cv.append(pool_cv)
            batch_pool_feasible.append(pool_feasible)
            selected_objective_values.append(float(selected_f[0]))
            selected_cv_values.append(float(selected_cv[0]))
            selected_feasible_values.append(bool(selected_feasible[0]))

        union_x = np.vstack(batch_pool_x)
        _, unique_indices = np.unique(union_x, axis=0, return_index=True)
        keep = np.sort(unique_indices)
        union_objective = np.concatenate(batch_pool_objective)[keep]
        union_cv = np.concatenate(batch_pool_cv)[keep]
        union_feasible = np.concatenate(batch_pool_feasible)[keep]
        selected_objective_array = np.asarray(selected_objective_values)
        selected_cv_array = np.asarray(selected_cv_values)
        selected_feasible_array = np.asarray(selected_feasible_values)
        opportunity = _opportunity_metrics(
            phase,
            incumbent_objective,
            incumbent_cv,
            union_objective,
            union_cv,
            union_feasible,
            selected_objective_array,
            selected_cv_array,
            selected_feasible_array,
        )
        if phase == "objective" and np.any(selected_feasible_array):
            feasible_positions = np.flatnonzero(selected_feasible_array)
            best_selected_position = int(
                feasible_positions[
                    np.argmin(selected_objective_array[feasible_positions])
                ]
            )
        else:
            best_selected_position = int(np.argmin(selected_cv_array))
        selected_percentile = _selected_rank_percentile(
            union_objective,
            union_cv,
            union_feasible,
            float(selected_objective_array[best_selected_position]),
            float(selected_cv_array[best_selected_position]),
            bool(selected_feasible_array[best_selected_position]),
        )
        batch_rows.append(
            {
                "suite": config["suite"],
                "problem": config["problem"],
                "dimension": int(config["dimension"]),
                "seed": seed,
                "generation": generation,
                "evaluations_before_batch": len(prior_rows),
                "phase": phase,
                "eligible_union_size": len(union_objective),
                "incumbent_objective": incumbent_objective,
                "incumbent_normalized_cv": incumbent_cv,
                "selected_true_rank_percentile": selected_percentile,
                "selected_in_true_top_10pct": selected_percentile <= 0.10,
                **opportunity,
            }
        )
    return slot_rows, batch_rows


def _mean_boolean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows]))


def _summaries(
    slot_rows: list[dict[str, Any]], batch_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    slot_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in slot_rows:
        key = (
            row["suite"], row["problem"], row["dimension"],
            row["phase"], row["source"],
        )
        slot_groups[key].append(row)
    for key, rows in slot_groups.items():
        correlations = np.asarray(
            [row["prediction_true_spearman"] for row in rows], dtype=float
        )
        recalls = np.asarray(
            [row["robust_feasible_recall"] for row in rows], dtype=float
        )
        precisions = np.asarray(
            [row["robust_feasible_precision"] for row in rows], dtype=float
        )
        summaries.append(
            {
                "level": "slot",
                "suite": key[0],
                "problem": key[1],
                "dimension": key[2],
                "phase": key[3],
                "source": key[4],
                "count": len(rows),
                "opportunity_rate": _mean_boolean(
                    rows, "pool_improves_incumbent"
                ),
                "capture_rate_given_opportunity": (
                    np.mean(
                        [
                            row["selected_improves_incumbent"]
                            for row in rows
                            if row["pool_improves_incumbent"]
                        ]
                    )
                    if any(row["pool_improves_incumbent"] for row in rows)
                    else None
                ),
                "missed_opportunities": sum(
                    bool(row["missed_improvement"]) for row in rows
                ),
                "top_10pct_selection_rate": _mean_boolean(
                    rows, "selected_in_true_top_10pct"
                ),
                "median_true_rank_percentile": float(
                    np.median(
                        [row["selected_true_rank_percentile"] for row in rows]
                    )
                ),
                "median_prediction_true_spearman": (
                    float(np.nanmedian(correlations))
                    if np.any(np.isfinite(correlations))
                    else None
                ),
                "mean_robust_feasible_recall": (
                    float(np.nanmean(recalls))
                    if np.any(np.isfinite(recalls))
                    else None
                ),
                "mean_robust_feasible_precision": (
                    float(np.nanmean(precisions))
                    if np.any(np.isfinite(precisions))
                    else None
                ),
            }
        )
    batch_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in batch_rows:
        key = (
            row["suite"], row["problem"], row["dimension"], row["phase"]
        )
        batch_groups[key].append(row)
    for key, rows in batch_groups.items():
        opportunities = [row for row in rows if row["pool_improves_incumbent"]]
        feasible_opportunities = [row for row in rows if row["pool_has_feasible"]]
        summaries.append(
            {
                "level": "batch",
                "suite": key[0],
                "problem": key[1],
                "dimension": key[2],
                "phase": key[3],
                "source": "combined",
                "count": len(rows),
                "opportunity_rate": len(opportunities) / len(rows),
                "capture_rate_given_opportunity": (
                    np.mean(
                        [row["selected_improves_incumbent"] for row in opportunities]
                    )
                    if opportunities
                    else None
                ),
                "missed_opportunities": sum(
                    bool(row["missed_improvement"]) for row in rows
                ),
                "feasible_opportunity_rate": (
                    len(feasible_opportunities) / len(rows)
                ),
                "feasible_capture_rate_given_opportunity": (
                    np.mean(
                        [row["selected_has_feasible"] for row in feasible_opportunities]
                    )
                    if feasible_opportunities
                    else None
                ),
                "top_10pct_selection_rate": _mean_boolean(
                    rows, "selected_in_true_top_10pct"
                ),
                "median_true_rank_percentile": float(
                    np.median(
                        [row["selected_true_rank_percentile"] for row in rows]
                    )
                ),
                "median_prediction_true_spearman": None,
                "mean_robust_feasible_recall": None,
                "mean_robust_feasible_precision": None,
            }
        )
    return summaries


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_candidate_pool_results(
    result_root: Path, output_directory: Path | None = None
) -> tuple[Path, Path, Path]:
    """Analyze every run below a result root and persist compact CSVs."""

    pool_paths = sorted(result_root.rglob("candidate_pools.npz"))
    if not pool_paths:
        raise ValueError(f"No candidate_pools.npz files found below {result_root}.")
    all_slots: list[dict[str, Any]] = []
    all_batches: list[dict[str, Any]] = []
    for pool_path in pool_paths:
        slots, batches = analyze_candidate_pool_run(pool_path.parent)
        all_slots.extend(slots)
        all_batches.extend(batches)
    summaries = _summaries(all_slots, all_batches)
    destination = output_directory or result_root / "candidate_oracle"
    destination.mkdir(parents=True, exist_ok=True)
    slot_path = destination / "oracle_slots.csv"
    batch_path = destination / "oracle_batches.csv"
    summary_path = destination / "oracle_summary.csv"
    _write_rows(slot_path, all_slots)
    _write_rows(batch_path, all_batches)
    _write_rows(summary_path, summaries)
    return slot_path, batch_path, summary_path
