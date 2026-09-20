"""Counterfactual replay of simple selectors on saved V2.7 pools."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmarks import make_problem
from ..core.problem import FloatArray
from .candidate_oracle import (
    _eligible_pool,
    _opportunity_metrics,
    _pool_records,
    _read_csv,
    _read_evaluations,
    _selected_rank_percentile,
    _true_values,
)


SELECTORS = (
    "current_p90",
    "raw_sum_cv",
    "raw_max_cv",
    "feasibility_probability",
    "p90_then_raw",
    "p90_then_probability",
)


def _argmin_with_uncertainty(
    values: FloatArray, uncertainty: FloatArray
) -> int:
    minimum = float(np.nanmin(values))
    tied = np.flatnonzero(
        np.isclose(values, minimum, rtol=1e-12, atol=1e-15)
    )
    return int(tied[np.nanargmax(uncertainty[tied])])


def _boundary_index(
    robust_cv: FloatArray,
    uncertainty: FloatArray,
    fraction: float,
) -> int:
    count = max(1, int(np.ceil(fraction * len(robust_cv))))
    order = np.argsort(robust_cv, kind="stable")
    cutoff = float(robust_cv[order[count - 1]])
    tolerance = max(1e-15, 1e-12 * abs(cutoff))
    shortlist = np.flatnonzero(robust_cv <= cutoff + tolerance)
    return int(shortlist[np.nanargmax(uncertainty[shortlist])])


def _raw_constraint_scores(
    predicted_constraints: FloatArray,
    scales: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    normalized = (predicted_constraints - tolerance) / scales
    positive = np.maximum(normalized, 0.0)
    return (
        np.sum(positive, axis=1),
        np.max(positive, axis=1),
        np.all(normalized <= 0.0, axis=1),
    )


def _raw_objective_index(
    predicted_objective: FloatArray,
    raw_feasible: np.ndarray,
    fallback_score: FloatArray,
    uncertainty: FloatArray,
) -> int:
    feasible = np.flatnonzero(raw_feasible)
    if len(feasible):
        order = np.lexsort(
            (uncertainty[feasible], predicted_objective[feasible])
        )
        return int(feasible[order[0]])
    return _argmin_with_uncertainty(fallback_score, uncertainty)


def _selector_index(
    selector: str,
    phase: str,
    role: str,
    scores: dict[str, FloatArray],
    scales: FloatArray,
    tolerance: float,
    hard_feasible_min_fraction: float,
    near_feasible_fraction: float,
) -> int:
    predicted_objective = scores["predicted_objective"]
    predicted_constraints = scores["predicted_constraints"]
    uncertainty = scores["surrogate_uncertainty"]
    raw_sum, raw_max, raw_feasible = _raw_constraint_scores(
        predicted_constraints, scales, tolerance
    )
    robust_cv = scores["robust_predicted_total_violation"]
    robust_feasible = scores["robust_predicted_feasible"].astype(bool)

    if role == "boundary_exploration":
        return _boundary_index(
            robust_cv, uncertainty, near_feasible_fraction
        )

    if selector in {
        "current_p90", "p90_then_raw", "p90_then_probability"
    }:
        if phase == "feasibility":
            return _argmin_with_uncertainty(robust_cv, uncertainty)
        feasible = np.flatnonzero(robust_feasible)
        required = max(
            1,
            int(np.ceil(hard_feasible_min_fraction * len(robust_feasible))),
        )
        if len(feasible) >= required:
            order = np.lexsort(
                (uncertainty[feasible], predicted_objective[feasible])
            )
            return int(feasible[order[0]])
        if selector == "p90_then_raw":
            return _raw_objective_index(
                predicted_objective,
                raw_feasible,
                raw_sum,
                uncertainty,
            )
        if selector == "p90_then_probability":
            probability = scores[
                "empirical_joint_feasibility_probability"
            ]
            if np.any(np.isfinite(probability)):
                best_probability = float(np.nanmax(probability))
                shortlist = np.flatnonzero(
                    np.isclose(
                        probability,
                        best_probability,
                        rtol=1e-12,
                        atol=1e-15,
                    )
                )
                order = np.lexsort(
                    (
                        uncertainty[shortlist],
                        predicted_objective[shortlist],
                    )
                )
                return int(shortlist[order[0]])
            return _raw_objective_index(
                predicted_objective,
                raw_feasible,
                raw_sum,
                uncertainty,
            )
        constrained_ei = np.nan_to_num(
            scores["constrained_expected_improvement"], nan=-np.inf
        )
        weight = np.nan_to_num(
            scores["continuous_feasibility_weight"], nan=-np.inf
        )
        safe_uncertainty = np.nan_to_num(uncertainty, nan=-np.inf)
        return int(
            np.lexsort(
                (-safe_uncertainty, -weight, -constrained_ei)
            )[0]
        )

    if selector == "raw_sum_cv":
        if phase == "feasibility":
            return _argmin_with_uncertainty(raw_sum, uncertainty)
        return _raw_objective_index(
            predicted_objective,
            raw_feasible,
            raw_sum,
            uncertainty,
        )

    if selector == "raw_max_cv":
        if phase == "feasibility":
            order = np.lexsort((-uncertainty, raw_sum, raw_max))
            return int(order[0])
        return _raw_objective_index(
            predicted_objective,
            raw_feasible,
            raw_max,
            uncertainty,
        )

    probability = scores["empirical_joint_feasibility_probability"]
    active = np.isfinite(probability)
    if not np.any(active):
        if phase == "objective":
            return _raw_objective_index(
                predicted_objective,
                raw_feasible,
                raw_sum,
                uncertainty,
            )
        return _argmin_with_uncertainty(raw_sum, uncertainty)
    best_probability = float(np.nanmax(probability))
    shortlist = np.flatnonzero(
        np.isclose(probability, best_probability, rtol=1e-12, atol=1e-15)
    )
    if phase == "objective":
        order = np.lexsort(
            (uncertainty[shortlist], predicted_objective[shortlist])
        )
    else:
        expected_cv = scores["expected_constraint_violation"]
        order = np.lexsort(
            (-uncertainty[shortlist], expected_cv[shortlist])
        )
    return int(shortlist[order[0]])


def _record_scores(
    record: dict[str, Any], raw_indices: np.ndarray
) -> dict[str, FloatArray]:
    names = (
        "predicted_objective",
        "predicted_constraints",
        "surrogate_uncertainty",
        "robust_predicted_total_violation",
        "robust_predicted_feasible",
        "empirical_joint_feasibility_probability",
        "expected_constraint_violation",
        "continuous_feasibility_weight",
        "constrained_expected_improvement",
    )
    return {
        name: np.asarray(record[name])[raw_indices]
        for name in names
    }


def replay_run(run_directory: Path) -> list[dict[str, Any]]:
    config = json.loads((run_directory / "config.json").read_text("utf-8"))
    problem = make_problem(
        config["suite"], config["problem"], int(config["dimension"])
    )
    algorithm_config = config["algorithm_config"]
    tolerance = float(algorithm_config["constraint_tolerance"])
    threshold = float(algorithm_config["distance_threshold"])
    hard_fraction = float(algorithm_config["hard_feasible_min_fraction"])
    near_fraction = float(algorithm_config["near_feasible_fraction"])
    evaluations = _read_evaluations(run_directory / "evaluations.csv.gz")
    history = _read_csv(run_directory / "history.csv")
    scales = np.asarray(json.loads(history[0]["constraint_scales"]), dtype=float)
    pools = _pool_records(run_directory / "candidate_pools.npz")
    output: list[dict[str, Any]] = []

    generations = sorted(
        {row["generation"] for row in evaluations if row["generation"] > 0}
    )
    for generation in generations:
        actual_selected = [
            row for row in evaluations if row["generation"] == generation
        ]
        prior = [row for row in evaluations if row["generation"] < generation]
        prior_x = np.vstack([row["x_value"] for row in prior])
        prior_objective = np.asarray([row["objective_value"] for row in prior])
        prior_constraints = np.vstack(
            [row["constraints_value"] for row in prior]
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
        phase = actual_selected[0]["acquisition_phase"]

        for selector in SELECTORS:
            chosen_x: list[FloatArray] = []
            chosen_objective: list[float] = []
            chosen_cv: list[float] = []
            chosen_feasible: list[bool] = []
            union_x: list[FloatArray] = []
            union_objective: list[FloatArray] = []
            union_cv: list[FloatArray] = []
            union_feasible: list[np.ndarray] = []

            for actual in actual_selected:
                source = actual["source"]
                role = actual["acquisition_role"]
                record = pools[(generation, source)]
                reference = (
                    prior_x
                    if not chosen_x
                    else np.vstack((prior_x, chosen_x))
                )
                eligible_x, raw_indices = _eligible_pool(
                    record["x"],
                    reference,
                    problem.lower_bounds,
                    problem.upper_bounds,
                    threshold,
                )
                pool_f, pool_cv, pool_feasible = _true_values(
                    problem, eligible_x, scales, tolerance
                )
                use_actual = (
                    actual["sampling_method"] != "two_stage_rbf"
                    or role == "candidate_shortage_fallback"
                )
                if use_actual:
                    selected_x = actual["x_value"]
                else:
                    scores = _record_scores(record, raw_indices)
                    index = _selector_index(
                        selector,
                        phase,
                        role,
                        scores,
                        scales,
                        tolerance,
                        hard_fraction,
                        near_fraction,
                    )
                    selected_x = eligible_x[index]
                selected_f, selected_cv, selected_feasible = _true_values(
                    problem, selected_x[None, :], scales, tolerance
                )
                chosen_x.append(selected_x)
                chosen_objective.append(float(selected_f[0]))
                chosen_cv.append(float(selected_cv[0]))
                chosen_feasible.append(bool(selected_feasible[0]))
                union_x.append(eligible_x)
                union_objective.append(pool_f)
                union_cv.append(pool_cv)
                union_feasible.append(pool_feasible)

            combined_x = np.vstack(union_x)
            _, first = np.unique(combined_x, axis=0, return_index=True)
            keep = np.sort(first)
            combined_f = np.concatenate(union_objective)[keep]
            combined_cv = np.concatenate(union_cv)[keep]
            combined_feasible = np.concatenate(union_feasible)[keep]
            selected_f = np.asarray(chosen_objective)
            selected_cv = np.asarray(chosen_cv)
            selected_feasible = np.asarray(chosen_feasible)
            opportunity = _opportunity_metrics(
                phase,
                incumbent_objective,
                incumbent_cv,
                combined_f,
                combined_cv,
                combined_feasible,
                selected_f,
                selected_cv,
                selected_feasible,
            )
            if phase == "objective" and np.any(selected_feasible):
                positions = np.flatnonzero(selected_feasible)
                best = int(positions[np.argmin(selected_f[positions])])
            else:
                best = int(np.argmin(selected_cv))
            percentile = _selected_rank_percentile(
                combined_f,
                combined_cv,
                combined_feasible,
                float(selected_f[best]),
                float(selected_cv[best]),
                bool(selected_feasible[best]),
            )
            output.append(
                {
                    "selector": selector,
                    "suite": config["suite"],
                    "problem": config["problem"],
                    "dimension": int(config["dimension"]),
                    "seed": int(config["seed"]),
                    "generation": generation,
                    "phase": phase,
                    "selected_true_rank_percentile": percentile,
                    "selected_in_true_top_10pct": percentile <= 0.10,
                    **opportunity,
                }
            )
    return output


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["selector"], row["suite"], row["problem"],
            row["dimension"], row["phase"],
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, group in groups.items():
        opportunities = [
            row for row in group if row["pool_improves_incumbent"]
        ]
        feasible_opportunities = [
            row for row in group if row["pool_has_feasible"]
        ]
        output.append(
            {
                "selector": key[0],
                "suite": key[1],
                "problem": key[2],
                "dimension": key[3],
                "phase": key[4],
                "batches": len(group),
                "opportunity_rate": len(opportunities) / len(group),
                "capture_rate_given_opportunity": (
                    np.mean(
                        [row["selected_improves_incumbent"] for row in opportunities]
                    )
                    if opportunities
                    else None
                ),
                "feasible_opportunity_rate": (
                    len(feasible_opportunities) / len(group)
                ),
                "feasible_capture_rate_given_opportunity": (
                    np.mean(
                        [row["selected_has_feasible"] for row in feasible_opportunities]
                    )
                    if feasible_opportunities
                    else None
                ),
                "top_10pct_selection_rate": np.mean(
                    [row["selected_in_true_top_10pct"] for row in group]
                ),
                "median_true_rank_percentile": float(
                    np.median(
                        [row["selected_true_rank_percentile"] for row in group]
                    )
                ),
            }
        )
    return output


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


def replay_candidate_selectors(
    result_root: Path, output_directory: Path | None = None
) -> tuple[Path, Path]:
    pool_paths = sorted(result_root.rglob("candidate_pools.npz"))
    if not pool_paths:
        raise ValueError(f"No candidate pools found below {result_root}.")
    rows: list[dict[str, Any]] = []
    for pool_path in pool_paths:
        rows.extend(replay_run(pool_path.parent))
    summary = _summarize(rows)
    destination = output_directory or result_root / "selector_replay"
    destination.mkdir(parents=True, exist_ok=True)
    rows_path = destination / "selector_replay_rows.csv"
    summary_path = destination / "selector_replay_summary.csv"
    _write_rows(rows_path, rows)
    _write_rows(summary_path, summary)
    return rows_path, summary_path
