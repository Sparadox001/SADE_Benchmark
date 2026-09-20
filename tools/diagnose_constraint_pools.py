"""Diagnose per-constraint supply and selection in saved SADE candidate pools."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from sade_benchmark.benchmarks import make_problem  # noqa: E402


def _json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    if np.ptp(left[mask]) == 0.0 or np.ptp(right[mask]) == 0.0:
        return float("nan")
    return float(spearmanr(left[mask], right[mask]).statistic)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_indices(
    pool: np.ndarray,
    evaluations: pd.DataFrame,
    generation: int,
    stage: str,
    span: np.ndarray,
) -> np.ndarray:
    selected = evaluations[
        (evaluations["generation"] == generation)
        & (evaluations["source"] == stage)
    ]
    indices: list[int] = []
    for value in selected["x"]:
        point = _json_array(value)
        distances = np.sqrt(np.mean(((pool - point) / span) ** 2, axis=1))
        index = int(np.argmin(distances))
        if distances[index] <= 1e-10:
            indices.append(index)
    return np.asarray(indices, dtype=int)


def diagnose_run(run_directory: Path) -> list[dict[str, Any]]:
    with (run_directory / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    problem = make_problem(
        config["suite"], config["problem"], int(config["dimension"])
    )
    tolerance = float(config["algorithm_config"]["constraint_tolerance"])
    history = pd.read_csv(run_directory / "history.csv")
    scales = _json_array(history.iloc[0]["constraint_scales"])
    evaluations = pd.read_csv(run_directory / "evaluations.csv.gz")
    evaluated_constraints = np.vstack(
        evaluations["constraints"].map(_json_array).to_numpy()
    )
    evaluated_normalized = np.maximum(
        (evaluated_constraints - tolerance) / scales,
        0.0,
    )

    saved = np.load(run_directory / "candidate_pools.npz")
    offsets = saved["offsets"]
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    span = upper - lower
    rows: list[dict[str, Any]] = []

    for pool_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        generation = int(saved["generation"][pool_index])
        stage = str(saved["stage"][pool_index])
        phase = str(saved["acquisition_phase"][pool_index])
        pool = np.asarray(saved["x"][start:end], dtype=float)
        _, constraints = problem.evaluate(pool)
        normalized = np.maximum((constraints - tolerance) / scales, 0.0)
        sum_cv = np.sum(normalized, axis=1)
        max_cv = np.max(normalized, axis=1)
        satisfied = np.sum(constraints <= tolerance, axis=1)
        feasible = satisfied == problem.n_constraints

        prior_mask = evaluations["generation"].to_numpy() < generation
        prior = evaluated_normalized[prior_mask]
        prior_sum = np.sum(prior, axis=1)
        prior_max = np.max(prior, axis=1)
        prior_satisfied = np.sum(prior <= 0.0, axis=1)
        archive_best_sum = float(np.min(prior_sum))
        archive_best_max = float(np.min(prior_max))
        archive_best_satisfied = int(np.max(prior_satisfied))

        selected = _selected_indices(
            pool, evaluations, generation, stage, span
        )
        selected_sum = sum_cv[selected] if len(selected) else np.empty(0)
        selected_max = max_cv[selected] if len(selected) else np.empty(0)
        selected_satisfied = (
            satisfied[selected] if len(selected) else np.empty(0, dtype=int)
        )
        selected_feasible = feasible[selected] if len(selected) else np.empty(0)

        sum_best = int(np.lexsort((max_cv, sum_cv))[0])
        worst_best = int(np.lexsort((sum_cv, max_cv))[0])
        predicted_constraints = np.asarray(
            saved["predicted_constraints"][start:end], dtype=float
        )
        raw_feasible_values = np.asarray(
            saved["predicted_feasible"][start:end], dtype=float
        )
        robust_feasible_values = np.asarray(
            saved["robust_predicted_feasible"][start:end], dtype=float
        )
        raw_predicted_feasible = np.isfinite(raw_feasible_values) & (
            raw_feasible_values > 0.5
        )
        robust_predicted_feasible = np.isfinite(robust_feasible_values) & (
            robust_feasible_values > 0.5
        )
        margins = np.asarray(
            saved["constraint_error_quantiles"][start:end], dtype=float
        )
        conservative = (
            (predicted_constraints - tolerance) / scales + margins
        )
        predicted_sum = np.sum(np.maximum(conservative, 0.0), axis=1)
        predicted_max = np.max(np.maximum(conservative, 0.0), axis=1)
        uncertainty = np.nan_to_num(
            np.asarray(saved["surrogate_uncertainty"][start:end], dtype=float),
            nan=-np.inf,
        )
        predicted_sum_best = int(
            np.lexsort((-uncertainty, predicted_sum))[0]
        )
        predicted_worst_best = int(
            np.lexsort((-uncertainty, predicted_sum, predicted_max))[0]
        )
        predicted_sum_rank = rankdata(
            np.nan_to_num(predicted_sum, nan=np.inf), method="average"
        ) / len(pool)
        shortlist_size = max(
            1,
            int(
                np.ceil(
                    float(config["algorithm_config"]["near_feasible_fraction"])
                    * len(pool)
                )
            ),
        )
        low_cv_shortlist = np.argsort(predicted_sum)[:shortlist_size]
        constraint_spearman = [
            _safe_spearman(conservative[:, index], normalized[:, index])
            for index in range(problem.n_constraints)
        ]
        sum_rank = rankdata(sum_cv, method="average") / len(pool)
        max_rank = rankdata(max_cv, method="average") / len(pool)

        pool_satisfaction = np.mean(constraints <= tolerance, axis=0)
        selected_satisfaction = (
            np.mean(constraints[selected] <= tolerance, axis=0)
            if len(selected)
            else np.full(problem.n_constraints, np.nan)
        )
        row: dict[str, Any] = {
            "algorithm": config["algorithm"],
            "suite": config["suite"],
            "problem": config["problem"],
            "dimension": int(config["dimension"]),
            "seed": int(config["seed"]),
            "generation": generation,
            "acquisition_phase": phase,
            "stage": stage,
            "pool_size": len(pool),
            "feasible_candidates": int(np.count_nonzero(feasible)),
            "all_but_one_candidates": int(
                np.count_nonzero(satisfied >= problem.n_constraints - 1)
            ),
            "best_satisfied_constraints": int(np.max(satisfied)),
            "best_sum_cv": float(sum_cv[sum_best]),
            "best_max_cv": float(max_cv[worst_best]),
            "sum_best_max_cv": float(max_cv[sum_best]),
            "worst_best_sum_cv": float(sum_cv[worst_best]),
            "sum_and_worst_best_differ": bool(sum_best != worst_best),
            "archive_best_sum_cv_before": archive_best_sum,
            "archive_best_max_cv_before": archive_best_max,
            "archive_best_satisfied_before": archive_best_satisfied,
            "pool_improves_archive_sum_cv": bool(
                sum_cv[sum_best] < archive_best_sum
            ),
            "pool_improves_archive_max_cv": bool(
                max_cv[worst_best] < archive_best_max
            ),
            "pool_improves_archive_satisfied_count": bool(
                np.max(satisfied) > archive_best_satisfied
            ),
            "selected_points_matched": len(selected),
            "selected_feasible_points": int(np.count_nonzero(selected_feasible)),
            "selected_best_satisfied_constraints": (
                int(np.max(selected_satisfied)) if len(selected) else None
            ),
            "selected_best_sum_cv": (
                float(np.min(selected_sum)) if len(selected) else None
            ),
            "selected_best_max_cv": (
                float(np.min(selected_max)) if len(selected) else None
            ),
            "selected_best_sum_rank_fraction": (
                float(np.min(sum_rank[selected])) if len(selected) else None
            ),
            "selected_best_max_rank_fraction": (
                float(np.min(max_rank[selected])) if len(selected) else None
            ),
            "pool_has_more_satisfied_than_selected": bool(
                len(selected)
                and np.max(satisfied) > np.max(selected_satisfied)
            ),
            "pool_has_lower_sum_than_selected": bool(
                len(selected) and sum_cv[sum_best] < np.min(selected_sum)
            ),
            "pool_has_lower_max_than_selected": bool(
                len(selected) and max_cv[worst_best] < np.min(selected_max)
            ),
            "feasible_candidate_missed": bool(
                np.any(feasible)
                and (not len(selected) or not np.any(selected_feasible))
            ),
            "predicted_true_sum_cv_spearman": _safe_spearman(
                predicted_sum, sum_cv
            ),
            "predicted_true_max_cv_spearman": _safe_spearman(
                predicted_max, max_cv
            ),
            "predicted_sum_rule_true_feasible": bool(
                feasible[predicted_sum_best]
            ),
            "predicted_sum_rule_true_satisfied_constraints": int(
                satisfied[predicted_sum_best]
            ),
            "predicted_sum_rule_true_sum_cv": float(
                sum_cv[predicted_sum_best]
            ),
            "predicted_sum_rule_true_max_cv": float(
                max_cv[predicted_sum_best]
            ),
            "predicted_worst_rule_true_feasible": bool(
                feasible[predicted_worst_best]
            ),
            "predicted_worst_rule_true_satisfied_constraints": int(
                satisfied[predicted_worst_best]
            ),
            "predicted_worst_rule_true_sum_cv": float(
                sum_cv[predicted_worst_best]
            ),
            "predicted_worst_rule_true_max_cv": float(
                max_cv[predicted_worst_best]
            ),
            "predicted_worst_rule_improves_true_max": bool(
                max_cv[predicted_worst_best] < max_cv[predicted_sum_best]
            ),
            "predicted_worst_rule_improves_satisfied_count": bool(
                satisfied[predicted_worst_best]
                > satisfied[predicted_sum_best]
            ),
            "true_feasible_raw_predicted_feasible_count": int(
                np.count_nonzero(feasible & raw_predicted_feasible)
            ),
            "true_feasible_robust_predicted_feasible_count": int(
                np.count_nonzero(feasible & robust_predicted_feasible)
            ),
            "best_true_feasible_predicted_sum_rank_fraction": (
                float(np.min(predicted_sum_rank[feasible]))
                if np.any(feasible)
                else None
            ),
            "low_cv_shortlist_contains_true_feasible": bool(
                np.any(feasible[low_cv_shortlist])
            ),
            "constraint_prediction_spearman": json.dumps(
                constraint_spearman, separators=(",", ":")
            ),
            "pool_min_normalized_violation_by_constraint": json.dumps(
                np.min(normalized, axis=0).tolist(), separators=(",", ":")
            ),
            "pool_constraint_satisfaction_rates": json.dumps(
                pool_satisfaction.tolist(), separators=(",", ":")
            ),
            "selected_constraint_satisfaction_rates": json.dumps(
                selected_satisfaction.tolist(), separators=(",", ":")
            ),
            "run_directory": str(run_directory.resolve()),
        }
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    keys = ["algorithm", "suite", "problem", "dimension", "seed"]
    for values, group in frame.groupby(keys, sort=True):
        feasibility = group[group["acquisition_phase"] == "feasibility"]
        source = feasibility if len(feasibility) else group
        constraint_rates = np.vstack(
            source["pool_constraint_satisfaction_rates"].map(_json_array)
        )
        constraint_correlations = np.vstack(
            source["constraint_prediction_spearman"].map(_json_array)
        )
        pool_constraint_minima = np.vstack(
            source["pool_min_normalized_violation_by_constraint"].map(
                _json_array
            )
        )
        feasible_generations = group.loc[
            group["feasible_candidates"] > 0, "generation"
        ]
        evaluations = pd.read_csv(
            Path(group.iloc[0]["run_directory"]) / "evaluations.csv.gz"
        )
        run_directory = Path(group.iloc[0]["run_directory"])
        run_history = pd.read_csv(run_directory / "history.csv")
        run_scales = _json_array(run_history.iloc[0]["constraint_scales"])
        with (run_directory / "config.json").open(encoding="utf-8") as stream:
            run_config = json.load(stream)
        run_tolerance = float(
            run_config["algorithm_config"]["constraint_tolerance"]
        )
        archive_constraints = np.vstack(
            evaluations["constraints"].map(_json_array).to_numpy()
        )
        archive_normalized = np.maximum(
            (archive_constraints - run_tolerance) / run_scales,
            0.0,
        )
        truly_feasible = evaluations[evaluations["feasible"].astype(bool)]
        summaries.append(
            {
                **dict(zip(keys, values)),
                "pools": len(group),
                "feasibility_phase_pools": len(feasibility),
                "pools_with_feasible_candidate": int(
                    np.count_nonzero(source["feasible_candidates"] > 0)
                ),
                "feasible_candidates_total": int(
                    source["feasible_candidates"].sum()
                ),
                "pools_with_all_but_one_candidate": int(
                    np.count_nonzero(source["all_but_one_candidates"] > 0)
                ),
                "pools_improving_archive_sum_cv": int(
                    source["pool_improves_archive_sum_cv"].sum()
                ),
                "pools_improving_archive_max_cv": int(
                    source["pool_improves_archive_max_cv"].sum()
                ),
                "pools_with_more_satisfied_than_selected": int(
                    source["pool_has_more_satisfied_than_selected"].sum()
                ),
                "pools_with_lower_sum_than_selected": int(
                    source["pool_has_lower_sum_than_selected"].sum()
                ),
                "pools_with_lower_max_than_selected": int(
                    source["pool_has_lower_max_than_selected"].sum()
                ),
                "feasible_candidate_missed_pools": int(
                    source["feasible_candidate_missed"].sum()
                ),
                "sum_and_worst_best_differ_pools": int(
                    source["sum_and_worst_best_differ"].sum()
                ),
                "median_selected_sum_rank_fraction": float(
                    source["selected_best_sum_rank_fraction"].median()
                ),
                "median_selected_max_rank_fraction": float(
                    source["selected_best_max_rank_fraction"].median()
                ),
                "median_predicted_true_sum_cv_spearman": float(
                    source["predicted_true_sum_cv_spearman"].median()
                ),
                "median_predicted_true_max_cv_spearman": float(
                    source["predicted_true_max_cv_spearman"].median()
                ),
                "predicted_sum_rule_feasible_selections": int(
                    source["predicted_sum_rule_true_feasible"].sum()
                ),
                "predicted_worst_rule_feasible_selections": int(
                    source["predicted_worst_rule_true_feasible"].sum()
                ),
                "predicted_worst_rule_improves_true_max_pools": int(
                    source["predicted_worst_rule_improves_true_max"].sum()
                ),
                "predicted_worst_rule_improves_satisfied_count_pools": int(
                    source[
                        "predicted_worst_rule_improves_satisfied_count"
                    ].sum()
                ),
                "true_feasible_candidates_raw_predicted_feasible": int(
                    source[
                        "true_feasible_raw_predicted_feasible_count"
                    ].sum()
                ),
                "true_feasible_candidates_robust_predicted_feasible": int(
                    source[
                        "true_feasible_robust_predicted_feasible_count"
                    ].sum()
                ),
                "feasible_pools_with_true_feasible_in_low_cv_shortlist": int(
                    source[
                        "low_cv_shortlist_contains_true_feasible"
                    ].sum()
                ),
                "median_best_true_feasible_predicted_sum_rank_fraction": (
                    float(
                        source[
                            "best_true_feasible_predicted_sum_rank_fraction"
                        ].median()
                    )
                ),
                "median_constraint_prediction_spearman": json.dumps(
                    np.nanmedian(constraint_correlations, axis=0).tolist(),
                    separators=(",", ":"),
                ),
                "minimum_pool_normalized_violation_by_constraint": json.dumps(
                    np.min(pool_constraint_minima, axis=0).tolist(),
                    separators=(",", ":"),
                ),
                "minimum_archive_normalized_violation_by_constraint": (
                    json.dumps(
                        np.min(archive_normalized, axis=0).tolist(),
                        separators=(",", ":"),
                    )
                ),
                "pool_constraint_satisfaction_rates": json.dumps(
                    np.mean(constraint_rates, axis=0).tolist(),
                    separators=(",", ":"),
                ),
                "earliest_generation_with_feasible_candidate": (
                    int(feasible_generations.min())
                    if len(feasible_generations)
                    else None
                ),
                "first_true_feasible_evaluation": (
                    int(truly_feasible["evaluation"].min())
                    if len(truly_feasible)
                    else None
                ),
                "run_directory": group.iloc[0]["run_directory"],
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for run_directory in args.run_directories:
        rows.extend(diagnose_run(run_directory.resolve()))
    summaries = summarize(rows)
    _write_rows(args.output_dir / "constraint_pool_diagnostics_by_pool.csv", rows)
    _write_rows(args.output_dir / "constraint_pool_diagnostics_summary.csv", summaries)
    for row in summaries:
        print(
            f"{row['problem']} D={row['dimension']} seed={row['seed']}: "
            f"feasible pools={row['pools_with_feasible_candidate']}/"
            f"{row['feasibility_phase_pools']}, "
            f"missed={row['feasible_candidate_missed_pools']}, "
            f"sum/max selectors differ="
            f"{row['sum_and_worst_best_differ_pools']}"
        )


if __name__ == "__main__":
    main()
