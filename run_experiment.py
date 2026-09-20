"""Run multiple algorithms, suites, problems, dimensions, and seeded runs."""

from __future__ import annotations

import argparse
import copy
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from sade_benchmark.benchmarks import BENCHMARKS, make_problem
from sade_benchmark.experiments import (
    ALGORITHMS,
    DEFAULT_ALPHA,
    DEFAULT_INFEASIBLE_VALUE,
    build_cases,
    load_configuration,
    make_algorithm_config,
    make_optimizer,
    paper_statistics_rows,
    save_run,
    scientific,
    write_json,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible multi-algorithm constrained benchmark experiments."
    )
    parser.add_argument("--list", action="store_true", help="List algorithms and cases.")
    parser.add_argument("--config", type=Path, help="JSON experiment configuration.")
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help=(
            "sade, sade_dynamic, sade_targeted, sade_v2, sade_v2_2, sade_v2_3, sade_v2_4, "
            "sade_v2_5, sade_v2_6, sade_v2_7, sade_v2_8, sade_v2_9, "
            "sade_v2_10, sade_v2_11, sade_v2_12, sade_v2_13, dsi, "
            "dsi_dynamic, or all."
        ),
    )
    parser.add_argument(
        "--suites", "--suite", nargs="+", default=None,
        help=(
            "CEC suites, cec2017_objcap, cec2017_objcap_first_mean, "
            "or all (original suites only)."
        )
    )
    parser.add_argument(
        "--problems", "--problem", nargs="+", default=None, help="Names, or all."
    )
    parser.add_argument(
        "--dimensions",
        "--dimension",
        nargs="+",
        default=None,
        help="Dimensions such as 10 30 50 100, or all; ignored for fixed-D CEC2006.",
    )
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="First run's seed.")
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--constraint-tolerance", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="SADE variants only.")
    parser.add_argument("--trials-per-target", type=int, default=None, help="SADE variants only.")
    parser.add_argument("--local-fraction", type=float, default=None, help="SADE variants only.")
    parser.add_argument(
        "--surrogate-min-samples", type=int, default=None, help="SADE variants only."
    )
    parser.add_argument("--p-best-fraction", type=float, default=None, help="SADE variants only.")
    parser.add_argument(
        "--distance-threshold", type=float, default=None,
        help="SADE variants only.",
    )
    parser.add_argument(
        "--dynamic-constraint-tightening",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Dynamic SADE/DSI variants: impose objective <= active limit.",
    )
    parser.add_argument(
        "--tightening-trigger-feasible-count",
        type=int,
        default=None,
        help="Dynamic SADE/DSI variants; defaults to the population size.",
    )
    parser.add_argument(
        "--tightening-keep-ratio", type=float, default=None,
        help="Dynamic SADE/DSI variants only.",
    )
    parser.add_argument(
        "--tightening-patience", type=int, default=None,
        help="Dynamic SADE/DSI variants only.",
    )
    parser.add_argument(
        "--tightening-improvement-tol", type=float, default=None,
        help="Dynamic SADE/DSI variants only.",
    )
    parser.add_argument(
        "--tightening-max-tightens", type=int, default=None,
        help="Dynamic SADE/DSI variants only.",
    )
    parser.add_argument(
        "--tightening-min-ratio", type=float, default=None,
        help="Dynamic SADE/DSI variants only.",
    )
    parser.add_argument(
        "--tightening-constraint-index", type=int, default=None,
        help="SADE targeted variant: index of the existing objective-derived constraint.",
    )
    parser.add_argument(
        "--near-feasible-fraction", type=float, default=None,
        help="SADE V2 variants only.",
    )
    parser.add_argument(
        "--constraint-error-quantile", type=float, default=None,
        help="SADE V2.3 through V2.13 only.",
    )
    parser.add_argument(
        "--constraint-error-window", type=int, default=None,
        help="SADE V2.3 through V2.13 only.",
    )
    parser.add_argument(
        "--constraint-error-min-samples", type=int, default=None,
        help="SADE V2.3 through V2.13 only.",
    )
    parser.add_argument(
        "--constraint-safety-factor", type=float, default=None,
        help="SADE V2.3 through V2.13 only.",
    )
    parser.add_argument(
        "--hard-feasible-min-fraction",
        type=float,
        default=None,
        help="SADE V2.4 through V2.13 only.",
    )
    parser.add_argument(
        "--boundary-stagnation-batches",
        type=int,
        default=None,
        help="SADE V2.7 through V2.13 only.",
    )
    parser.add_argument(
        "--objective-shortlist-fraction",
        type=float,
        default=None,
        help="SADE V2.8 only.",
    )
    parser.add_argument(
        "--objective-shortlist-min-size",
        type=int,
        default=None,
        help="SADE V2.8 only.",
    )
    parser.add_argument(
        "--objective-knn-neighbors",
        type=int,
        default=None,
        help="SADE V2.8 only.",
    )
    parser.add_argument("--wmax", type=int, default=None, help="DSI only.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--save-evaluations", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--save-populations", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--save-candidate-pools", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _algorithm_selection(requested: list[str]) -> tuple[str, ...]:
    normalized = tuple(value.lower() for value in requested)
    if "all" in normalized:
        return ALGORITHMS
    invalid = sorted(set(normalized) - set(ALGORITHMS))
    if invalid:
        raise ValueError(f"Unknown algorithms: {', '.join(invalid)}")
    return tuple(name for name in ALGORITHMS if name in normalized)


def _default_output_directory(name: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "experiment").strip("_")
    return Path("results") / f"{prefix or 'experiment'}_{stamp}"


def _apply_cli_overrides(
    configuration: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Apply only explicitly supplied CLI values over JSON/default values."""

    resolved = copy.deepcopy(configuration)
    cli_dimensions = None
    if args.dimensions is not None:
        cli_dimensions = [
            "all" if str(value).lower() == "all" else int(value)
            for value in args.dimensions
        ]

    experiment_mappings = (
        ("algorithms", args.algorithms),
        ("suites", args.suites),
        ("problems", args.problems),
        ("dimensions", cli_dimensions),
        ("runs", args.runs),
        ("first_seed", args.seed),
        ("output_dir", str(args.output_dir) if args.output_dir is not None else None),
    )
    for key, value in experiment_mappings:
        if value is not None:
            resolved["experiment"][key] = value

    common_mappings = (
        ("max_evaluations", args.max_evals),
        ("population_size", args.pop_size),
        ("constraint_tolerance", args.constraint_tolerance),
    )
    for key, value in common_mappings:
        if value is not None:
            resolved["common"][key] = value

    sade_mappings = (
        ("batch_size", args.batch_size),
        ("trials_per_target", args.trials_per_target),
        ("local_fraction", args.local_fraction),
        ("surrogate_min_samples", args.surrogate_min_samples),
        ("p_best_fraction", args.p_best_fraction),
        ("distance_threshold", args.distance_threshold),
    )
    for key, value in sade_mappings:
        if value is not None:
            resolved["algorithms"]["sade"][key] = value
            resolved["algorithms"]["sade_dynamic"][key] = value
            resolved["algorithms"]["sade_targeted"][key] = value
            resolved["algorithms"]["sade_v2"][key] = value
            resolved["algorithms"]["sade_v2_2"][key] = value
            resolved["algorithms"]["sade_v2_3"][key] = value
            resolved["algorithms"]["sade_v2_4"][key] = value
            resolved["algorithms"]["sade_v2_5"][key] = value
            resolved["algorithms"]["sade_v2_6"][key] = value
            resolved["algorithms"]["sade_v2_7"][key] = value
            resolved["algorithms"]["sade_v2_8"][key] = value
            resolved["algorithms"]["sade_v2_9"][key] = value
            resolved["algorithms"]["sade_v2_10"][key] = value
            resolved["algorithms"]["sade_v2_11"][key] = value
            resolved["algorithms"]["sade_v2_12"][key] = value
            resolved["algorithms"]["sade_v2_13"][key] = value
    tightening_mappings = (
        ("dynamic_constraint_tightening", args.dynamic_constraint_tightening),
        (
            "tightening_trigger_feasible_count",
            args.tightening_trigger_feasible_count,
        ),
        ("tightening_keep_ratio", args.tightening_keep_ratio),
        ("tightening_patience", args.tightening_patience),
        ("tightening_improvement_tol", args.tightening_improvement_tol),
        ("tightening_max_tightens", args.tightening_max_tightens),
        ("tightening_min_ratio", args.tightening_min_ratio),
    )
    for key, value in tightening_mappings:
        if value is not None:
            resolved["algorithms"]["sade"][key] = value
            resolved["algorithms"]["sade_dynamic"][key] = value
            resolved["algorithms"]["sade_targeted"][key] = value
            resolved["algorithms"]["dsi_dynamic"][key] = value
    if args.tightening_constraint_index is not None:
        resolved["algorithms"]["sade_targeted"]["tightening_constraint_index"] = (
            args.tightening_constraint_index
        )
    if args.near_feasible_fraction is not None:
        resolved["algorithms"]["sade_v2"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_2"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_3"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_4"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_5"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_6"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_7"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_8"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_9"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_10"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_11"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_12"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
        resolved["algorithms"]["sade_v2_13"]["near_feasible_fraction"] = (
            args.near_feasible_fraction
        )
    v2_3_mappings = (
        ("constraint_error_quantile", args.constraint_error_quantile),
        ("constraint_error_window", args.constraint_error_window),
        ("constraint_error_min_samples", args.constraint_error_min_samples),
        ("constraint_safety_factor", args.constraint_safety_factor),
    )
    for key, value in v2_3_mappings:
        if value is not None:
            resolved["algorithms"]["sade_v2_3"][key] = value
            resolved["algorithms"]["sade_v2_4"][key] = value
            resolved["algorithms"]["sade_v2_5"][key] = value
            resolved["algorithms"]["sade_v2_6"][key] = value
            resolved["algorithms"]["sade_v2_7"][key] = value
            resolved["algorithms"]["sade_v2_8"][key] = value
            resolved["algorithms"]["sade_v2_9"][key] = value
            resolved["algorithms"]["sade_v2_10"][key] = value
            resolved["algorithms"]["sade_v2_11"][key] = value
            resolved["algorithms"]["sade_v2_12"][key] = value
            resolved["algorithms"]["sade_v2_13"][key] = value
    if args.hard_feasible_min_fraction is not None:
        resolved["algorithms"]["sade_v2_4"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
        resolved["algorithms"]["sade_v2_5"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
        resolved["algorithms"]["sade_v2_6"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
        resolved["algorithms"]["sade_v2_7"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
        resolved["algorithms"]["sade_v2_8"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
        resolved["algorithms"]["sade_v2_9"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
        resolved["algorithms"]["sade_v2_10"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
        resolved["algorithms"]["sade_v2_11"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
        resolved["algorithms"]["sade_v2_12"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
        resolved["algorithms"]["sade_v2_13"]["hard_feasible_min_fraction"] = (
            args.hard_feasible_min_fraction
        )
    if args.boundary_stagnation_batches is not None:
        resolved["algorithms"]["sade_v2_7"]["boundary_stagnation_batches"] = (
            args.boundary_stagnation_batches
        )
        resolved["algorithms"]["sade_v2_8"]["boundary_stagnation_batches"] = (
            args.boundary_stagnation_batches
        )
        resolved["algorithms"]["sade_v2_9"]["boundary_stagnation_batches"] = (
            args.boundary_stagnation_batches
        )
        resolved["algorithms"]["sade_v2_10"]["boundary_stagnation_batches"] = (
            args.boundary_stagnation_batches
        )
        resolved["algorithms"]["sade_v2_11"]["boundary_stagnation_batches"] = (
            args.boundary_stagnation_batches
        )
        resolved["algorithms"]["sade_v2_12"]["boundary_stagnation_batches"] = (
            args.boundary_stagnation_batches
        )
        resolved["algorithms"]["sade_v2_13"]["boundary_stagnation_batches"] = (
            args.boundary_stagnation_batches
        )
    v2_8_mappings = (
        ("objective_shortlist_fraction", args.objective_shortlist_fraction),
        ("objective_shortlist_min_size", args.objective_shortlist_min_size),
        ("objective_knn_neighbors", args.objective_knn_neighbors),
    )
    for key, value in v2_8_mappings:
        if value is not None:
            resolved["algorithms"]["sade_v2_8"][key] = value
    if args.wmax is not None:
        resolved["algorithms"]["dsi"]["wmax"] = args.wmax
        resolved["algorithms"]["dsi_dynamic"]["wmax"] = args.wmax

    recording_mappings = (
        ("save_evaluations", args.save_evaluations),
        ("save_populations", args.save_populations),
        ("save_candidate_pools", args.save_candidate_pools),
    )
    for key, value in recording_mappings:
        if value is not None:
            resolved["recording"][key] = value
    return resolved


def _run_path(root: Path, algorithm: str, problem: Any, run: int, seed: int) -> Path:
    return (
        root
        / algorithm
        / problem.suite
        / problem.name
        / f"D{problem.dimension}"
        / f"run_{run:03d}_seed_{seed}"
    )


def _public_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hide internal numeric helpers; displayed objectives stay scientific."""

    return [
        {key: value for key, value in row.items() if not key.endswith("_numeric")}
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    if args.list:
        print(json.dumps({"algorithms": ALGORITHMS, "benchmarks": BENCHMARKS}, indent=2))
        return

    try:
        configuration = _apply_cli_overrides(load_configuration(args.config), args)
        experiment = configuration["experiment"]
        if experiment["suites"] is None:
            raise ValueError(
                "No suites selected; provide --suites or set experiment.suites "
                "in the JSON configuration."
            )
        if int(experiment["runs"]) < 1:
            raise ValueError("runs must be positive.")
        algorithms = _algorithm_selection(experiment["algorithms"])
        cases = build_cases(
            experiment["suites"],
            experiment["problems"],
            experiment["dimensions"],
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    common = configuration["common"]
    algorithm_configuration = configuration["algorithms"]
    recording = configuration["recording"]
    runs = int(experiment["runs"])
    first_seed = int(experiment["first_seed"])
    try:
        for algorithm in algorithms:
            make_algorithm_config(
                algorithm,
                common_parameters=common,
                algorithm_parameters=algorithm_configuration[algorithm],
                seed=first_seed,
                save_candidate_pools=recording["save_candidate_pools"],
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    total = len(algorithms) * len(cases) * runs
    statistics_control = "sade" if "sade" in algorithms else algorithms[0]
    plan = {
        "source_config": str(args.config.resolve()) if args.config else None,
        "resolved_configuration": configuration,
        "resolved_cases": [
            {"suite": case.suite, "problem": case.problem, "dimension": case.dimension}
            for case in cases
        ],
        "total_runs": total,
        "dimension_scope": {
            "paper_reproduction": [10, 30],
            "optional_extension": [50, 100],
        },
        "statistics_method": {
            "control_algorithm": statistics_control,
            "per_instance_test": "paired two-sided Wilcoxon signed-rank",
            "alpha": DEFAULT_ALPHA,
            "infeasible_value": DEFAULT_INFEASIBLE_VALUE,
            "multiple_comparison": "Friedman average-rank post-hoc with Holm adjustment",
        },
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    configured_output = experiment.get("output_dir")
    output_root = (
        Path(configured_output)
        if configured_output
        else _default_output_directory(experiment.get("name"))
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "experiment_config.json", plan)
    rows: list[dict[str, Any]] = []
    completed = 0

    for algorithm in algorithms:
        for case in cases:
            for run in range(1, runs + 1):
                seed = first_seed + run - 1
                problem = make_problem(case.suite, case.problem, case.dimension)
                run_directory = _run_path(output_root, algorithm, problem, run, seed)
                completed += 1
                try:
                    optimizer, config = make_optimizer(
                        algorithm,
                        problem,
                        common_parameters=common,
                        algorithm_parameters=algorithm_configuration[algorithm],
                        seed=seed,
                        save_candidate_pools=recording["save_candidate_pools"],
                    )
                    result = optimizer.optimize()
                    row = save_run(
                        run_directory,
                        algorithm=algorithm,
                        problem=problem,
                        run=run,
                        seed=seed,
                        config=config,
                        result=result,
                        save_evaluations=recording["save_evaluations"],
                        save_populations=recording["save_populations"],
                    )
                    print(
                        f"[{completed}/{total}] {algorithm} {problem.suite}/{problem.name} "
                        f"D={problem.dimension} run={run} objective={scientific(result.objective)} "
                        f"feasible={result.feasible}"
                    )
                except Exception as error:  # keep a large experiment recoverable
                    run_directory.mkdir(parents=True, exist_ok=True)
                    error_data = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    write_json(run_directory / "error.json", error_data)
                    row = {
                        "status": "failed",
                        "algorithm": algorithm,
                        "suite": problem.suite,
                        "problem": problem.name,
                        "dimension": problem.dimension,
                        "run": run,
                        "seed": seed,
                        "error": f"{type(error).__name__}: {error}",
                        "run_directory": str(run_directory),
                    }
                    print(f"[{completed}/{total}] FAILED {row['error']}")
                    if args.fail_fast:
                        rows.append(row)
                        write_rows(
                            output_root / "summary_runs.csv",
                            _public_summary_rows(rows),
                        )
                        raise
                rows.append(row)
                write_rows(
                    output_root / "summary_runs.csv",
                    _public_summary_rows(rows),
                )

    statistics = paper_statistics_rows(
        rows,
        algorithms,
        control_algorithm=statistics_control,
    )
    write_rows(output_root / "summary_statistics.csv", statistics)

    print(f"Completed {total} run(s). Results: {output_root}")


if __name__ == "__main__":
    main()
