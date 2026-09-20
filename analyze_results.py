"""Regenerate the compact paper-style table for an existing experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sade_benchmark.experiments import (
    DEFAULT_ALPHA,
    DEFAULT_CONTROL,
    DEFAULT_INFEASIBLE_VALUE,
    paper_statistics_rows,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-style WSR and average-rank statistics."
    )
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--control", default=DEFAULT_CONTROL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--infeasible-value", type=float, default=DEFAULT_INFEASIBLE_VALUE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.results_directory.resolve()
    runs_path = root / "summary_runs.csv"
    config_path = root / "experiment_config.json"
    if not runs_path.is_file():
        raise SystemExit(f"Missing run data: {runs_path}")
    with runs_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if config_path.is_file():
        plan = json.loads(config_path.read_text(encoding="utf-8"))
        algorithms = plan["resolved_configuration"]["experiment"]["algorithms"]
    else:
        algorithms = list(dict.fromkeys(row["algorithm"] for row in rows))
    control = args.control.lower()
    if control not in algorithms and len(algorithms) == 1:
        control = algorithms[0]
    try:
        statistics = paper_statistics_rows(
            rows,
            algorithms,
            control_algorithm=control,
            alpha=args.alpha,
            infeasible_value=args.infeasible_value,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output = root / "summary_statistics.csv"
    write_rows(output, statistics)
    print(f"Saved {len(statistics)} paper-style row(s) to {output}")


if __name__ == "__main__":
    main()
