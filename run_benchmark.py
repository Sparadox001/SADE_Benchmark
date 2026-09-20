"""Command-line entry point for one or more seeded SADE benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from sade_benchmark import SADE, SADEConfig
from sade_benchmark.benchmarks import BENCHMARKS, make_problem


def scientific(value: float) -> str:
    """Format a scalar like the scientific notation normally used in papers."""

    return f"{float(value):.16e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the generic continuous SADE on a selected CEC problem."
    )
    parser.add_argument("--list", action="store_true", help="List retained problems and exit.")
    parser.add_argument("--suite", choices=tuple(BENCHMARKS))
    parser.add_argument("--problem", help="For example g01, c07, or c28.")
    parser.add_argument("--dimension", type=int, help="Omit for fixed-D CEC2006.")
    parser.add_argument("--max-evals", type=int, default=300)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1, help="First run's seed.")
    parser.add_argument("--output", type=Path, help="Optional CSV result path.")
    return parser.parse_args()


def result_row(problem: Any, run: int, seed: int, result: Any) -> dict[str, Any]:
    return {
        "suite": problem.suite,
        "problem": problem.name,
        "dimension": problem.dimension,
        "run": run,
        "seed": seed,
        "objective": scientific(result.objective),
        "total_violation": float(np.sum(result.violation)),
        "feasible": result.feasible,
        "evaluations": result.evaluations,
        "generations": result.generations,
        "x": json.dumps(result.x.tolist(), separators=(",", ":")),
        "constraints": json.dumps(result.constraints.tolist(), separators=(",", ":")),
    }


def main() -> None:
    args = parse_args()
    if args.list:
        print(json.dumps(BENCHMARKS, indent=2))
        return
    if args.suite is None or args.problem is None:
        raise SystemExit("--suite and --problem are required unless --list is used.")
    if args.runs < 1:
        raise SystemExit("--runs must be positive.")

    rows: list[dict[str, Any]] = []
    for run in range(1, args.runs + 1):
        seed = args.seed + run - 1
        problem = make_problem(args.suite, args.problem, args.dimension)
        config = SADEConfig(
            max_evaluations=args.max_evals,
            population_size=args.pop_size,
            batch_size=args.batch_size,
            seed=seed,
        )
        result = SADE(problem, config).optimize()
        row = result_row(problem, run, seed, result)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} run(s) to {args.output.resolve()}")


if __name__ == "__main__":
    main()
