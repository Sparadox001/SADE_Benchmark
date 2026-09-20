"""Validate and summarize the 5-run DSI 1000-FE CEC2017 feasibility pilot."""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


PROBLEMS = ("c01", "c02", "c04", "c05", "c13", "c19", "c20", "c22", "c28")
DIMENSIONS = (10, 30)
SEEDS = (1, 2, 3, 4, 5)
EVALUATIONS = 1000


def read_rows(path: Path, *, gzipped: bool = False) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing data file: {path}")
    opener = gzip.open if gzipped else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_directories(parallel: Path, pilot: Path) -> list[Path]:
    paths = [parallel / name for name in PROBLEMS if name != "c05"]
    paths.extend((parallel / "c05_remaining", parallel / "c05_d10_seed1", pilot))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parallel = args.parallel.resolve()
    pilot = args.pilot.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    indexed: dict[tuple[str, int, int], dict[str, str]] = {}
    for source in source_directories(parallel, pilot):
        for row in read_rows(source / "summary_runs.csv"):
            if row["status"] != "ok" or row["algorithm"] != "dsi":
                raise ValueError(f"Non-DSI or failed row from {source}: {row}")
            key = row["problem"], int(row["dimension"]), int(row["seed"])
            if key in indexed:
                raise ValueError(f"Duplicate run key: {key}")
            indexed[key] = row

    expected = {
        (name, dimension, seed)
        for name in PROBLEMS
        for dimension in DIMENSIONS
        for seed in SEEDS
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"Incomplete run keys: missing={missing}, extra={extra}")

    run_rows: list[dict[str, object]] = []
    by_instance: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for name, dimension, seed in sorted(indexed):
        source_row = indexed[name, dimension, seed]
        directory = Path(source_row["run_directory"]).resolve()
        if int(source_row["evaluations"]) != EVALUATIONS:
            raise ValueError(f"Wrong FE count in {directory}")
        evaluations = read_rows(directory / "evaluations.csv.gz", gzipped=True)
        if len(evaluations) != EVALUATIONS:
            raise ValueError(f"Wrong archive length in {directory}")
        feasible_rows = [
            item
            for item in evaluations
            if item["feasible"].strip().lower() == "true"
        ]
        feasible = bool(feasible_rows)
        if feasible != (source_row["feasible"].strip().lower() == "true"):
            raise ValueError(f"Summary/archive feasibility mismatch: {directory}")
        first_fe = (
            int(feasible_rows[0]["evaluation"]) if feasible_rows else None
        )
        best_feasible = (
            min(float(item["objective"]) for item in feasible_rows)
            if feasible_rows
            else None
        )
        if feasible and not abs(
            best_feasible - float(source_row["objective"])
        ) <= 1e-8 * max(1.0, abs(best_feasible)):
            raise ValueError(f"Final objective/archive best mismatch: {directory}")
        result = {
            "suite": "cec2017",
            "problem": name,
            "dimension": dimension,
            "run": seed,
            "seed": seed,
            "evaluations": EVALUATIONS,
            "successful": feasible,
            "first_feasible_fe": first_fe,
            "feasible_evaluations": len(feasible_rows),
            "best_feasible_objective": (
                f"{best_feasible:.16e}" if best_feasible is not None else ""
            ),
            "run_directory": str(directory),
        }
        run_rows.append(result)
        by_instance[name, dimension].append(result)

    instance_rows: list[dict[str, object]] = []
    for name, dimension in sorted(by_instance):
        group = by_instance[name, dimension]
        successful = [row for row in group if row["successful"]]
        first = [int(row["first_feasible_fe"]) for row in successful]
        best = [float(row["best_feasible_objective"]) for row in successful]
        counts = [int(row["feasible_evaluations"]) for row in group]
        instance_rows.append(
            {
                "suite": "cec2017",
                "problem": name,
                "dimension": dimension,
                "runs": len(group),
                "successful_runs": len(successful),
                "success_rate": f"{len(successful) / len(group):.2%}",
                "total_feasible_evaluations": sum(counts),
                "median_feasible_evaluations_per_run": median(counts),
                "median_first_feasible_fe": median(first) if first else "",
                "best_feasible_min": f"{min(best):.8e}" if best else "",
                "best_feasible_median": f"{median(best):.8e}" if best else "",
                "best_feasible_max": f"{max(best):.8e}" if best else "",
                "best_feasible_mean": f"{mean(best):.8e}" if best else "",
                "best_feasible_std": (
                    f"{stdev(best):.8e}" if len(best) > 1 else ""
                ),
            }
        )

    write_rows(output / "feasibility_by_run.csv", run_rows)
    write_rows(output / "feasibility_by_instance.csv", instance_rows)
    print(f"Validated {len(run_rows)} runs and {len(instance_rows)} instances.")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
