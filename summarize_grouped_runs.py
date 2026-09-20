"""Merge independently run problem groups and report feasible-objective means."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from sade_benchmark.experiments.persistence import write_json, write_rows
from sade_benchmark.experiments.statistics import paper_statistics_rows


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def summarize(root: Path, *, expected_runs: int, expected_evaluations: int) -> None:
    group_files = sorted(root.glob("group_*/summary_runs.csv"))
    if not group_files:
        raise ValueError(f"No group summaries found below {root}")

    rows: list[dict[str, str]] = []
    keys: set[tuple[str, str, str, int, int]] = set()
    for path in group_files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "ok":
                    raise ValueError(f"Failed run in {path}: {row}")
                if int(row["evaluations"]) != expected_evaluations:
                    raise ValueError(f"Wrong evaluation count in {path}: {row}")
                key = (
                    row["algorithm"],
                    row["suite"],
                    row["problem"],
                    int(row["dimension"]),
                    int(row["run"]),
                )
                if key in keys:
                    raise ValueError(f"Duplicate run: {key}")
                keys.add(key)
                rows.append(row)

    algorithms = ("sade", "sade_targeted", "dsi")
    grouped: dict[tuple[str, str, int], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        instance = (row["suite"], row["problem"], int(row["dimension"]))
        grouped[instance][row["algorithm"]].append(row)

    means: list[dict[str, object]] = []
    for (suite, problem, dimension), by_algorithm in sorted(grouped.items()):
        if set(by_algorithm) != set(algorithms):
            raise ValueError(f"Missing algorithms for {suite}/{problem}/D{dimension}")
        for algorithm in algorithms:
            records = by_algorithm[algorithm]
            if len(records) != expected_runs:
                raise ValueError(
                    f"Expected {expected_runs} runs for "
                    f"{suite}/{problem}/D{dimension}/{algorithm}"
                )
            if {int(r["run"]) for r in records} != set(range(1, expected_runs + 1)):
                raise ValueError(f"Incomplete run numbers for {problem}/{algorithm}")
            feasible_values = [
                float(r["objective"]) for r in records if _as_bool(r["feasible"])
            ]
            means.append(
                {
                    "suite": suite,
                    "problem": problem,
                    "dimension": dimension,
                    "algorithm": algorithm,
                    "feasible_runs": len(feasible_values),
                    "total_runs": expected_runs,
                    "mean_feasible_objective": (
                        f"{mean(feasible_values):.8e}" if feasible_values else ""
                    ),
                    "std_feasible_objective": (
                        f"{stdev(feasible_values):.8e}"
                        if len(feasible_values) > 1
                        else ""
                    ),
                }
            )

    rows.sort(
        key=lambda r: (
            r["suite"], r["problem"], int(r["dimension"]),
            algorithms.index(r["algorithm"]), int(r["run"]),
        )
    )
    write_rows(root / "summary_runs.csv", rows)
    write_rows(root / "mean_objectives.csv", means)
    write_rows(
        root / "summary_statistics.csv",
        paper_statistics_rows(rows, algorithms, control_algorithm="sade"),
    )
    write_json(
        root / "merge_manifest.json",
        {
            "group_summaries": [str(p) for p in group_files],
            "instances": len(grouped),
            "runs": len(rows),
            "expected_evaluations_per_run": expected_evaluations,
            "mean_definition": "Arithmetic mean over feasible runs only; blanks mean none",
        },
    )
    print(f"Merged {len(rows)} runs across {len(grouped)} instances into {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--expected-evaluations", type=int, required=True)
    args = parser.parse_args()
    summarize(
        args.root.resolve(),
        expected_runs=args.expected_runs,
        expected_evaluations=args.expected_evaluations,
    )
