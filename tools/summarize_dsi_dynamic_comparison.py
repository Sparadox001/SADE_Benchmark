"""Print a compact JSON diagnostic for a DSI/DSI-dynamic comparison."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def instance_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["suite"], row["problem"], int(row["dimension"])


def run_key(row: dict[str, str]) -> tuple[str, str, int, int, int]:
    suite, problem, dimension = instance_key(row)
    return suite, problem, dimension, int(row["run"]), int(row["seed"])


def metric(row: dict[str, str]) -> float:
    field = "objective_error_numeric" if row["suite"] == "cec2006" else "objective_numeric"
    return float(row[field])


def score(row: dict[str, str]) -> float:
    return metric(row) if as_bool(row["feasible"]) else 10e20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    return parser.parse_args()


def main() -> None:
    root = parse_args().comparison.resolve()
    rows = read_csv(root / "summary_runs.csv")
    stats = read_csv(root / "summary_statistics.csv")
    by_algorithm = {
        algorithm: {
            run_key(row): row for row in rows if row["algorithm"] == algorithm
        }
        for algorithm in ("dsi", "dsi_dynamic")
    }

    feasibility: dict[str, dict[str, dict[str, float | int]]] = {}
    for algorithm, indexed in by_algorithm.items():
        scopes: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in indexed.values():
            scopes[row["suite"]].append(row)
            scopes["ALL"].append(row)
        feasibility[algorithm] = {}
        for scope, group in scopes.items():
            count = sum(as_bool(row["feasible"]) for row in group)
            feasibility[algorithm][scope] = {
                "feasible": count,
                "runs": len(group),
                "rate": count / len(group),
            }

    pair_outcomes = {"dynamic_better": 0, "tie": 0, "dsi_better": 0}
    for key, dsi_row in by_algorithm["dsi"].items():
        dsi_score = score(dsi_row)
        dynamic_score = score(by_algorithm["dsi_dynamic"][key])
        if np.isclose(dsi_score, dynamic_score, rtol=1e-12, atol=1e-12):
            pair_outcomes["tie"] += 1
        elif dynamic_score < dsi_score:
            pair_outcomes["dynamic_better"] += 1
        else:
            pair_outcomes["dsi_better"] += 1

    rank_summary: dict[str, dict[str, str]] = {}
    count_summary: dict[str, str] = {}
    significant = {"dynamic_better": [], "dsi_better": []}
    for row in stats:
        scope = row["suite"]
        if row["problem"] == "Average Rank":
            rank_summary[scope] = {
                "dsi": row["dsi"],
                "dsi_dynamic": row["dsi_dynamic"],
            }
        elif row["problem"] == "Adjusted p-value":
            rank_summary.setdefault(scope, {})["adjusted_p"] = row["dsi_dynamic"]
        elif row["problem"] == "+/≈/−":
            count_summary[scope] = row["dsi_dynamic"]
        else:
            value = row["dsi_dynamic"]
            label = f"{scope}/{row['problem']}/D{row['dimension']}"
            if value.endswith(" −"):
                significant["dynamic_better"].append(label)
            elif value.endswith(" +"):
                significant["dsi_better"].append(label)

    trigger_records: list[tuple[str, bool, int, int | None]] = []
    for row in by_algorithm["dsi_dynamic"].values():
        path = Path(row["run_directory"]) / "dynamic_constraint_history.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        history = state.get("history", [])
        first_generation = int(history[0]["generation"]) if history else None
        trigger_records.append(
            (
                row["suite"],
                bool(state.get("threshold_active", False)),
                int(state.get("tighten_count", 0)),
                first_generation,
            )
        )

    trigger_summary: dict[str, dict[str, float | int | None]] = {}
    for scope in ("cec2006", "cec2010", "cec2017", "ALL"):
        selected = [item for item in trigger_records if scope == "ALL" or item[0] == scope]
        activated = [item for item in selected if item[1]]
        counts = [item[2] for item in selected]
        first = [item[3] for item in activated if item[3] is not None]
        trigger_summary[scope] = {
            "activated": len(activated),
            "runs": len(selected),
            "activation_rate": len(activated) / len(selected),
            "median_tighten_count": float(np.median(counts)),
            "mean_tighten_count": float(np.mean(counts)),
            "median_first_activation_generation": (
                float(np.median(first)) if first else None
            ),
        }

    output = {
        "feasibility": feasibility,
        "paired_run_outcomes": pair_outcomes,
        "average_ranks": rank_summary,
        "wsr_counts_control_dsi": count_summary,
        "significant_instances": significant,
        "dynamic_trigger": trigger_summary,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
