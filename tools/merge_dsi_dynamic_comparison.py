"""Build a paired DSI versus DSI-dynamic comparison result directory.

The source experiments are kept untouched.  The merged ``summary_runs.csv``
stores absolute paths to the original per-run directories, avoiding a second
copy of the evaluation histories while preserving full traceability.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sade_benchmark.experiments import paper_statistics_rows, write_rows


KEY_FIELDS = ("suite", "problem", "dimension", "run", "seed")
EXPECTED_RUNS = 1075
EXPECTED_INSTANCES = 43
EXPECTED_RUNS_PER_INSTANCE = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge paired DSI and DSI-dynamic benchmark summaries."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing summary file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def run_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def instance_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["suite"]), str(row["problem"]), str(row["dimension"])


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output.pop("", None)
    output["dimension"] = int(output["dimension"])
    output["run"] = int(output["run"])
    output["seed"] = int(output["seed"])
    output["evaluations"] = int(output["evaluations"])
    output["generations"] = int(output["generations"])
    if not output.get("objective_numeric"):
        output["objective_numeric"] = float(output["objective"])
    if output.get("objective_error") and not output.get("objective_error_numeric"):
        output["objective_error_numeric"] = float(output["objective_error"])
    output["run_directory"] = str(Path(output["run_directory"]).resolve())
    return output


def select_unique(
    rows: list[dict[str, Any]], algorithm: str, label: str
) -> dict[tuple[str, ...], dict[str, Any]]:
    selected = [normalize_row(row) for row in rows if row.get("algorithm") == algorithm]
    if len(selected) != EXPECTED_RUNS:
        raise ValueError(
            f"{label} has {len(selected)} run rows; expected {EXPECTED_RUNS}."
        )
    indexed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in selected:
        if row.get("status") != "ok":
            raise ValueError(f"{label} contains a failed run: {run_key(row)}")
        key = run_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate {label} run key: {key}")
        directory = Path(row["run_directory"])
        if not directory.is_dir() or not (directory / "final_result.json").is_file():
            raise ValueError(f"Missing artifacts for {label} run: {directory}")
        indexed[key] = row

    counts = Counter(instance_key(row) for row in selected)
    if len(counts) != EXPECTED_INSTANCES:
        raise ValueError(
            f"{label} has {len(counts)} instances; expected {EXPECTED_INSTANCES}."
        )
    invalid = {key: count for key, count in counts.items() if count != EXPECTED_RUNS_PER_INSTANCE}
    if invalid:
        raise ValueError(f"{label} instance run counts are invalid: {invalid}")
    return indexed


def dynamic_rows(root: Path) -> list[dict[str, Any]]:
    summaries = sorted(root.glob("*/summary_runs.csv"))
    if len(summaries) != 8:
        raise ValueError(f"Found {len(summaries)} dynamic shard summaries; expected 8.")
    rows: list[dict[str, Any]] = []
    for path in summaries:
        shard_rows = read_csv(path)
        unexpected = {
            row.get("algorithm") for row in shard_rows if row.get("algorithm") != "dsi_dynamic"
        }
        if unexpected:
            raise ValueError(f"Unexpected algorithms in {path}: {sorted(unexpected)}")
        rows.extend(shard_rows)
    return rows


def build_config(
    baseline: Path, dynamic_root: Path, output: Path, row_count: int
) -> dict[str, Any]:
    return {
        "experiment": {
            "name": output.name,
            "algorithms": ["dsi", "dsi_dynamic"],
            "control_algorithm": "dsi",
            "runs_per_instance": EXPECTED_RUNS_PER_INSTANCE,
            "max_function_evaluations": 300,
            "instances": EXPECTED_INSTANCES,
            "total_rows": row_count,
        },
        "resolved_configuration": {
            "experiment": {"algorithms": ["dsi", "dsi_dynamic"]}
        },
        "statistics_method": {
            "control_algorithm": "dsi",
            "alpha": 0.05,
            "infeasible_replacement_value": 10e20,
            "symbols": {
                "+": "DSI significantly better",
                "−": "DSI significantly worse",
                "≈": "no significant paired WSR difference",
            },
        },
        "merge_provenance": {
            "baseline_summary": str((baseline / "summary_runs.csv").resolve()),
            "dynamic_shard_root": str(dynamic_root.resolve()),
            "run_directories_are_source_references": True,
            "source_directories_modified": False,
        },
    }


def main() -> None:
    args = parse_args()
    baseline = args.baseline.resolve()
    dynamic_root = args.dynamic_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    dsi = select_unique(read_csv(baseline / "summary_runs.csv"), "dsi", "DSI")
    dynamic = select_unique(dynamic_rows(dynamic_root), "dsi_dynamic", "DSI-dynamic")
    if set(dsi) != set(dynamic):
        missing = sorted(set(dsi) - set(dynamic))
        unexpected = sorted(set(dynamic) - set(dsi))
        raise SystemExit(
            "Paired run keys differ: "
            f"missing_dynamic={len(missing)}, unexpected_dynamic={len(unexpected)}"
        )

    combined = list(dsi.values()) + list(dynamic.values())
    order = {"dsi": 0, "dsi_dynamic": 1}
    combined.sort(
        key=lambda row: (
            order[row["algorithm"]],
            row["suite"],
            row["problem"],
            int(row["dimension"]),
            int(row["run"]),
            int(row["seed"]),
        )
    )

    output.mkdir(parents=True)
    write_rows(output / "summary_runs.csv", combined)
    statistics = paper_statistics_rows(
        combined, ["dsi", "dsi_dynamic"], control_algorithm="dsi"
    )
    write_rows(output / "summary_statistics.csv", statistics)
    config = build_config(baseline, dynamic_root, output, len(combined))
    (output / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Created {output} with {len(combined)} paired rows and "
        f"{len(statistics)} statistics rows."
    )


if __name__ == "__main__":
    main()
