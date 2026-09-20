"""Merge a completed SADE-dynamic run with the historical SADE/DSI experiment.

The command validates the paired run keys before creating anything.  It then
copies all per-run artifacts into a new, self-contained result directory and
rewrites ``run_directory`` values to point at the copies.  Source experiments
are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path


KEY_FIELDS = ("suite", "problem", "dimension", "run", "seed")
RESULT_FIELDS = (
    "status",
    "objective",
    "objective_error",
    "total_violation",
    "feasible",
    "evaluations",
    "generations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a new SADE/DSI/SADE-dynamic comparison directory."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--dynamic-source", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(root: Path) -> list[dict[str, str]]:
    path = root / "summary_runs.csv"
    if not path.is_file():
        raise ValueError(f"Missing summary file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_source_root"] = str(root)
    return rows


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def require_ok(rows: list[dict[str, str]], label: str) -> None:
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        raise ValueError(f"{label} contains {len(failures)} failed run(s)")


def index_unique(
    rows: list[dict[str, str]], algorithm: str, label: str
) -> dict[tuple[str, ...], dict[str, str]]:
    selected = [row for row in rows if row.get("algorithm") == algorithm]
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in selected:
        item_key = key(row)
        if item_key in indexed:
            raise ValueError(f"Duplicate {label} run key: {item_key}")
        indexed[item_key] = row
    return indexed


def collect_dynamic(
    roots: list[Path],
) -> tuple[dict[tuple[str, ...], dict[str, str]], int]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    duplicates = 0
    for root in roots:
        rows = read_rows(root)
        require_ok(rows, str(root))
        for row in rows:
            if row.get("algorithm") != "sade_dynamic":
                continue
            item_key = key(row)
            previous = indexed.get(item_key)
            if previous is not None:
                mismatches = [
                    field
                    for field in RESULT_FIELDS
                    if previous.get(field) != row.get(field)
                ]
                if mismatches:
                    raise ValueError(
                        f"Conflicting duplicate dynamic run {item_key}: "
                        f"{', '.join(mismatches)}"
                    )
                duplicates += 1
            # Later sources intentionally win.  This lets a complete shard
            # replace a matching run from an interrupted preliminary job.
            indexed[item_key] = row
    return indexed, duplicates


def copied_run_directory(output: Path, row: dict[str, str]) -> Path:
    return (
        output
        / row["algorithm"]
        / row["suite"]
        / row["problem"]
        / f"D{row['dimension']}"
        / Path(row["run_directory"]).name
    )


def copy_dynamic_runs(
    output: Path, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for row in rows:
        source = Path(row["run_directory"])
        if not source.is_dir():
            raise ValueError(f"Missing dynamic run directory: {source}")
        target = copied_run_directory(output, row)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        output_row = {k: v for k, v in row.items() if not k.startswith("_")}
        output_row["run_directory"] = str(target.resolve())
        copied.append(output_row)
    return copied


def rewrite_baseline_rows(
    output: Path, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    rewritten: list[dict[str, str]] = []
    for row in rows:
        output_row = {k: v for k, v in row.items() if not k.startswith("_")}
        output_row["run_directory"] = str(copied_run_directory(output, row).resolve())
        rewritten.append(output_row)
    return rewritten


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_plan(
    baseline: Path,
    dynamic_sources: list[Path],
    output: Path,
    row_count: int,
    duplicate_count: int,
) -> dict:
    baseline_plan = json.loads(
        (baseline / "experiment_config.json").read_text(encoding="utf-8")
    )
    dynamic_plan = json.loads(
        (dynamic_sources[0] / "experiment_config.json").read_text(encoding="utf-8")
    )
    plan = deepcopy(baseline_plan)
    experiment = plan["resolved_configuration"]["experiment"]
    experiment["name"] = output.name
    experiment["algorithms"] = ["sade", "dsi", "sade_dynamic"]
    experiment["output_dir"] = str(output.resolve())
    algorithm_configs = plan["resolved_configuration"]["algorithms"]
    algorithm_configs["sade_dynamic"] = dynamic_plan["resolved_configuration"][
        "algorithms"
    ]["sade_dynamic"]
    plan["source_config"] = None
    plan["total_runs"] = row_count
    plan["statistics_method"]["control_algorithm"] = "sade"
    plan["merge_provenance"] = {
        "baseline_source": str(baseline.resolve()),
        "dynamic_sources": [str(path.resolve()) for path in dynamic_sources],
        "duplicate_dynamic_rows_replaced_by_later_sources": duplicate_count,
        "source_directories_preserved": True,
    }
    return plan


def main() -> None:
    args = parse_args()
    baseline = args.baseline.resolve()
    dynamic_sources = [path.resolve() for path in args.dynamic_source]
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    baseline_rows = read_rows(baseline)
    require_ok(baseline_rows, str(baseline))
    sade = index_unique(baseline_rows, "sade", "SADE")
    dsi = index_unique(baseline_rows, "dsi", "DSI")
    if set(sade) != set(dsi):
        raise SystemExit("Historical SADE and DSI run keys are not paired")

    dynamic, duplicate_count = collect_dynamic(dynamic_sources)
    if set(dynamic) != set(sade):
        missing = sorted(set(sade) - set(dynamic))
        unexpected = sorted(set(dynamic) - set(sade))
        raise SystemExit(
            "Dynamic run keys do not match the historical experiment: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    output.mkdir(parents=True)
    shutil.copytree(baseline / "sade", output / "sade")
    shutil.copytree(baseline / "dsi", output / "dsi")

    algorithm_order = {"sade": 0, "dsi": 1, "sade_dynamic": 2}
    baseline_output = rewrite_baseline_rows(output, baseline_rows)
    dynamic_output = copy_dynamic_runs(output, list(dynamic.values()))
    combined = baseline_output + dynamic_output
    combined.sort(
        key=lambda row: (
            algorithm_order[row["algorithm"]],
            row["suite"],
            row["problem"],
            int(row["dimension"]),
            int(row["run"]),
            int(row["seed"]),
        )
    )
    write_rows(output / "summary_runs.csv", combined)
    plan = build_plan(
        baseline,
        dynamic_sources,
        output,
        len(combined),
        duplicate_count,
    )
    (output / "experiment_config.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Created {output} with {len(combined)} rows; "
        f"replaced {duplicate_count} identical duplicate dynamic row(s)."
    )


if __name__ == "__main__":
    main()
