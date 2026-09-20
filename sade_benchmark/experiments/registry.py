"""Algorithm registry, experiment expansion, and aggregate statistics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..algorithms.dsi import DSI, DSIConfig
from ..algorithms.dsi_dynamic import DSIDynamic, DSIDynamicConfig
from ..algorithms.sade import SADE, SADEConfig
from ..algorithms.sade_targeted import SADETargeted, SADETargetedConfig
from ..algorithms.sade_v2 import SADEV2, SADEV2Config
from ..algorithms.sade_v2_2 import SADEV2_2, SADEV2_2Config
from ..algorithms.sade_v2_3 import SADEV2_3, SADEV2_3Config
from ..algorithms.sade_v2_4 import SADEV2_4, SADEV2_4Config
from ..algorithms.sade_v2_5 import SADEV2_5, SADEV2_5Config
from ..algorithms.sade_v2_6 import SADEV2_6, SADEV2_6Config
from ..algorithms.sade_v2_7 import SADEV2_7, SADEV2_7Config
from ..algorithms.sade_v2_8 import SADEV2_8, SADEV2_8Config
from ..algorithms.sade_v2_9 import SADEV2_9, SADEV2_9Config
from ..algorithms.sade_v2_10 import SADEV2_10, SADEV2_10Config
from ..algorithms.sade_v2_11 import SADEV2_11, SADEV2_11Config
from ..algorithms.sade_v2_12 import SADEV2_12, SADEV2_12Config
from ..algorithms.sade_v2_13 import SADEV2_13, SADEV2_13Config
from ..benchmarks import BENCHMARKS
from .persistence import scientific


ALGORITHMS = (
    "sade", "sade_dynamic", "sade_targeted", "sade_v2", "sade_v2_2", "sade_v2_3", "sade_v2_4",
    "sade_v2_5", "sade_v2_6", "sade_v2_7", "sade_v2_8", "sade_v2_9",
    "sade_v2_10", "sade_v2_11", "sade_v2_12", "sade_v2_13", "dsi",
    "dsi_dynamic"
)


@dataclass(frozen=True)
class ExperimentCase:
    suite: str
    problem: str
    dimension: int | None


def _selection(values: Iterable[str] | None, allowed: Iterable[str], label: str) -> tuple[str, ...]:
    allowed_values = tuple(allowed)
    requested = tuple(str(item).lower() for item in (values or ("all",)))
    if "all" in requested:
        return allowed_values
    invalid = sorted(set(requested) - set(allowed_values))
    if invalid:
        raise ValueError(f"Unknown {label}: {', '.join(invalid)}")
    return tuple(value for value in allowed_values if value in requested)


def build_cases(
    suites: Iterable[str] | None,
    problems: Iterable[str] | None,
    dimensions: Iterable[str | int] | None,
) -> list[ExperimentCase]:
    """Expand CLI selectors into deterministic problem/dimension cases."""

    suite_tokens = tuple(str(item).lower() for item in (suites or ("all",)))
    selected_suites = (
        tuple(
            name for name, inventory in BENCHMARKS.items()
            if inventory.get("included_in_all", True)
        )
        if "all" in suite_tokens
        else _selection(suite_tokens, BENCHMARKS, "suite")
    )
    problem_tokens = tuple(str(item).lower() for item in (problems or ("all",)))
    all_problems = "all" in problem_tokens
    dimension_tokens = tuple(str(item).lower() for item in (dimensions or (10, 30)))
    all_dimensions = "all" in dimension_tokens
    if not all_dimensions:
        try:
            requested_dimensions = {int(value) for value in dimension_tokens}
        except ValueError as error:
            raise ValueError("Dimensions must be integers or 'all'.") from error
    else:
        requested_dimensions = set()

    cases: list[ExperimentCase] = []
    matched_problems: set[str] = set()
    for suite in selected_suites:
        inventory = BENCHMARKS[suite]
        selected_problems = tuple(
            name
            for name in inventory["problems"]
            if all_problems or name in problem_tokens
        )
        matched_problems.update(selected_problems)
        if inventory["dimensions"] == "fixed":
            cases.extend(ExperimentCase(suite, name, None) for name in selected_problems)
            continue
        allowed_dimensions = tuple(int(value) for value in inventory["dimensions"])
        selected_dimensions = (
            allowed_dimensions
            if all_dimensions
            else tuple(value for value in allowed_dimensions if value in requested_dimensions)
        )
        dimensions_by_problem = inventory.get("dimensions_by_problem", {})
        cases.extend(
            ExperimentCase(suite, name, dimension)
            for name in selected_problems
            for dimension in selected_dimensions
            if dimension in dimensions_by_problem.get(name, allowed_dimensions)
        )

    if not all_problems:
        unmatched = sorted(set(problem_tokens) - matched_problems)
        if unmatched:
            raise ValueError(
                "Requested problems are not present in the selected suites: "
                + ", ".join(unmatched)
            )
    if not cases:
        raise ValueError("The selectors did not produce any benchmark cases.")
    return cases


def make_algorithm_config(
    algorithm: str,
    *,
    common_parameters: Mapping[str, Any],
    algorithm_parameters: Mapping[str, Any],
    seed: int,
    save_candidate_pools: bool,
) -> (
    SADEConfig
    | SADETargetedConfig
    | SADEV2Config
    | SADEV2_2Config
    | SADEV2_3Config
    | SADEV2_4Config
    | SADEV2_5Config
    | SADEV2_6Config
    | SADEV2_7Config
    | SADEV2_8Config
    | SADEV2_9Config
    | SADEV2_10Config
    | SADEV2_11Config
    | SADEV2_12Config
    | SADEV2_13Config
    | DSIConfig
    | DSIDynamicConfig
):
    """Merge common and algorithm-specific values into a validated config."""

    name = algorithm.lower()
    parameters = {
        **dict(common_parameters),
        **dict(algorithm_parameters),
        "seed": seed,
        "save_candidate_pools": save_candidate_pools,
    }
    try:
        if name in ("sade", "sade_dynamic"):
            return SADEConfig(**parameters)
        if name == "sade_targeted":
            return SADETargetedConfig(**parameters)
        if name == "sade_v2":
            return SADEV2Config(**parameters)
        if name == "sade_v2_2":
            return SADEV2_2Config(**parameters)
        if name == "sade_v2_3":
            return SADEV2_3Config(**parameters)
        if name == "sade_v2_4":
            return SADEV2_4Config(**parameters)
        if name == "sade_v2_5":
            return SADEV2_5Config(**parameters)
        if name == "sade_v2_6":
            return SADEV2_6Config(**parameters)
        if name == "sade_v2_7":
            return SADEV2_7Config(**parameters)
        if name == "sade_v2_8":
            return SADEV2_8Config(**parameters)
        if name == "sade_v2_9":
            return SADEV2_9Config(**parameters)
        if name == "sade_v2_10":
            return SADEV2_10Config(**parameters)
        if name == "sade_v2_11":
            return SADEV2_11Config(**parameters)
        if name == "sade_v2_12":
            return SADEV2_12Config(**parameters)
        if name == "sade_v2_13":
            return SADEV2_13Config(**parameters)
        if name == "dsi":
            return DSIConfig(**parameters)
        if name == "dsi_dynamic":
            return DSIDynamicConfig(**parameters)
    except TypeError as error:
        raise ValueError(f"Invalid {name} configuration: {error}") from error
    raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {ALGORITHMS}.")


def make_optimizer(
    algorithm: str,
    problem: Any,
    *,
    common_parameters: Mapping[str, Any],
    algorithm_parameters: Mapping[str, Any],
    seed: int,
    save_candidate_pools: bool,
) -> tuple[Any, Any]:
    """Construct a registered optimizer and expose its immutable config."""

    config = make_algorithm_config(
        algorithm,
        common_parameters=common_parameters,
        algorithm_parameters=algorithm_parameters,
        seed=seed,
        save_candidate_pools=save_candidate_pools,
    )
    if algorithm.lower() in ("sade", "sade_dynamic"):
        return SADE(problem, config), config
    if algorithm.lower() == "sade_targeted":
        return SADETargeted(problem, config), config
    if algorithm.lower() == "sade_v2":
        return SADEV2(problem, config), config
    if algorithm.lower() == "sade_v2_2":
        return SADEV2_2(problem, config), config
    if algorithm.lower() == "sade_v2_3":
        return SADEV2_3(problem, config), config
    if algorithm.lower() == "sade_v2_4":
        return SADEV2_4(problem, config), config
    if algorithm.lower() == "sade_v2_5":
        return SADEV2_5(problem, config), config
    if algorithm.lower() == "sade_v2_6":
        return SADEV2_6(problem, config), config
    if algorithm.lower() == "sade_v2_7":
        return SADEV2_7(problem, config), config
    if algorithm.lower() == "sade_v2_8":
        return SADEV2_8(problem, config), config
    if algorithm.lower() == "sade_v2_9":
        return SADEV2_9(problem, config), config
    if algorithm.lower() == "sade_v2_10":
        return SADEV2_10(problem, config), config
    if algorithm.lower() == "sade_v2_11":
        return SADEV2_11(problem, config), config
    if algorithm.lower() == "sade_v2_12":
        return SADEV2_12(problem, config), config
    if algorithm.lower() == "sade_v2_13":
        return SADEV2_13(problem, config), config
    if algorithm.lower() == "dsi":
        return DSI(problem, config), config
    return DSIDynamic(problem, config), config


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate completed runs by algorithm and benchmark case."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row["algorithm"],
            row["suite"],
            row["problem"],
            row["dimension"],
        )
        groups.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for key, group in groups.items():
        feasible_group = [row for row in group if row["feasible"]]
        use_error = group[0].get("objective_error_numeric") is not None
        metric_key = "objective_error_numeric" if use_error else "objective_numeric"
        values = np.asarray(
            [row[metric_key] for row in feasible_group if row.get(metric_key) is not None],
            dtype=float,
        )
        statistics = {
            "mean": scientific(float(np.mean(values))) if len(values) else None,
            "std": scientific(float(np.std(values, ddof=1))) if len(values) > 1 else None,
            "median": scientific(float(np.median(values))) if len(values) else None,
            "best": scientific(float(np.min(values))) if len(values) else None,
            "worst": scientific(float(np.max(values))) if len(values) else None,
        }
        aggregates.append(
            {
                "algorithm": key[0],
                "suite": key[1],
                "problem": key[2],
                "dimension": key[3],
                "runs": len(group),
                "feasible_runs": len(feasible_group),
                "feasible_rate": f"{len(feasible_group) / len(group):.6f}",
                "metric": "objective_error" if use_error else "objective",
                **statistics,
            }
        )
    return aggregates
