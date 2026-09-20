"""Paper-style statistics for repeated constrained-optimization experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any
import warnings

import numpy as np
from scipy.stats import norm, rankdata, wilcoxon


DEFAULT_CONTROL = "sade"
DEFAULT_ALPHA = 0.05
DEFAULT_INFEASIBLE_VALUE = 10e20


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _as_float(row: dict[str, Any], numeric_key: str, public_key: str) -> float:
    value = row.get(numeric_key)
    if value is None or value == "":
        value = row.get(public_key)
    return float(value)


def _metric(row: dict[str, Any]) -> tuple[str, float]:
    if str(row["suite"]).lower() == "cec2006":
        return "objective_error", _as_float(
            row, "objective_error_numeric", "objective_error"
        )
    return "objective", _as_float(row, "objective_numeric", "objective")


def _instance_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["suite"]), str(row["problem"]), int(row["dimension"])


def _run_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["run"]), int(row["seed"])


def _display_value(values: np.ndarray, feasible: np.ndarray) -> str:
    if not np.all(feasible):
        return f"{100.0 * float(np.mean(feasible)):.2f}%"
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.2e} ({std:.2e})"


def _performance_key(values: np.ndarray, feasible: np.ndarray) -> tuple[int, float]:
    """Match the paper table: success count first, mean only at 100% success."""

    failures = int(np.count_nonzero(~feasible))
    return failures, float(np.mean(values)) if failures == 0 else 0.0


def _instance_ranks(
    performance: dict[str, tuple[int, float]], algorithms: Sequence[str]
) -> dict[str, float]:
    ordered_keys = sorted(set(performance.values()))
    ranks: dict[tuple[int, float], float] = {}
    position = 1
    for key in ordered_keys:
        tied = sum(value == key for value in performance.values())
        ranks[key] = (position + position + tied - 1) / 2.0
        position += tied
    return {algorithm: ranks[performance[algorithm]] for algorithm in algorithms}


def _signed_rank(
    control: np.ndarray,
    comparator: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float, str]:
    differences = control - comparator
    nonzero = differences != 0.0
    if not np.any(nonzero):
        return 0.0, 1.0, "≈"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = wilcoxon(
            control,
            comparator,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
    statistic, p_value = float(result.statistic), float(result.pvalue)
    if p_value >= alpha:
        return statistic, p_value, "≈"

    signed_rank_sum = float(
        np.sum(rankdata(np.abs(differences[nonzero])) * np.sign(differences[nonzero]))
    )
    symbol = "+" if signed_rank_sum < 0.0 else "−"
    return statistic, p_value, symbol


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values."""

    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    previous = 0.0
    for index, algorithm in enumerate(ordered):
        candidate = min(1.0, (count - index) * p_values[algorithm])
        previous = max(previous, candidate)
        adjusted[algorithm] = previous
    return adjusted


def _format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.2e}"


def _summary_rows(
    scope: str,
    rank_records: Sequence[dict[str, float]],
    symbol_records: dict[str, list[str]],
    algorithms: Sequence[str],
    control_algorithm: str,
    alpha: float,
) -> list[dict[str, Any]]:
    instance_count = len(rank_records)
    average_rank = {
        algorithm: float(np.mean([record[algorithm] for record in rank_records]))
        for algorithm in algorithms
    }
    standard_error = np.sqrt(
        len(algorithms) * (len(algorithms) + 1) / (6.0 * instance_count)
    )
    posthoc_raw = {
        algorithm: float(
            2.0
            * norm.sf(
                abs(average_rank[algorithm] - average_rank[control_algorithm])
                / standard_error
            )
        )
        for algorithm in algorithms
        if algorithm != control_algorithm
    }
    posthoc_adjusted = _holm_adjust(posthoc_raw)

    rank_row: dict[str, Any] = {
        "suite": scope,
        "problem": "Average Rank",
        "dimension": "",
        "metric": "",
    }
    adjusted_row: dict[str, Any] = {
        "suite": scope,
        "problem": "Adjusted p-value",
        "dimension": "",
        "metric": "Friedman rank post-hoc + Holm",
    }
    count_row: dict[str, Any] = {
        "suite": scope,
        "problem": "+/≈/−",
        "dimension": "",
        "metric": f"paired WSR, alpha={alpha:g}",
    }
    for algorithm in algorithms:
        rank_row[algorithm] = f"{average_rank[algorithm]:.4f}"
        if algorithm == control_algorithm:
            adjusted_row[algorithm] = "NaN"
            count_row[algorithm] = ""
        else:
            adjusted_row[algorithm] = _format_p_value(posthoc_adjusted[algorithm])
            records = symbol_records[algorithm]
            count_row[algorithm] = (
                f"{records.count('+')}/{records.count('≈')}/{records.count('−')}"
            )
    return [rank_row, adjusted_row, count_row]


def paper_statistics_rows(
    rows: Iterable[dict[str, Any]],
    algorithms: Sequence[str],
    *,
    control_algorithm: str = DEFAULT_CONTROL,
    alpha: float = DEFAULT_ALPHA,
    infeasible_value: float = DEFAULT_INFEASIBLE_VALUE,
) -> list[dict[str, Any]]:
    """Build one compact table using the paper's success-rate and WSR rules.

    Symbols are written only beside comparator algorithms. ``+`` means the
    control is significantly better, ``−`` means the control is significantly
    worse, and ``≈`` means no significant paired WSR difference.
    """

    selected = tuple(dict.fromkeys(str(name).lower() for name in algorithms))
    control_algorithm = str(control_algorithm).lower()
    if not selected:
        raise ValueError("At least one algorithm is required for statistics.")
    if control_algorithm not in selected:
        raise ValueError(
            f"Control algorithm {control_algorithm!r} is not in the experiment."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("Statistics alpha must be in (0, 1).")
    if not np.isfinite(infeasible_value) or infeasible_value <= 0.0:
        raise ValueError("The infeasible replacement value must be positive and finite.")

    display_algorithms = tuple(
        name for name in selected if name != control_algorithm
    ) + (control_algorithm,)
    completed = [row for row in rows if row.get("status", "ok") == "ok"]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in completed:
        grouped.setdefault(_instance_key(row), []).append(row)
    if not grouped:
        raise ValueError("No completed runs are available for statistics.")

    instance_rows: list[dict[str, Any]] = []
    rank_records: list[tuple[str, dict[str, float]]] = []
    symbol_records: dict[str, list[tuple[str, str]]] = {
        algorithm: []
        for algorithm in display_algorithms
        if algorithm != control_algorithm
    }

    for instance, group in grouped.items():
        by_algorithm: dict[str, list[dict[str, Any]]] = {}
        for row in group:
            by_algorithm.setdefault(str(row["algorithm"]).lower(), []).append(row)
        missing = set(display_algorithms) - set(by_algorithm)
        if missing:
            raise ValueError(
                f"Incomplete algorithms for {instance}: {', '.join(sorted(missing))}."
            )

        run_keys = {
            algorithm: {_run_key(row) for row in by_algorithm[algorithm]}
            for algorithm in display_algorithms
        }
        expected_keys = run_keys[control_algorithm]
        if any(keys != expected_keys for keys in run_keys.values()):
            raise ValueError(f"Runs are not paired by run and seed for {instance}.")

        values_by_algorithm: dict[str, np.ndarray] = {}
        feasible_by_algorithm: dict[str, np.ndarray] = {}
        test_by_algorithm: dict[str, np.ndarray] = {}
        performance: dict[str, tuple[int, float]] = {}
        metric_name: str | None = None
        for algorithm in display_algorithms:
            ordered = sorted(by_algorithm[algorithm], key=_run_key)
            metrics = [_metric(row) for row in ordered]
            current_names = {name for name, _ in metrics}
            if len(current_names) != 1:
                raise ValueError(f"Mixed metrics for {instance}/{algorithm}.")
            current_name = current_names.pop()
            if metric_name is None:
                metric_name = current_name
            elif metric_name != current_name:
                raise ValueError(f"Algorithms use different metrics for {instance}.")
            values = np.asarray([value for _, value in metrics], dtype=float)
            feasible = np.asarray([_as_bool(row["feasible"]) for row in ordered])
            values_by_algorithm[algorithm] = values
            feasible_by_algorithm[algorithm] = feasible
            test_by_algorithm[algorithm] = np.where(
                feasible, values, infeasible_value
            )
            performance[algorithm] = _performance_key(values, feasible)

        current_ranks = _instance_ranks(performance, display_algorithms)
        rank_records.append((instance[0], current_ranks))

        output: dict[str, Any] = {
            "suite": instance[0],
            "problem": instance[1],
            "dimension": instance[2],
            "metric": metric_name,
        }
        for algorithm in display_algorithms:
            display = _display_value(
                values_by_algorithm[algorithm], feasible_by_algorithm[algorithm]
            )
            if algorithm != control_algorithm:
                _, _, symbol = _signed_rank(
                    test_by_algorithm[control_algorithm],
                    test_by_algorithm[algorithm],
                    alpha=alpha,
                )
                symbol_records[algorithm].append((instance[0], symbol))
                display = f"{display} {symbol}"
            output[algorithm] = display
        instance_rows.append(output)

    table: list[dict[str, Any]] = []
    suites = tuple(dict.fromkeys(row["suite"] for row in instance_rows))
    for suite in suites:
        table.extend(row for row in instance_rows if row["suite"] == suite)
        suite_ranks = [rank for name, rank in rank_records if name == suite]
        suite_symbols = {
            algorithm: [
                symbol for name, symbol in records if name == suite
            ]
            for algorithm, records in symbol_records.items()
        }
        table.extend(
            _summary_rows(
                suite,
                suite_ranks,
                suite_symbols,
                display_algorithms,
                control_algorithm,
                alpha,
            )
        )
    if len(suites) > 1:
        all_ranks = [rank for _, rank in rank_records]
        all_symbols = {
            algorithm: [symbol for _, symbol in records]
            for algorithm, records in symbol_records.items()
        }
        table.extend(
            _summary_rows(
                "ALL",
                all_ranks,
                all_symbols,
                display_algorithms,
                control_algorithm,
                alpha,
            )
        )
    return table
