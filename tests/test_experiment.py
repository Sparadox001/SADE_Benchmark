from __future__ import annotations

import json

import numpy as np
import pytest

from sade_benchmark import CallableProblem, SADE, SADEConfig
from sade_benchmark.experiments import (
    aggregate_rows,
    build_cases,
    load_configuration,
    make_algorithm_config,
    paper_statistics_rows,
    save_run,
)


def test_all_problem_expansion_covers_expected_case_counts() -> None:
    standard = build_cases(["all"], ["all"], [10, 30])
    extended = build_cases(["all"], ["all"], ["all"])
    assert len(standard) == 43
    assert len(extended) == 61


def test_configuration_defaults_and_json_override(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "experiment": {
                    "suites": ["cec2017"],
                    "runs": 2,
                },
                "algorithms": {
                    "sade": {"surrogate_min_samples": 90},
                },
            }
        ),
        encoding="utf-8",
    )
    configuration = load_configuration(path)
    assert configuration["experiment"]["runs"] == 2
    assert configuration["experiment"]["dimensions"] == [10, 30]
    assert configuration["algorithms"]["sade"]["surrogate_min_samples"] == 90
    assert configuration["algorithms"]["sade"]["dynamic_constraint_tightening"] is False
    assert configuration["algorithms"]["sade_dynamic"]["dynamic_constraint_tightening"] is True
    assert configuration["algorithms"]["dsi"]["wmax"] == 10


def test_sade_dynamic_registry_alias_enables_tightening() -> None:
    configuration = load_configuration()
    config = make_algorithm_config(
        "sade_dynamic",
        common_parameters=configuration["common"],
        algorithm_parameters=configuration["algorithms"]["sade_dynamic"],
        seed=1,
        save_candidate_pools=False,
    )
    assert isinstance(config, SADEConfig)
    assert config.dynamic_constraint_tightening


def test_sade_targeted_registry_has_last_constraint_as_default() -> None:
    configuration = load_configuration()
    config = make_algorithm_config(
        "sade_targeted",
        common_parameters=configuration["common"],
        algorithm_parameters=configuration["algorithms"]["sade_targeted"],
        seed=1,
        save_candidate_pools=False,
    )
    assert config.dynamic_constraint_tightening
    assert config.tightening_constraint_index == -1


def test_configuration_rejects_misspelled_parameters(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"algorithms": {"sade": {"surrogte_min_samples": 80}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="surrogte_min_samples"):
        load_configuration(path)


def test_selective_case_expansion() -> None:
    cases = build_cases(["cec2017"], ["c01", "c22"], [10, 50])
    assert [(case.problem, case.dimension) for case in cases] == [
        ("c01", 10),
        ("c01", 50),
        ("c22", 10),
        ("c22", 50),
    ]


def test_objective_cap_case_expansion_is_opt_in_and_has_12_cases() -> None:
    cases = build_cases(["cec2017_objcap"], ["all"], [10, 30])
    assert len(cases) == 12
    assert ("c13", 30) not in {(case.problem, case.dimension) for case in cases}
    assert ("c22", 30) not in {(case.problem, case.dimension) for case in cases}
    assert all(case.suite == "cec2017_objcap" for case in cases)
    assert len(build_cases(["all"], ["all"], [10, 30])) == 43
    mean_cases = build_cases(["cec2017_objcap_first_mean"], ["all"], [10, 30])
    assert len(mean_cases) == 12
    assert ("c13", 30) not in {
        (case.problem, case.dimension) for case in mean_cases
    }


def test_each_run_persists_trace_files(tmp_path) -> None:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.sum(x**2, axis=1), (np.sum(x, axis=1) - 1.0)[:, None]

    problem = CallableProblem(
        name="persist_test",
        dimension=2,
        n_constraints=1,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )
    problem.suite = "test"
    config = SADEConfig(
        max_evaluations=8,
        population_size=6,
        batch_size=2,
        save_candidate_pools=True,
    )
    result = SADE(problem, config).optimize()
    row = save_run(
        tmp_path,
        algorithm="sade",
        problem=problem,
        run=1,
        seed=1,
        config=config,
        result=result,
    )

    assert row["objective"].endswith("e-02") or "e" in row["objective"]
    for filename in (
        "config.json",
        "final_result.json",
        "history.csv",
        "evaluations.csv.gz",
        "populations.npz",
        "candidate_pools.npz",
    ):
        assert (tmp_path / filename).is_file()
    final = json.loads((tmp_path / "final_result.json").read_text(encoding="utf-8"))
    assert "e" in final["objective"]


def test_dynamic_constraint_run_persists_tightening_history(tmp_path) -> None:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.sum(x**2, axis=1), -np.ones((len(x), 1))

    problem = CallableProblem(
        name="dynamic_persist_test",
        dimension=2,
        n_constraints=1,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        evaluator=evaluate,
    )
    problem.suite = "test"
    config = SADEConfig(
        max_evaluations=18,
        population_size=6,
        batch_size=6,
        dynamic_constraint_tightening=True,
        tightening_patience=1,
        tightening_improvement_tol=1e9,
    )
    result = SADE(problem, config).optimize()
    save_run(
        tmp_path,
        algorithm="sade",
        problem=problem,
        run=1,
        seed=1,
        config=config,
        result=result,
    )

    history_path = tmp_path / "dynamic_constraint_history.json"
    assert history_path.is_file()
    state = json.loads(history_path.read_text(encoding="utf-8"))
    assert state["source"] == "objective"
    assert state["tighten_count"] == 1


def test_aggregate_uses_feasible_runs_only() -> None:
    rows = [
        {
            "status": "ok",
            "algorithm": "sade",
            "suite": "test",
            "problem": "p01",
            "dimension": 2,
            "feasible": True,
            "objective_numeric": 1.0,
            "objective_error_numeric": None,
        },
        {
            "status": "ok",
            "algorithm": "sade",
            "suite": "test",
            "problem": "p01",
            "dimension": 2,
            "feasible": False,
            "objective_numeric": -100.0,
            "objective_error_numeric": None,
        },
    ]
    aggregate = aggregate_rows(rows)[0]
    assert aggregate["feasible_runs"] == 1
    assert aggregate["mean"] == "1.0000000000000000e+00"


def _stat_row(
    algorithm: str,
    run: int,
    objective: float,
    *,
    feasible: bool = True,
    suite: str = "cec2017",
) -> dict[str, object]:
    return {
        "status": "ok",
        "algorithm": algorithm,
        "suite": suite,
        "problem": "c01" if suite != "cec2006" else "g01",
        "dimension": 10,
        "run": run,
        "seed": run,
        "objective_numeric": objective,
        "objective_error_numeric": objective if suite == "cec2006" else None,
        "total_violation": 0.0 if feasible else 1.0,
        "feasible": feasible,
    }


def test_paper_statistics_uses_success_rate_and_sade_control_symbols() -> None:
    rows = []
    for run in range(1, 7):
        rows.append(_stat_row("sade", run, 1.0))
        rows.append(_stat_row("dsi", run, 10.0, feasible=(run != 6)))
    table = paper_statistics_rows(rows, ["sade", "dsi"])
    assert table[0]["sade"] == "1.00e+00 (0.00e+00)"
    assert table[0]["dsi"].startswith("83.33%")
    assert table[0]["dsi"].endswith("+")
    assert table[-3]["problem"] == "Average Rank"
    assert table[-1]["dsi"] == "1/0/0"


def test_five_pairs_cannot_be_significant_for_two_sided_exact_wsr() -> None:
    rows = []
    for run in range(1, 6):
        rows.append(_stat_row("sade", run, 1.0))
        rows.append(_stat_row("dsi", run, 10.0))
    table = paper_statistics_rows(rows, ["sade", "dsi"])
    assert table[0]["dsi"].endswith("≈")
    assert table[-1]["dsi"] == "0/1/0"


def test_multiple_suites_have_suite_and_overall_summary_rows() -> None:
    rows = []
    for suite in ("cec2006", "cec2017"):
        for run in range(1, 6):
            rows.append(_stat_row("sade", run, 1.0, suite=suite))
            rows.append(_stat_row("dsi", run, 10.0, suite=suite))
    table = paper_statistics_rows(rows, ["sade", "dsi"])
    rank_scopes = [
        row["suite"] for row in table if row["problem"] == "Average Rank"
    ]
    assert rank_scopes == ["cec2006", "cec2017", "ALL"]
