"""Consolidate the completed SADE development experiments.

The script deliberately uses explicit source files.  Several result folders are
smoke tests or superseded reruns, so globbing every summary would double count
the same algorithm/seed.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent

HISTORICAL_BASELINE = RESULTS / "baselines" / "cec_10d30d_25runs_20260913_231757_699643" / "summary_runs.csv"
CURRENT_SADE = RESULTS / "baselines" / "original_sade_current_5runs_20260916" / "summary_runs.csv"

SOURCES = {
    "sade_v2_3": [
        RESULTS / "development" / "sade_v2_3_hard_p90_cec2017_10d30d_3runs_20260914" / "summary_runs.csv"
    ],
    "sade_v2_4": [
        RESULTS / "development" / "sade_v2_4_hybrid_cec2017_10d30d_3runs_20260914" / "summary_runs.csv"
    ],
    "sade_v2_5": [
        RESULTS / "development" / "sade_v2_5_min_mean_cec2017_10d30d_3runs_20260915" / "summary_runs.csv"
    ],
    "sade_v2_6": [
        RESULTS / "development" / "sade_v2_6_two_point_batch_20260915_175732_630650" / "summary_runs.csv"
    ],
    # The earlier 18:20 V2.7 run predates the corrected implementation and is excluded.
    "sade_v2_7": [
        RESULTS / "development" / "sade_v2_7_dynamic_two_point_batch_20260915_183230_836582" / "summary_runs.csv",
        RESULTS / "development" / "sade_v2_7_dynamic_two_point_batch_20260915_222418_660404" / "summary_runs.csv",
    ],
    "sade_v2_8": [
        RESULTS / "development" / "sade_v2_8_knn_rerank_20260915_214741_570297" / "summary_runs.csv",
        RESULTS / "development" / "sade_v2_8_knn_rerank_20260915_221331_219410" / "summary_runs.csv",
    ],
    "sade_v2_9": [
        RESULTS / "development" / "sade_v2_9_representative_5runs_300fe_20260915" / "summary_runs.csv"
    ],
    "sade_v2_13": [
        RESULTS / "development" / "sade_v2_13_probability_fallback_5runs_20260916" / "summary_runs.csv"
    ],
}

CASES = [(p, d) for p in ("c01", "c05", "c20", "c22") for d in (10, 30)]


def load(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        result.append(
            {
                "algorithm": row["algorithm"],
                "suite": row["suite"],
                "problem": row["problem"],
                "dimension": int(row["dimension"]),
                "seed": int(row["seed"]),
                "objective": float(row["objective"]),
                "total_violation": float(row["total_violation"]),
                "feasible": row["feasible"].lower() == "true",
                "evaluations": int(row["evaluations"]),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return result


def average_ranks(values: list[tuple[tuple[object, ...], str]]) -> dict[str, float]:
    ordered = sorted(values, key=lambda item: item[0])
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for _, algorithm in ordered[start:end]:
            ranks[algorithm] = rank
        start = end
    return ranks


def comparison_key(row: dict[str, object]) -> tuple[object, ...]:
    if bool(row["feasible"]):
        return (0, float(row["objective"]))
    return (1, float(row["total_violation"]), float(row["objective"]))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


baseline = [
    row
    for row in load(HISTORICAL_BASELINE)
    if row["algorithm"] == "dsi"
    and row["suite"] == "cec2017"
    and (row["problem"], row["dimension"]) in CASES
]
baseline.extend(
    row
    for row in load(CURRENT_SADE)
    if row["algorithm"] == "sade"
    and row["suite"] == "cec2017"
    and (row["problem"], row["dimension"]) in CASES
)
all_rows = list(baseline)
for algorithm, paths in SOURCES.items():
    for path in paths:
        all_rows.extend(
            row
            for row in load(path)
            if row["algorithm"] == algorithm
            and row["suite"] == "cec2017"
            and (row["problem"], row["dimension"]) in CASES
        )

# Reject accidental duplicate algorithm/case/seed records in the canonical data.
seen: set[tuple[object, ...]] = set()
for row in all_rows:
    key = (row["algorithm"], row["problem"], row["dimension"], row["seed"])
    if key in seen:
        raise RuntimeError(f"duplicate canonical record: {key}")
    seen.add(key)

write_csv(
    OUT / "canonical_runs.csv",
    all_rows,
    [
        "algorithm",
        "suite",
        "problem",
        "dimension",
        "seed",
        "objective",
        "total_violation",
        "feasible",
        "evaluations",
        "source",
    ],
)


def build_comparison(
    algorithms: list[str], seeds: list[int], name: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected = [
        row
        for row in all_rows
        if row["algorithm"] in algorithms and row["seed"] in seeds
    ]
    index = {
        (row["algorithm"], row["problem"], row["dimension"], row["seed"]): row
        for row in selected
    }
    complete_cases = []
    for problem, dimension in CASES:
        if all(
            (algorithm, problem, dimension, seed) in index
            for algorithm in algorithms
            for seed in seeds
        ):
            complete_cases.append((problem, dimension))

    instance_rows: list[dict[str, object]] = []
    run_ranks: dict[str, list[float]] = defaultdict(list)
    wins_dsi = defaultdict(int)
    losses_dsi = defaultdict(int)
    ties_dsi = defaultdict(int)
    wins_sade = defaultdict(int)
    losses_sade = defaultdict(int)
    ties_sade = defaultdict(int)

    for problem, dimension in complete_cases:
        for seed in seeds:
            ranked = average_ranks(
                [
                    (comparison_key(index[(algorithm, problem, dimension, seed)]), algorithm)
                    for algorithm in algorithms
                ]
            )
            for algorithm, rank in ranked.items():
                run_ranks[algorithm].append(rank)
            for algorithm in algorithms:
                key = comparison_key(index[(algorithm, problem, dimension, seed)])
                for control, wins, losses, ties in (
                    ("dsi", wins_dsi, losses_dsi, ties_dsi),
                    ("sade", wins_sade, losses_sade, ties_sade),
                ):
                    control_key = comparison_key(index[(control, problem, dimension, seed)])
                    if key < control_key:
                        wins[algorithm] += 1
                    elif key > control_key:
                        losses[algorithm] += 1
                    else:
                        ties[algorithm] += 1

        for algorithm in algorithms:
            rows = [index[(algorithm, problem, dimension, seed)] for seed in seeds]
            feasible = [row for row in rows if row["feasible"]]
            objectives = [float(row["objective"]) for row in feasible]
            violations = [float(row["total_violation"]) for row in rows]
            instance_rows.append(
                {
                    "algorithm": algorithm,
                    "problem": problem,
                    "dimension": dimension,
                    "runs": len(rows),
                    "success_rate": len(feasible) / len(rows),
                    "feasible_objective_mean": mean(objectives) if objectives else "",
                    "feasible_objective_std": pstdev(objectives) if objectives else "",
                    "feasible_objective_median": median(objectives) if objectives else "",
                    "violation_median": median(violations),
                }
            )

    algorithm_rows = []
    total_units = len(complete_cases) * len(seeds)
    for algorithm in algorithms:
        success_count = sum(
            1
            for problem, dimension in complete_cases
            for seed in seeds
            if index[(algorithm, problem, dimension, seed)]["feasible"]
        )
        algorithm_rows.append(
            {
                "algorithm": algorithm,
                "instances": len(complete_cases),
                "seeds": len(seeds),
                "run_case_units": total_units,
                "successful_units": success_count,
                "success_rate": success_count / total_units if total_units else math.nan,
                "average_run_rank": mean(run_ranks[algorithm]),
                "wins_vs_dsi": wins_dsi[algorithm],
                "losses_vs_dsi": losses_dsi[algorithm],
                "ties_vs_dsi": ties_dsi[algorithm],
                "wins_vs_sade": wins_sade[algorithm],
                "losses_vs_sade": losses_sade[algorithm],
                "ties_vs_sade": ties_sade[algorithm],
            }
        )
    algorithm_rows.sort(key=lambda row: float(row["average_run_rank"]))

    write_csv(
        OUT / f"{name}_algorithm_summary.csv",
        algorithm_rows,
        list(algorithm_rows[0]),
    )
    write_csv(
        OUT / f"{name}_instance_summary.csv",
        instance_rows,
        list(instance_rows[0]),
    )
    return algorithm_rows, instance_rows


common3_algorithms, common3_instances = build_comparison(
    [
        "sade",
        "dsi",
        "sade_v2_3",
        "sade_v2_4",
        "sade_v2_5",
        "sade_v2_6",
        "sade_v2_7",
        "sade_v2_8",
        "sade_v2_9",
    ],
    [1, 2, 3],
    "common_3seed",
)

key5_algorithms, key5_instances = build_comparison(
    ["sade", "dsi", "sade_v2_7", "sade_v2_8", "sade_v2_9"],
    [1, 2, 3, 4, 5],
    "key_5seed",
)


def display_result(row: dict[str, object]) -> str:
    success_rate = float(row["success_rate"])
    if success_rate < 1.0:
        return f"{success_rate:.0%}"
    return (
        f"{float(row['feasible_objective_mean']):.2e} "
        f"({float(row['feasible_objective_std']):.2e})"
    )


with (OUT / "version_changes.csv").open(encoding="utf-8-sig", newline="") as handle:
    versions = list(csv.DictReader(handle))
with (
    RESULTS
    / "baselines"
    / "cec_10d30d_25runs_20260913_231757_699643"
    / "summary_statistics.csv"
).open(encoding="utf-8-sig", newline="") as handle:
    historical_statistics = list(csv.DictReader(handle))

key_algorithms = ["sade", "dsi", "sade_v2_7", "sade_v2_8", "sade_v2_9"]
selected_runs = [
    row
    for row in all_rows
    if row["algorithm"] in key_algorithms
    and row["seed"] in {1, 2, 3, 4, 5}
    and (row["problem"], row["dimension"]) in CASES
]

workbook_data = {
    "key_5seed_algorithm_summary": key5_algorithms,
    "key_5seed_instance_summary": key5_instances,
    "common_3seed_algorithm_summary": common3_algorithms,
    "common_3seed_instance_summary": common3_instances,
    "version_changes": versions,
    "selected_runs": selected_runs,
    "historical_25run_statistics": historical_statistics,
    "notes": [
        "当前聚焦对比：CEC2017 C01/C05/C20/C22，10D/30D，seed 1--5，300 FEs，种群30。",
        "原SADE已用当前源码重新运行；DSI取自已有25-run实验中的相同seed。",
        "平均rank按40个配对运行-实例单元计算，采用可行优先规则，数值越小越好。",
        "5 runs的双侧Wilcoxon检验无法在0.05水平显著；配对胜负只用于开发筛选。",
        "历史25-run表使用后续源码修改前的原SADE，只作为宽范围历史证据。",
    ],
}
(OUT / "workbook_data.json").write_text(
    json.dumps(workbook_data, ensure_ascii=False, indent=2), encoding="utf-8"
)

lookup = {
    (row["algorithm"], row["problem"], int(row["dimension"])): row
    for row in key5_instances
}
lines = [
    "# SADE版本实验整理",
    "",
    "## 当前结论",
    "",
    "- V2.7是当前最合适的主基线：结构清楚，综合稳定，没有V2.8的C01退化，也没有V2.9的明显专项偏向。",
    "- V2.8保留为目标排序专项版本：C05-10D和C20较强，但C01明显退化。",
    "- V2.9保留为可行性阶段专项版本：C05-30D明显改善，但C05-10D和C22-10D退化。",
    "- V2.5是关键转折版本，但只有3个共同seed，证据等级低于V2.7/V2.8/V2.9。",
    "- V2.10至V2.13均未稳定超过V2.7，作为机制诊断和失败消融保留。",
    "",
    "## 当前代码5-run对比",
    "",
    "CEC2017 C01/C05/C20/C22，10D/30D，seed 1--5，300 FEs。全部可行时显示目标均值（总体标准差）；存在不可行运行时显示成功率。",
    "",
    "| 实例 | 原SADE | DSI | V2.7 | V2.8 | V2.9 |",
    "|---|---:|---:|---:|---:|---:|",
]
for problem, dimension in CASES:
    values = [
        display_result(lookup[(algorithm, problem, dimension)])
        for algorithm in key_algorithms
    ]
    lines.append(
        f"| {problem.upper()}-{dimension}D | " + " | ".join(values) + " |"
    )
lines.extend(
    [
        "",
        "### 总体配对概览",
        "",
        "| 算法 | 成功单元/40 | 平均运行排名 | 对DSI 胜/负 | 对原SADE 胜/负 |",
        "|---|---:|---:|---:|---:|",
    ]
)
for row in key5_algorithms:
    lines.append(
        f"| {row['algorithm']} | {row['successful_units']}/40 | "
        f"{float(row['average_run_rank']):.3f} | "
        f"{row['wins_vs_dsi']}/{row['losses_vs_dsi']} | "
        f"{row['wins_vs_sade']}/{row['losses_vs_sade']} |"
    )
lines.extend(
    [
        "",
        (
            "V2.8和V2.7的平均排名几乎相同，但V2.8依靠C05-10D/C20获得优势，"
            "同时在C01明显变差。因此不把0.0125的排名差解释为整体胜出。"
            "V2.9对DSI为22胜18负，是三者中配对胜负最好，但主要由C05-30D拉动。"
        ),
        "",
        (
            "原SADE在C01/C05明显落后，但C20仍有竞争力，C22-10D成功率为4/5"
            "且可行目标优于三个V2版本。说明V2主线改善了目标搜索，却没有全面"
            "继承原SADE在C22薄可行域上的优势。"
        ),
        "",
        "## 历史25-run结果",
        "",
        (
            "历史完整实验覆盖CEC2006/2010/2017共43个实例。DSI总体平均rank为"
            "1.2558，原SADE为1.7442；以SADE为控制算法，DSI列的+/≈/−为"
            "3/10/30，Holm调整p值为0.0014。该实验使用的是后续源码修改前的"
            "原SADE，只作为宽范围历史证据，不与当前5-run数值直接合并。"
        ),
        "",
        "## 统计口径",
        "",
        (
            "当前5-run表用于筛选方向。五个配对样本的双侧Wilcoxon检验即使同向"
            "也无法达到0.05，因此这里只报告配对胜负和运行级平均rank，不写显著"
            "优劣。正式论文表应对最终保留算法补足25或30个配对runs后再做WSR、"
            "Holm校正和平均rank。"
        ),
    ]
)
(OUT / "SADE_VERSION_EXPERIMENT_REVIEW.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

print(f"Wrote consolidated analysis to {OUT}")
