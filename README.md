# SADE/DSI Benchmark（连续、不等式约束）

这是用于比较 SADE、SADE V2系列、DSI 及后续算法的通用连续优化实验框架。原电路代码与本地 MATLAB DSI 目录均不改动；新代码只处理单目标最小化与不等式约束，统一采用：

```text
min f(x)
s.t. g_j(x) <= 0
```

SADE 保留原连续版本的 10 种 DE 变异公式、`F/CR` 分布、RBF、Expected Improvement（EI）采样和 `local_fraction=0.2` 的全局—局部配额；配额经过整数化，默认 `batch_size=3` 时实际为 2 个全局点和 1 个局部点。已去掉电路初值扰动、分类器、仿真失败规则和断点代码。初始化改为整个变量域上的 Latin hypercube sampling。种群淘汰按离散版采用“归一化自适应约束罚值优先、综合 fitness 次优先”。

基线配置和算法细节见 [`docs/algorithms/SADE_BASELINE.md`](docs/algorithms/SADE_BASELINE.md)；它与电路版 SADE 的逐项差异见 [`docs/algorithms/SADE_BENCHMARK_VS_CIRCUIT.md`](docs/algorithms/SADE_BENCHMARK_VS_CIRCUIT.md)。可选的目标值动态虚拟约束默认关闭，其配置和结果已单独归档，不计入基础算法对照。

基线 SADE 和 V2.1 的 RBF 都在有效样本数达到可配置的 `surrogate_min_samples` 后启用，Python 默认值为 30；V2.1 的初始种群默认也是 30，并用整个初始种群确定一次固定约束尺度。距离筛选后的候选不足时，参照离散版先从对应的原始候选池随机补足；极端情况下再用全变量域随机样本兜底。连续局部搜索参照离散版的调度方式，按实际总迭代数将半径从 0.30 衰减到 0.10，但不做任何离散取整。

`sade_v2` 保留 SADE 的 DE 和局部搜索，将昂贵样本选择改成约束感知的两阶段策略；V2.2–V2.13 是后续单因素改进和诊断版本。逐版本说明集中在 [`docs/algorithms/versions/`](docs/algorithms/versions/)，候选池诊断见 [`docs/diagnostics/`](docs/diagnostics/)。DSI 是本地 MATLAB `DSI_ECOP.m` 的独立 Python 移植，说明见 [`docs/algorithms/DSI.md`](docs/algorithms/DSI.md)。

## 已纳入的问题

- CEC2006（固定维度）：`g01,g02,g04,g06,g07,g08,g09,g10,g12,g16,g18,g19,g24`
- CEC2010（`D=10,30`）：`c01,c07,c08,c13,c14,c15`
- CEC2017（`D=10,30,50,100`）：`c01,c02,c04,c05,c13,c19,c20,c22,c28`

CEC2017 的 `c12`、`c21` 已按当前项目决定明确排除。三个集合均不含等式约束。

## 目录

```text
sade_benchmark/
├── algorithms/
│   ├── sade/
│   │   ├── config.py           SADE 参数默认值与检查
│   │   ├── optimizer.py        SADE 主循环和记录逻辑
│   │   ├── de.py               原 SADE 的 DE 策略与 F/CR
│   │   ├── penalty.py          自适应约束罚函数
│   │   ├── sampling.py         距离过滤和候选补足
│   │   ├── surrogate.py        cubic RBF 与 EI
│   │   └── tightening.py       可选目标值动态虚拟约束
│   ├── sade_v2/                 V2.1 配置、两阶段采样及多输出 RBF
│   ├── sade_v2_2/               V2.2 统一真实种群排序与引导
│   ├── sade_v2_3/               V2.3 在线约束预测误差校准
│   ├── sade_v2_4/               V2.4 混合P90门控与连续可行性权重
│   ├── sade_v2_5/               V2.5 鲁棒可行集内最小预测目标
│   ├── sade_v2_6/               V2.6 一个全局点加一个局部点
│   ├── sade_v2_7/               V2.7 停滞触发动态边界探索
│   ├── sade_v2_8/               V2.8 全局RBF短名单与5近邻重排
│   ├── sade_v2_9/               V2.9 首次可行前使用原始预测CV
│   ├── sade_v2_10/              V2.10 双尺度局部候选
│   ├── sade_v2_11/              V2.11 可行性阶段最近邻淘汰
│   ├── sade_v2_12/              V2.12 可行性阶段混合淘汰
│   ├── sade_v2_13/              V2.13 P90不足时可行概率优先回退
│   ├── dsi/
│   └── dsi_dynamic/
│       ├── config.py           DSI 参数默认值与检查
│       ├── optimizer.py        DSI 主循环与 C2oDE
│       └── surrogate.py        DSI 多输出 cubic RBF
├── benchmarks/                 CEC 题目、数据及注册表
├── core/                       通用问题接口、结果结构和数值工具
└── experiments/                算法注册、组合展开、统计和持久化
configs/
├── reference/                 正式基准与常用运行方案
├── development/               V2 开发/消融方案
└── optional_constraint_tightening/  可选约束收紧方案
docs/                           算法、版本与诊断文档
results/                        按 baselines/development/diagnostics 分类的结果
run_experiment.py               多算法、多题、多维度、多次运行入口
run_benchmark.py                原 SADE 单题入口（向后兼容）
tests/                          问题接口、算法逻辑与运行框架测试
diagnose_candidate_pools.py     离线真实评价候选池的oracle诊断入口
replay_candidate_selectors.py   在固定候选池上回放简单选择规则
```

CEC2006 使用 `pymoo` 的 G 问题实现；CEC2010 的公式和常量来自本地 DSI 参考；CEC2017 的公式及 `.mat` 数据来自同一参考。运行时不依赖 `DSI` 目录。

## 在 python310 环境运行

```powershell
conda activate python310
cd C:\Users\Szy\Desktop\Analog_circult_sizing\code_local\SADE_Benchmark
python run_experiment.py --list
python run_experiment.py --config configs\reference\quick_test.json
python run_experiment.py --config configs\reference\cec_10d30d_25runs.json --dry-run
python run_experiment.py --config configs\reference\sade_v2_cec_10d30d_25runs.json --runs 1
python run_experiment.py --config configs\development\sade_v2_7_development.json --runs 1
python run_experiment.py --algorithms sade dsi --suites cec2017 --problems c01 c04 --dimensions 10 30 --runs 30
python run_experiment.py --algorithms all --suites all --problems all --dimensions 10 30 --runs 30
python run_experiment.py --algorithms dsi --suites cec2017 --problems all --dimensions 50 100 --runs 1 --dry-run
pytest -q
```

候选生成与代理排序可用独立oracle诊断器拆分分析，方法和当前V2.7结论见
[`docs/diagnostics/CANDIDATE_POOL_DIAGNOSTIC.md`](docs/diagnostics/CANDIDATE_POOL_DIAGNOSTIC.md)。该诊断的额外
真实评价不计入正常FE，也不会反馈给优化器。

`--dimensions` 对 CEC2006 自动忽略，因为每道题的维度固定。默认维度为 10、30；50、100 需要显式指定或使用 `--dimensions all`。相同算法、配置和 seed 可复现，且相同 run 的 seed 会在算法之间配对。

算法参数的类型、默认值与合法性检查分别位于各算法的 `config.py`。日常运行建议修改 `configs/` 下的 JSON；命令行只覆盖显式提供的字段，优先级为“命令行 > JSON > Python 默认值”。每次实验保存的是合并后的完整配置，而不只是原始 JSON。

统计汇总以 SADE 为控制算法。`summary_statistics.csv` 采用论文式宽表：全部
runs 可行时，CEC2006 显示目标误差的 `mean (std)`，CEC2010/2017 显示目标值的
`mean (std)`；只要某算法存在不可行 run，该单元格就改为成功率。配对 WSR 将
不可行 run 记为 `10e+20`，其他算法列的 `+ / ≈ / −` 分别表示 SADE 显著更好、
无显著差异和 SADE 显著更差。表底同时保存 Average Rank、Holm adjusted
p-value 和 `+/≈/−` 计数。已有实验可以重新汇总：

```powershell
python analyze_results.py <results_directory>
```

## 输出内容

每次执行会创建一个实验根目录，且每次 run 单独保存：

```text
results/experiment_<time>/
├── experiment_config.json
├── summary_runs.csv
├── summary_statistics.csv
├── sade/cec2017/c01/D10/run_001_seed_1/
│   ├── config.json
│   ├── final_result.json
│   ├── history.csv
│   ├── evaluations.csv.gz
│   └── populations.npz
└── dsi/cec2017/c01/D10/run_001_seed_1/
    └── ...
```

- `history.csv` 保存每代的 FEs、最优可行目标、最小违反度、可行数以及 RBF/EI、局部半径或 `wmax` 状态。
- `evaluations.csv.gz` 保存每个真实评价点的来源、变量、目标、逐约束值、违反度和可用的代理预测/EI 信息。
- `populations.npz` 保存每代完整种群快照。
- `summary_statistics.csv` 是单一论文式统计表，包含 `mean (std)` 或成功率、配对 WSR 符号、分测试集及总体 Average Rank、Holm adjusted p-value 和 `+/≈/−` 计数。
- 默认不保存所有未真实评价的候选池。调试时添加 `--save-candidate-pools`，每个 run 会额外生成 `candidate_pools.npz`。

逐 run 目标值以 16 位小数的科学计数法保存；论文统计表按论文样式保留两位小数。批量实验某个 run 失败时会在该 run 目录保存 `error.json` 并继续其余任务；添加 `--fail-fast` 可在首次失败时停止。
