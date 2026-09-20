# Benchmark 版连续 SADE 算法说明

本文档对应当前代码，不包含已经移除的电路分类器，也不包含任何离散编码或取整。

## 原代码名称与当前位置

| 原 `SADE_TED` 名称 | Benchmark 版名称与位置 |
|---|---|
| `getF_CR` | `algorithms/sade/de.py::getF_CR` |
| `generate_trials` | `algorithms/sade/de.py::generate_trials` |
| `generate_trial_candidates_group` | `algorithms/sade/de.py::generate_trial_candidates_group` |
| `local_search` | `algorithms/sade/optimizer.py::SADE.local_search` |
| `calculate_distance` | `algorithms/sade/sampling.py::calculate_distance` |
| `distance_filter_log` | `algorithms/sade/sampling.py::distance_filter_log` |
| `sample_fun` | `algorithms/sade/sampling.py::sample_fun` |
| `select_nsamples` | `algorithms/sade/surrogate.py::select_nsamples` |
| `expected_improvement` | `algorithms/sade/surrogate.py::expected_improvement` |
| `rbf_create / rbf_interp` | `algorithms/sade/surrogate.py::CubicRBF.fit / predict_with_uncertainty` |
| `Population.update_fitness` | `algorithms/sade/penalty.py::adaptive_penalty_fitness` |
| `DE.demo` 与主循环 | `algorithms/sade/optimizer.py::SADE.optimize` |

以上路径均位于 `sade_benchmark/` 下。

## 与原 SADE_TED 的一致性边界

Benchmark 版保留的是 SADE 的连续搜索骨架，不是电路程序的逐行复制。以下逻辑与原版保持一致：

- 自适应罚函数的目标归一化、逐约束满足率权重和违反量归一化公式。
- `fbest`、`Gbest`、`pbest`、种群均值四类引导信息。
- 4 个多样性变异策略和 6 个收敛变异策略；默认每个 target 随机选择 3+3 个。
- `F ~ N(0.7,0.3)`、`CR ~ N(0.5,0.3)` 的截断范围，以及不强制 `j_rand` 的原版 crossover。
- 每 10 代随机选局部中心，其余代从前 20% pbest 中选择。
- 全局/局部候选池、归一化 RMS 距离过滤、cubic RBF、核矩阵启发式方差和 EI 公式。
- 全局/局部点按 `local_fraction` 分配，以及“父代 + 本轮真实评价点”的种群更新范围。
- 第一轮候选生成和父子代淘汰均使用 `gen=0` 的罚权重，之后逐代递增。

以下是为了通用 Benchmark 或按此前决定引入的差异：

| 环节 | 原 `SADE_TED` | 当前 Benchmark 版 |
|---|---|---|
| 初始化 | 电路初始尺寸附近 ±20% 扰动 | 完整定义域 LHS |
| 问题范围 | 电路仿真，可含多目标数据结构 | 单目标、连续变量、仅不等式 `g(x)<=0` |
| 无效样本 | 约束等于 0 或 Inf 会被过滤 | 仅 NaN/Inf 无效；`g=0` 是合法边界 |
| 分类器 | 前 50 代使用随机森林 | 完全移除 |
| RBF 启用 | `gen >= 20` | 有效 archive 样本数至少达到可配置阈值，当前默认 30 |
| RBF 求解 | 正则化后 `solve` | `lstsq`，降低奇异矩阵中断风险 |
| 候选不足 | 返回 `None`，原流程可能无法继续 | 按离散版从原池补足，再用全域随机兜底 |
| 局部半径 | 以固定 125 代从 0.30 降到 0.05 | 按实际总代数从 0.30 降到 0.10 |
| 种群淘汰 | 违反约束数量优先、fitness 次优先 | 按离散版：adaptive penalty 优先、fitness 次优先 |
| 评价预算 | 固定整批，余数可能不使用 | 最后一批缩短，严格使用且不超过预算 |
| 最终解 | 当前种群中 fitness 最小 | 完整 archive 中可行优先，再取目标最小 |
| 随机数 | NumPy 固定 42，另有未统一的 Python `random` | 每个 run 一个可配置 NumPy Generator |

因此，“基本一致”适用于候选生成、罚函数、RBF/EI 和全局—局部搜索主干；初始化、代理启用时机、淘汰准则等实验行为则是明确的 Benchmark 版本定义。

## 默认超参数

默认值及参数合法性检查定义在 `sade_benchmark/algorithms/sade/config.py`。

```text
max_evaluations       = 300
population_size       = 30
batch_size            = 3
trials_per_target     = 6
local_fraction        = 0.2
surrogate_min_samples = 30
p_best_fraction       = 0.2
distance_threshold    = 1e-3
constraint_tolerance  = 0
dynamic_constraint_tightening = false
```

## 可选的目标值动态虚拟约束

基线 `sade` 可以通过 `dynamic_constraint_tightening=true` 启用约束收紧。该功能默认关闭，因此现有基准配置和随机搜索轨迹不受影响。

启用后，原始问题仍然是 `g_j(x) <= 0`，但搜索内部会在首次触发后追加：

```text
f(x) <= active_limit
```

每个完成的评价批次都在当前搜索种群上统计同时满足原始约束和当前虚拟约束的个体。数量达到 `tightening_trigger_feasible_count` 后，记录当前最优目标，并更新本阶段截至当前的历史最优目标：

```text
current_best = min(当前搜索可行目标)
phase_best   = min(本阶段此前的 phase_best, current_best)
```

若 `phase_best` 相比 `tightening_patience` 个批次前的相对改善不超过 `tightening_improvement_tol`，则判定本阶段停滞。候选新上限按副本实现的当前可行目标区间比例计算：

```text
current_worst = max(当前搜索可行目标)
new_limit = current_best
            + tightening_keep_ratio * (current_worst - current_best)
```

首次触发时建立 `active_limit=new_limit`；后续收紧还必须达到 `tightening_min_ratio`，总次数不超过 `tightening_max_tightens`。任一批的搜索可行个体数不足时，当前阶段的连续历史清空。`tightening_keep_ratio` 在这里表示目标区间位置，不是种群分位比例，因此新阈值内实际保留的个体数不固定。

每次收紧后，完整 archive 都按新目标上限重算虚拟违反量和自适应 fitness，并从完整 archive 重新组建种群。archive 不参与停滞判断或新阈值计算，只用于重标记、种群重建、代理训练和诊断。真实目标、原始约束以及 `OptimizationResult.archive_violation` 不被改写；最终解选择、成功率和统计表仍只使用原始约束。运行目录会额外保存 `dynamic_constraint_history.json`。

默认参数为：

```text
tightening_trigger_feasible_count = population_size
tightening_keep_ratio             = 0.3
tightening_patience               = 3
tightening_improvement_tol        = 1e-3
tightening_max_tightens           = 100
tightening_min_ratio              = 1e-3
```

## 完整流程

### 1. 全域初始化

在每一维的 `[lower_bound, upper_bound]` 上做 Latin hypercube sampling，产生 `population_size` 个初始点。这里不再使用电路初始尺寸附近的局部扰动。

### 2. 真实评价与约束违反量

问题接口返回：

```text
objective  = f(x)
constraints = [g1(x), ..., gm(x)]
```

所有不等式统一为 `gj(x) <= 0`，违反量为：

```text
vj(x) = max(gj(x) - constraint_tolerance, 0)
```

目标或约束出现 `NaN/Inf` 时，该样本被记为无效；约束恰好等于零是合法边界点。

### 3. 自适应罚函数

目标值在当前数据集合内做 min-max 归一化：

```text
f_norm = (f - f_min) / (f_max - f_min)
```

第 `j` 个约束在当前集合中的满足比例为 `rj`，其权重为：

```text
wj = clip((1 + 0.05 * generation) / (rj + 1e-6), 0.1, 100)
```

每个约束的违反量再除以该列当前最大违反量：

```text
penalty = sum_j((vj / max(vj)) * wj)
fitness = f_norm + penalty
```

这里有两套独立计算：

- `population_fitness`：只在当前种群内归一化，用于 DE 引导个体。
- `archive_fitness`：在完整历史 archive 内归一化，用于训练 RBF 和计算 EI。

### 4. 全局 DE 候选池

对当前种群中的每个 target，选择互不重复的 `a,b,c,d`，同时确定：

- `fbest`：当前种群目标最小个体。
- `Gbest`：当前种群自适应 penalty 最小个体；并列时随机选择。
- `pbest`：先按违反约束数量、再按 fitness 排序后的前 20% 中随机选择。
- `mean`：当前种群均值。

保留原 SADE 的 4 个多样性策略：

```text
a + F(b-c)
a + F(b-c) + F(d-target)
a + F(pbest-a) + F(b-c)
target + F(a-target) + F(b-c)
```

以及 6 个收敛策略：

```text
pbest + F(a-b)
target + F(pbest-target) + F(a-b)
target + F(pbest-target) + F(a-b) + F(c-d)
target + F(fbest-target) + F(a-b)
target + F(mean-target) + F(a-b)
a + F(Gbest-b) + F(c-d)
```

默认每个 target 随机选择 3 个多样性策略和 3 个收敛策略，因此全局池大小通常为：

```text
population_size * trials_per_target = 30 * 6 = 180
```

每个 trial 独立采样：

```text
F  ~ Normal(0.7, 0.3), clip 到 [0.2, 1.0]
CR ~ Normal(0.5, 0.3), clip 到 [0.1, 1.0]
```

随后执行原版 binomial crossover，并裁剪到变量边界。

### 5. 局部候选池

局部中心的选择与原 SADE 一致：

- `generation % 10 == 0`：从当前种群随机选一个中心。
- 其他代：从前 20% 的 pbest 集合中随机选一个中心。

局部半径采用离散版调度，但保持连续扰动：

```text
T = ceil((max_evaluations - population_size) / batch_size)
radius = max(0.10, 0.30 - (0.30-0.10) * generation / T)
```

每个局部候选随机改变：

```text
max(1, int(radius * dimension))
```

个维度，扰动为区间 `[-1,1] * radius * (upper-lower)` 上的均匀随机数。局部池大小为 `batch_size * population_size`，当前默认是 90。

### 6. 距离过滤

全局池和局部池分别去重。所有点先缩放到 `[0,1]^D`，候选与 archive 的距离定义为：

```text
EuclideanDistance / sqrt(D)
```

仅保留与 archive 最小距离大于 `1e-3` 的候选。

### 7. RBF 与 EI 采样

有效 archive 样本少于 `surrogate_min_samples` 时不训练 RBF，分别从过滤后的全局池和局部池随机选择；当前默认阈值为30。

达到 `surrogate_min_samples` 个样本后（Python 默认值为 30），每一代使用完整 archive 重新训练 cubic RBF：

```text
phi(r) = r^3
```

RBF 包含线性多项式尾项；输入按问题边界缩放到 `[-1,1]^D`，训练 fitness 缩放到 `[-1,1]`。线性系统增加 `1e-10` 正则项，并使用 `numpy.linalg.lstsq` 求解。

对每个候选，RBF 给出预测均值 `mu(x)` 和原 SADE 使用的核矩阵启发式方差 `sigma^2(x)`。令 archive 当前最小 fitness 为 `f_best`：

```text
improvement = f_best - mu(x)
Z = improvement / sigma(x)
EI(x) = improvement * Phi(Z) + sigma(x) * phi(Z)
```

`Phi` 和 `phi` 分别为标准正态分布的 CDF 与 PDF。代码中的 `expected_improvement` 返回 `-EI`，所以升序排列等价于选择 EI 最大的候选。

全局/局部配额由 `local_fraction` 决定，两个池分别执行 EI 排序。当前默认 batch 为3，因此每批通常包含2个全局点和1个局部点；若显式使用 batch=10，则对应8个全局点和2个局部点。

### 8. 候选不足

如果距离过滤后的某个候选池不足其 quota，参照离散版：保留已经过滤出来的候选，再从该原始候选池随机补足。若去重后仍未达到完整 batch，最后使用全变量域随机点兜底。每轮真实评价数始终等于计划 batch，最后一轮可自动缩短以严格满足评价预算。

### 9. 真实评价、archive 和种群淘汰

选中的候选进行真实函数评价并追加到永久 archive。种群淘汰只考虑：

```text
当前父代 + 本轮新评价个体
```

在这个临时集合内重新计算自适应 fitness 和 penalty，然后按照离散版规则排序：

```text
第一关键字：penalty，越小越好
第二关键字：fitness，越小越好
```

保留前 `population_size` 个个体进入下一代。找到可行解后不会提前停止，仍会用完评价预算继续优化目标。

### 10. 最终结果

结束后扫描完整 archive：

1. 如果存在可行解，返回其中目标值最小者。
2. 如果没有可行解，先比较逐约束归一化后的总违反量，再用目标值打破并列。

运行历史的 `sampling_method` 会明确记录每个 batch 使用的是 `random` 还是 `expected_improvement`。
