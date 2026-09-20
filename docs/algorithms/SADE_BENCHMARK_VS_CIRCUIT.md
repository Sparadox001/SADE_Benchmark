# Benchmark SADE 与电路版 SADE 对照

## 对照范围

本文比较以下两个连续版本：

- Benchmark 基线：`sade_benchmark/algorithms/sade/`，按默认配置运行，`dynamic_constraint_tightening=false`。
- 电路版：`code_ted/SADE_TED/`；`data_sampling/tabpfn_eval/_sade_ted/` 是其可复现实验副本，算法主体相同，只补充了 seed、边界检查和去除 `pdb`。

本文刻意不讨论目标值动态虚拟约束（约束收紧），避免把可选实验策略混入基础算法差异。

## 一句话结论

Benchmark SADE 保留了电路版的 **10 个 DE 变异式、F/CR 分布、自适应罚函数、全局/局部候选、cubic RBF、启发式不确定性和 EI**。它不是逐行复制：初始化、无效点定义、分类器、RBF 启用条件、候选不足处理、局部半径、种群淘汰、预算和最终解选择都已改造成通用 benchmark 规则。

## 算法差异

| 环节 | 电路版 SADE | Benchmark 基线 SADE | 影响 |
|---|---|---|---|
| 问题接口 | `sim_pop` 返回 `obj, obj_min, con, con_vio`，并携带电路变量类型 | `InequalityProblem` 返回一维 `f` 和二维 `g` | Benchmark 明确限定为连续、单目标、多不等式最小化 |
| 初始化 | 围绕电路初值，每维在全范围的 ±20% 内扰动；整数变量取整 | 在完整定义域做 best-of-five maximin LHS | Benchmark 初始覆盖更全，不依赖设计初值 |
| 无效样本 | 任一原始约束值等于 0 或非有限即无效 | 只有目标/约束 NaN、Inf 无效；`g=0` 是合法边界 | 适配 CEC 的 `g(x)<=0` 语义 |
| 分类器 | 前 51 代训练 100 棵树的 balanced RF，先过滤预测无效点 | 完全移除 | Benchmark 不学习“仿真是否成功” |
| RBF 训练数据 | 仅电路版“valid” archive，目标为 archive fitness | 全部有限的真实评价 archive，目标仍为 archive fitness | 两者都不是分别拟合目标和各约束；那是 V2 系列的变化 |
| RBF 启用 | `gen >= 20` | 有限样本数达到 `surrogate_min_samples`，默认 30 | Benchmark 启用时机由数据量决定 |
| RBF 数值解 | 正则化线性系统后 `np.linalg.solve` | 去除重复节点后 `np.linalg.lstsq` | Benchmark 对奇异/病态矩阵更稳健 |
| EI | 对预测 fitness 使用同一启发式方差和 EI，取最大 EI | 公式和排序方向保持一致 | 核心代理采样逻辑保留 |
| 批次配额 | `int(0.8B)` 全局 + `int(0.2B)` 局部 | `round(0.2B)` 个局部；当 `B>1` 至少 1 个局部，其余为全局 | 电路版在 `B=3` 时实际只选 2+0=2 个；Benchmark 为 2+1=3 个 |
| 候选不足 | 距离/RF 后数量不足会返回 `None`，主流程可能无法继续 | 先从原候选池补足，再用全域随机点兜底 | Benchmark 保证计划批次完整 |
| 局部中心 | 每 10 代随机，否则从 violation-count/fitness 的前 20% 选 | 同一规则 | 保留原骨架 |
| 局部半径 | 固定按 125 代从 0.30 降到 0.05 | 按本次实际总代数从 0.30 降到 0.10 | Benchmark 后期扰动更粗，且与预算联动 |
| 种群淘汰 | 首先比较违反约束的“个数”，再比较 adaptive fitness | 首先比较归一化 adaptive penalty，再以 fitness 打破并列 | Benchmark 能区分“违反同样数量约束但违反幅度不同”的个体 |
| 评价预算 | `floor((max_evals-pop_size)/batch_size)` 个整批，余数不用 | 最后一批自动缩短 | Benchmark 严格用满且不超过 FEs |
| 最终解 | 当前 valid population 中 fitness 最小 | 扫描完整 archive；可行优先，可行时目标最小 | Benchmark 的论文统计不受最后一代归一化 fitness 影响 |
| 随机数 | 主目录版本固定 `np.random.seed(42)`，但 Python `random` 未同步；实验副本同时设置二者 | 每个 run 独立 `np.random.Generator(seed)` | Benchmark 多次 runs 可配对复现，随机状态不污染其他任务 |
| 过程保存 | 原内核基本只保留内存对象并打印；主目录末尾还有 `pdb` | 每 run 保存配置、最终值、逐代 history、全部真实评价和种群快照 | Benchmark 面向批量科研实验和追溯 |

## 保持一致的核心

### DE 候选生成

两版都使用 4 个多样性策略和 6 个收敛策略。默认每个 target 随机抽 3+3 个策略，因此种群为 30 时全局候选池通常为 180 个。

```text
F  ~ N(0.7, 0.3), clip 到 [0.2, 1.0]
CR ~ N(0.5, 0.3), clip 到 [0.1, 1.0]
```

交叉仍沿用原实现：每维以 CR 决定使用 mutant 或 target，**没有强制 `j_rand`**。

### 自适应 fitness

两版都在参与计算的数据集合内将目标缩放到 `[0,1]`，逐约束违反量除以该列最大违反量，并按当前约束满足比例赋权：

```text
w_j = clip((1 + 0.05 * generation) / (r_j + 1e-6), 0.1, 100)
fitness = normalized_objective + normalized_weighted_violation
```

Benchmark 同时计算两套尺度：当前 population 的 fitness 用于 DE 引导，完整 archive 的 fitness 用于 RBF/EI。归一化值只能在各自集合内比较，不能把两个集合的 objective normalization 直接混用。

### RBF 与 EI

两版都是一个 cubic RBF 拟合综合 fitness，而不是“一个目标 RBF + 每个约束一个 RBF”。输入缩放到 `[-1,1]^D`，训练 fitness 缩放到 `[-1,1]`，带线性多项式尾项。所谓不确定性来自 cubic kernel 矩阵的距离启发式，不是高斯过程后验方差。

```text
improvement = best_archive_fitness - predicted_fitness
z           = improvement / sigma
EI          = improvement * Phi(z) + sigma * phi(z)
```

代码返回 `-EI` 并升序取点，因此等价于最大化 EI。

## 参数对照

| 参数 | Benchmark 默认值 | 电路内核 | 电路数据生成入口默认值 |
|---|---:|---:|---:|
| `max_evaluations` | 300 | 构造时必传 | 5000 |
| `population_size` | 30 | 构造时必传 | `clip(D, 20, 60)` |
| `batch_size` | 3 | 构造时必传 | 等于 `population_size` |
| `trials_per_target` | 6 | 代码固定 6 | 6 |
| `local_fraction` | 0.2 | 代码固定 0.2 | 0.2 |
| `surrogate_min_samples` | 30 | 无此参数；固定第 20 代启用 | 同内核 |
| `p_best_fraction` | 0.2 | 代码固定 0.2 | 0.2 |
| `distance_threshold` | `1e-3` | 代码固定 `1e-3` | `1e-3` |
| `constraint_tolerance` | 0 | 使用仿真器直接给出的 `con_vio` | 同内核 |
| 局部半径 | 0.30 → 0.10 | 0.30 → 0.05，按 125 代 | 同内核 |
| RF 分类器 | 无 | 100 trees，balanced，前 51 代 | 同内核 |
| seed | 每 run 配置，默认 1 | 主目录固定 NumPy 42、Python random 未固定 | 副本默认 42，同时固定 NumPy/Python |

需要特别注意：电路 `SADE` 类本身没有 `max_evals/pop_size/batch_size` 默认值；表中 5000、`clip(D,20,60)` 和 `batch=population_size` 来自当前电路数据生成入口，不应误写成算法论文中的固定参数。

## 科研解释边界

因此，Benchmark `sade` 适合当作“从电路版抽出的通用连续 SADE 基线”，但实验论文中应明确写出上述工程改造。尤其是全域 LHS、移除分类器、30 样本启用 RBF、淘汰准则和局部半径终值，它们都会改变搜索轨迹，不能把两版数值结果视为同一实现的直接复现。
