# SADE V2.4：混合硬约束门控与连续可行性权重

V2.4 只改 V2.3 在目标阶段的候选选择规则。真实个体排序、固定约束尺度、
DE 候选池、局部搜索、P90 约束校准和每代三个真实评价点均保持不变。

## 1. 为什么需要 V2.4

V2.3 在预测可行候选存在时，只在该集合中最大化目标 EI；集合为空时退化为
最小预测 CV。后者完全丢弃目标改进信息，而且硬 P90 在困难问题上可能频繁
清空预测可行集合。

V2.4 仍优先使用可靠的硬 P90 可行集，但在该集合过小时，连续地同时考虑
目标改进概率和可行性，不再退化为纯 CV 搜索。

## 2. 联合约束残差

每个被真实评价的代理候选保留预测前的有符号归一化残差向量：

```text
r_i,j = (g_true_i,j - g_pred_i,j) / s_j
```

`s_j` 仍是初始种群确定的固定约束尺度。一个样本的全部约束残差作为一个
完整场景保存，不把不同约束假设成独立事件。仅使用最近
`constraint_error_window=30` 个场景；至少有
`constraint_error_min_samples=12` 个场景后启用经验估计。

## 3. 连续可行性权重

候选的归一化预测约束为：

```text
z_j(x) = (g_pred_j(x) - tolerance) / s_j
```

对每个历史残差场景计算：

```text
CV_i(x) = sum_j max(z_j(x) + r_i,j, 0)
```

经验联合可行概率使用 Laplace 平滑，避免小样本时直接得到 0 或 1：

```text
Pf(x) = (1 + count[all_j(z_j(x) + r_i,j <= 0)]) / (n + 2)
ECV(x) = mean_i CV_i(x)
Wf(x) = Pf(x) / (1 + ECV(x))
CEI(x) = EI(x) * Wf(x)
```

残差样本尚不足时，临时使用：

```text
Wf(x) = 1 / (1 + predicted_CV(x))
```

## 4. 硬门控与连续规则的切换

对当前经过距离过滤的候选池，若 robust-feasible 候选至少占
`hard_feasible_min_fraction=0.05`，继续沿用 V2.3：

```text
argmax EI(x), x in robust-feasible set
```

否则选择：

```text
argmax CEI(x)
```

CEI 相同时依次偏向更大的连续可行性权重和代理不确定性。边界探索槽仍使用
V2.3 的低 CV 候选短名单加最大不确定性规则。

## 5. 日志字段

`evaluations.csv.gz` 和可选的 `candidate_pools.npz` 新增：

- `empirical_joint_feasibility_probability`
- `expected_constraint_violation`
- `continuous_feasibility_weight`
- `constrained_expected_improvement`
- `constraint_residual_samples`
- `continuous_feasibility_active`

当连续规则实际选点时，`acquisition_role` 以 `_continuous_cei` 结尾；因此可以
直接统计硬 P90 与连续规则各自的触发次数、真实可行率和目标改进率。

## 6. 本版本刻意不改的内容

- 每代仍真实评价三个点，仍为两个全局槽和一个局部槽。
- 第二个全局槽仍是边界探索。
- 不增加 DSI 式代理内部多代演化。
- 不改变局部搜索扰动维数和半径。
- 不改变 RBF 结构及正则化。

这些内容留给后续独立消融，避免把连续可行性权重的作用与候选生成能力混在
一起。
