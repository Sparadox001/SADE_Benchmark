# DSI Dynamic：目标约束收紧变体

`dsi_dynamic` 是独立于 `dsi` 的单因素实验版本。原始 DSI 目录和默认行为不变。

## 唯一算法变化

达到与 SADE Dynamic 相同的停滞触发条件后，搜索内部加入：

```text
f(x) <= active_limit
```

动态违反量为：

```text
max(f(x) - active_limit, 0) / max(abs(active_limit), 1e-12)
```

它直接加到 DSI 原有总约束违反度中，参与 C2oDE 预选、epsilon 替换、候选选择、
`wmax` 自适应和 A1 淘汰。预测阶段使用目标 RBF 的 `predicted_objective` 计算该项，
不增加新的约束 RBF。

每次收紧后，完整真实 archive 按新上限重算搜索 CV，并重新构建 A1。DSI 原有的
约束模型纠错仍只判断原始物理约束，避免把目标模型误差归因于约束模型。

最终可行性、成功率、约束输出和最优解选择始终只使用原始约束；动态约束只影响
搜索过程。

## 默认参数

```text
population_size                    = 30
wmax                              = 10
dynamic_constraint_tightening     = true
tightening_trigger_feasible_count = population_size
tightening_keep_ratio             = 0.3
tightening_patience               = 3
tightening_improvement_tol        = 1e-3
tightening_max_tightens           = 100
tightening_min_ratio              = 1e-3
```

开发实验配置位于：

```text
configs/optional_constraint_tightening/dsi_dynamic_tightening_development.json
```
