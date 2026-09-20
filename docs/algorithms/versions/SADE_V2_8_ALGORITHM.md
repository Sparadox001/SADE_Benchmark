# SADE V2.8：全局RBF短名单与5近邻重排

V2.8只修改V2.7目标阶段在鲁棒预测可行集内的最终选点。候选生成、真实
可行优先排序、约束尺度、P90校准、连续CEI回退、两点批次和动态边界探索均
保持不变。

当鲁棒预测可行候选数量达到V2.7原有门槛时，先按全局RBF预测目标保留前
`objective_shortlist_fraction`，且至少保留`objective_shortlist_min_size`个。
随后对短名单中每个候选，在按变量上下界归一化的设计空间内寻找最近的
`objective_knn_neighbors`个历史真实样本，并计算反平方距离加权目标：

```text
f_local(x) = sum_i w_i f_i / sum_i w_i
w_i = 1 / max(d_i, 1e-12)^2
```

最终选择`f_local`最小的候选；完全相同时用全局RBF预测目标打破平局。默认
参数是短名单20%、至少10个候选、5个近邻。鲁棒预测可行候选不足时仍使用
V2.7的连续CEI回退；可行性阶段不使用近邻目标重排。

评估日志中的`acquisition_role`会记录
`global_robust_feasible_knn_rerank`或
`local_robust_feasible_knn_rerank`，并保存被评估点的
`local_neighbor_objective`，便于检查局部重排是否真正发挥作用。
