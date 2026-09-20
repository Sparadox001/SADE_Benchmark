# 实验配置

日常实验优先复制并修改 JSON 文件，不需要编辑算法源码：

- `reference/`：正式基准、快速链路检查和高维扩展。
- `development/`：V2.2–V2.13 的开发与消融。
- `optional_constraint_tightening/`：默认关闭的约束收紧实验，基础 SADE 对照暂不使用。

- `cec_10d30d_25runs.json`：CEC2006 固定维度，以及 CEC2010/2017 的 10D、30D；SADE 与 DSI 各运行 25 次。
- `sade_v2_cec_10d30d_25runs.json`：同一范围下比较 SADE、SADE V2.1 与 DSI。
- `sade_v2_2_quick_test.json`：C05/C22 10D、300 FEs 的 V2.2 单次链路与效果检查。
- `sade_v2_3_development.json`：只运行新V2.3，在CEC2017的C01/C05/C20/C22上测试10D、30D；默认3次、300 FEs。
- `sade_v2_4_development.json`：只运行新V2.4，在同一组代表题和维度上测试混合P90/连续CEI；默认3次、300 FEs。
- `sade_v2_5_development.json`：只运行新V2.5；在V2.4基础上把鲁棒可行集内的最大EI改为最小预测目标。
- `sade_v2_6_development.json`：只运行新V2.6；保留V2.5的全局点1和局部点1，删除第二个全局边界探索点。
- `sade_v2_7_development.json`：只运行新V2.7；连续3批无真实改善时，将第二槽位由局部点临时切换为边界探索点。
- `sade_v2_8_development.json`：只运行新V2.8；用历史真实5近邻目标重排全局RBF预测短名单。
- `sade_v2_9_development.json`：只运行新V2.9；首次可行前使用原始预测CV，之后恢复P90稳健约束采集。
- `sade_v2_10_development.json`：复现V2.10双尺度局部候选失败消融；不作为当前主线配置。
- `sade_v2_11_development.json`：复现首次可行前纯最近邻竞争淘汰的失败消融；不作为当前主线配置。
- `sade_v2_12_development.json`：复现首次可行前“全局槽位全局截断、第二槽位最近邻竞争”的混合淘汰诊断；默认C22 10D/30D各5次，当前不替代V2.7。
- `sade_v2_13_development.json`：复现P90候选不足时“最大联合可行概率、预测目标打破平局”的回退消融；5次闭环配对实验未优于V2.7，不作为当前主线配置。
- `cec2017_high_dimension.json`：CEC2017 的 50D、100D 扩展实验。
- `quick_test.json`：单问题、小预算的运行链路检查。
- `sade_dynamic_tightening_development.json`：基线 SADE 启用目标值动态虚拟约束的开发实验配置。
- `sade_dynamic_cec_10d30d_25runs.json`：与既有 SADE/DSI 25次实验范围严格配对的动态约束运行配置；只运行独立注册名 `sade_dynamic`，不会写入原 `sade/` 或 `dsi/` 目录。
- `dsi_dynamic_tightening_development.json`：原 DSI 与独立 `dsi_dynamic` 变体的配对开发实验；原 `dsi/` 实现和结果目录不会被改写。
- `dsi_dynamic_cec_10d30d_25runs.json`：`dsi_dynamic` 的完整43实例、25次、300 FE正式配置，用于和既有DSI结果配对比较。

运行完整配置：

```powershell
python run_experiment.py --config configs\reference\cec_10d30d_25runs.json
```

命令行只覆盖显式给出的字段：

```powershell
python run_experiment.py --config configs\reference\sade_v2_cec_10d30d_25runs.json --runs 2 --algorithms sade sade_v2
```

参数优先级为：命令行显式参数、JSON 配置、`algorithms/<name>/config.py` 默认值。

基线 SADE 的动态目标约束默认关闭。可在 `algorithms.sade` 中设置
`"dynamic_constraint_tightening": true`，或使用命令行参数
`--dynamic-constraint-tightening` 开启；`--no-dynamic-constraint-tightening` 可显式关闭。
实验注册名 `sade_dynamic` 使用同一个 SADE 实现但默认开启该开关，适合与已有
`sade`、`dsi` 结果合并比较时保持算法列和目录互不冲突。

两个 25-runs CEC 配置把 `population_size` 和 `surrogate_min_samples` 显式设为
30；部分旧的快速或高维基线配置仍显式使用 80。V2.1 用整个初始种群确定固定
约束尺度，这两个参数均可配置，Python 默认值为 30。
