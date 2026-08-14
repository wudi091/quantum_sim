# 第三方来源

## SeQUeNCe

- 上游：<https://github.com/sequence-toolbox/SeQUeNCe>
- 依赖：`sequence==1.0.0`
- 角色：项目唯一物理仿真后端

## Q-CAST

- 上游：<https://github.com/sshi27/QuantumRouting>
- 本地参考目录：`QCAST/`
- 角色：Q-CAST 算法与 EXT 公式的来源参考

`QCAST/` 中的上游模拟器不参与本项目实验。正式 Q-CAST 基线位于
`algorithms/qcast/`，物理执行仍统一交给 SeQUeNCe。
