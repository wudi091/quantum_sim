# 第三方来源

## SeQUeNCe

- 上游：<https://github.com/sequence-toolbox/SeQUeNCe>
- 依赖：`sequence==1.0.0`
- 角色：项目唯一物理仿真后端

## ARC-Q

`algorithms/rl_routing/` contains the construction-aware online
reinforcement-learning router. It uses a sparse graph policy and a
feasibility-preserving autoregressive action process. The method is trained
from SeQUeNCe feedback and does not use LP, MILP, TELGEN, supervised labels,
or a post-hoc decoder.

## Q-CAST

- 上游：<https://github.com/sshi27/QuantumRouting>
- 本地参考目录：`QCAST/`
- 角色：Q-CAST 的 EXT、剩余资源主路径预留、恢复路径分配与断裂边修复逻辑
  的来源参考

`QCAST/` 中的上游模拟器不参与本项目实验。适配实现位于
`algorithms/qcast/`，物理执行仍统一交给 SeQUeNCe；该实现用于源码核对和
正式实验中的 Q-CAST 基线。

同一上游仓库中的以下源码也用于核对无训练基线的行为：

- `GreedyHopRouting.kt`：Greedy；
- `AdaptiveAlgorithms.kt` 中的 `CreationRate`：Q-PASS；
- `Plot.kt`：确认源码名称 `CR` 对应论文中的 Q-PASS。

## Q-PATH 与 Q-LEAP

- 作者源码：<https://github.com/infonetlijian/Fidelity-Guaranteed-Entanglement-Routing>
- 新版 SimQN 实现：<https://github.com/QNLab-USTC/Fidelity-Guaranteed-Entanglement-Routing-in-Quantum-Networks>
- 论文：Jian Li 等，*Fidelity-Guaranteed Entanglement Routing in Quantum
  Networks*，IEEE Transactions on Communications，2022，
  DOI `10.1109/TCOMM.2022.3200115`
- 参考文件：`src/qpath.py`、`src/qleap.py`、`src/mqpath.py`、
  `src/mqleap.py`

本项目没有复制这些仓库的模拟器。`algorithms/baselines/` 只把路径、保真度
与纯化资源选择原则映射到本项目统一的候选和资源—时隙合同，物理执行仍由
SeQUeNCe 完成。

Q-DDCA 未纳入、未调用，也不是本项目无训练对比方法。
