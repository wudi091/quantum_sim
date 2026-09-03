# 规划算法

`algorithms` 只包含规划逻辑，不直接访问 SeQUeNCe 实体。

## TELGEN 主方法

- `dataset.py`：生成路径—构造候选并建立在线规划问题；
- `time_expansion.py`：把构造 DAG 映射为资源—时隙占用；
- `optimization_model.py`：单阶段期望截尾完成时延 LP/MILP 的稀疏目标和约束矩阵；
- `milp_oracle.py`：使用同一时延目标的精确 0/1 MILP 参考解；
- `ipm_trajectory_pilot.py`：TELGEN 风格 LP 内点法轨迹教师和图 GNN；
- `ipm_policy.py`：加载 IPM checkpoint，输出连续规划解并做容量安全舍入；
- `online.py`：共享滚动请求队列与 TELGEN 在线控制；
- `physical_validation.py`：把已选变量编译为中性物理计划；
- `compare_online_gnn.py`：GNN、MILP 与 Q-CAST 的配对在线比较；

IPM 轨迹 GNN 在三部图上共享迭代参数，输出连续 LP 规划解。请求结构在读出
阶段保持不变，最终使用与教师一致的共享容量安全舍入生成离散执行计划。

LP 的每个变量对应一个“请求—路径—构造—开始时隙”候选。对请求未在截止
边界前完成的情况，模型加入截尾时延惩罚；选择候选后，按其成功概率折算
完成时延。目标就是所有请求的期望截尾完成时延之和，LP 与 MILP 只在变量
域上不同，不再进行第二次目标优化。

实验协议统一由 `experiments/run_core_value.py` 驱动，分为 LP 学习质量、
拓扑泛化和在线端到端比较三组。

## Q-CAST 基线

- `qcast/expected_throughput.py`：Q-CAST EXT 期望吞吐公式；
- `qcast/online_planner.py`：按 EXT 排序路径并在共享资源—时隙合同中分配；
- `qcast/online.py`：使用与 TELGEN 相同请求队列和 SeQUeNCe 执行边界。

Q-CAST 的构造方式固定，不读取 MILP 标签或 GNN 状态。
