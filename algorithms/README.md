# 规划算法

`algorithms` 只包含规划逻辑，不直接访问 SeQUeNCe 实体。

## TELGEN 主方法

这里目前存在两条必须分开表述的学习路线：

- `ipm_trajectory_pilot.py`：参考 TELGEN 论文与官方提交
  `64684ebb3a7e856de86346da46232f8ceca6666c` 的连续 LP 轨迹学习试验。它用
  SciPy `interior-point` callback 记录教师原变量轨迹，使用变量—约束—目标
  三部图、六类有向关系、共享外层 IPM 循环和三项论文损失。默认在同一拓扑
  族的多张、多规模图上训练，再测试更大未见图；不使用解码器、投影、贪心
  修复或量子语义人工特征。它只输出连续 LP 松弛解，尚不是在线离散调度器。
- 以下 `milp_imitation.py` 到 `online.py` 的现有主链：自回归 GNN 学习精确
  0/1 MILP 的离散选择。它是本项目此前的量子网络方法，不是原始 TELGEN
  IPM 轨迹复现，也不能与上面的 pilot 共用实验口径或 checkpoint。

- `dataset.py`：生成路径—构造候选并建立在线规划问题；
- `time_expansion.py`：把构造 DAG 映射为资源—时隙占用；
- `optimization_model.py`：两阶段 MILP 的稀疏目标和约束矩阵；
- `milp_oracle.py`：精确 0/1 MILP 标签教师；
- `milp_imitation.py`：候选—约束图、自回归状态和 GNN；
- `online_milp_dataset.py`：在线标签数据读取；
- `train_online_milp_gnn.py`：训练、验证和 checkpoint 选择；
- `gnn_policy.py`：加载 checkpoint 并直接输出离散动作序列；
- `online.py`：共享滚动请求队列与 TELGEN 在线控制；
- `physical_validation.py`：把已选变量编译为中性物理计划；
- `comparison_methods.py`：固定正式方法集合与论文展示顺序；
- `compare_online_gnn.py`：GNN、MILP、Q-PASS 与 Greedy 的配对在线比较；
- `validate_construction_milp.py`：自适应交换树与固定交换树的 MILP 消融；
- `validate_construction_physics.py`：在 SeQUeNCe 中复放构造消融计划。

GNN 每一步在当前可行候选和 `STOP` 中作一次模型决策。请求唯一性与
资源—时隙容量在 softmax 前形成动态动作掩码，因此不会先输出任意连续分数
再交给独立硬解码器修复。

运行论文对齐的 IPM 轨迹 pilot：

```bash
python -m algorithms.telgen.ipm_trajectory_pilot \
  --train-nodes 10 12 14 \
  --test-nodes 18 20 \
  --train-samples 24 \
  --validation-samples 8 \
  --test-samples 8
```

这里的“一个拓扑训练”指同一个拓扑生成族（例如 Waxman）中的多张随机图，
不是固定一张邻接图反复生成请求。跨到 Barabasi--Albert 的结果只作为额外
压力测试，不能冒充原论文的同分布规模泛化结论。

正式主对比固定为 `GNN / MILP / Q-PASS / Greedy`。固定构造版本属于
construction-awareness 消融，在相同 episode 上与自适应构造 GNN 配对比较；
它不是第五种外部路由算法。大规模泛化和负载实验允许因精确求解成本省略
MILP，但不会替换或新增其他路由基线。`compare_online_gnn.py` 通过
`formal`、`scalable` 和 `construction_ablation` 三个固定 profile 选择上述
集合，不支持任意拼接方法。

## Q-CAST 适配（非正式主对比）

- `qcast/expected_throughput.py`：Q-CAST EXT 期望吞吐公式；
- `qcast/online_planner.py`：按实时剩余资源反复选择并预留最高 EXT 主路径，
  再为主路径区间预留恢复路径；
- `qcast/recovery.py`：根据真实生成结果识别断裂边、选择恢复路径并生成
  新的交换后缀；
- `qcast/online.py`：使用与 TELGEN 相同请求队列和 SeQUeNCe 执行边界。

Q-CAST 不读取 MILP 标签或 GNN 状态。其主路径选择、全局逐路径预留、恢复
路径分配和断裂边覆盖来自官方 `OnlineAlgorithm.kt`；上游模拟器没有接入，
所有生成、交换和随机失败仍由 SeQUeNCe 执行。该实现保留用于源码核对和
补充实验，不属于当前固定正式方法集合。

## 无训练基线

- `baselines/planner.py`：Greedy、Strict FIFO、Best FIFO、Q-PASS、Q-PATH、
  Q-LEAP 的统一资源—时隙适配；
- `baselines/online.py`：复用同一滚动请求队列、资源预留和 SeQUeNCe
  持久执行器；
- `baselines/run_online.py`：在同一个 `EpisodeSpec` 上配对运行一个或全部
  无训练方法。

这些方法不加载模型，也不调用 LP/MILP。Greedy、Q-PASS 以 Q-CAST 作者源码
为行为依据；Q-PATH、Q-LEAP 以作者公开的单请求和多请求源码为依据。
第三方源码自身的拓扑、资源和物理模拟器没有接入本项目，只有规划原则被
适配到统一合同中。Strict FIFO 与 Best FIFO 是本项目的调度控制组，不声称是
第三方官方复现。

为避免给传统路由方法额外的构造感知能力，每个基线固定使用一种交换构造。
Q-PATH 与 Q-LEAP 可在原始保真度不足时选择当前环境已支持的一轮基础链路
纯化。当前 `EpisodeSpec` 对所有边使用同一物理参数，因此 Q-PASS 的链路代价
和 Q-LEAP 的原始保真度排序在部分拓扑上会退化为跳数偏好，这是场景模型的
性质，不是额外规则。

```bash
python -m algorithms.baselines.run_online \
  --algorithm all \
  --output results/baselines_online \
  --arrival-rounds 20 \
  --requests-per-batch 10 \
  --decision-interval 4
```

`--arrival-rounds` 固定流量轮数，每轮生成 `--requests-per-batch` 个新请求；
总请求数自动推导。旧的 `--requests` 仅用于复现历史固定总量实验，两者不能
同时设置。

Q-DDCA 不在这组基线中。
