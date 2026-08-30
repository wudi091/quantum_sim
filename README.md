# 构造感知量子路由仿真

当前仓库只保留一条研究主链：

```text
在线精确 MILP 标签
        ↓
可行性掩码自回归 GNN
        ↓
中性离散构造计划
        ↓
SeQUeNCe 物理执行
        ↓
与 Q-PASS、Greedy 和 MILP 配对比较
```

研究对象是多请求场景下的联合路径与交换构造计划选择。MILP 先最大化
期望完成量，再在最优完成量下最小化期望完成延迟；GNN 学习 MILP 的离散
选择集合，在线推理时不调用 LP/MILP，也不使用事后硬解码或局部搜索。

## 目录

- `algorithms/telgen/`：候选展开、MILP、约束图、自回归 GNN、训练与评估；
- `algorithms/qcast/`：按官方源码适配的 Q-CAST 主路径与恢复路径基线；
- `algorithms/baselines/`：Greedy、FIFO、Q-PASS、Q-PATH、Q-LEAP 等
  无训练规划基线；
- `qnet_core/`：规划层与 SeQUeNCe 之间的中性接口和持久执行器；
- `QCAST/`：Q-CAST 上游源码参考，不参与物理仿真；
- `results/`：实验数据、模型与评估结果；
- `server_training_job.sh`：已标记失效，暂不删除，不再用于训练。

## 分层边界

规划层只处理拓扑、请求、候选构造 DAG、资源—时隙容量和离散选择结果。
它不能访问或修改 SeQUeNCe 内部对象。

SeQUeNCe 是唯一物理后端，负责纠缠生成、量子内存、交换、纯化、退相干、
保真度、随机失败和物理事件时间。规划时隙只是资源调度抽象，不替代真实
物理时间。

## 环境与检查

项目使用 Python 3.12 和 SeQUeNCe 1.0：

```bash
conda env create -f environment.yml
conda activate quantum-sim
python -m qnet_core.sequence_smoke
python -m pytest -q
```

## 基本流程

生成在线精确 MILP 标签：

```bash
python -m algorithms.telgen.generate_online_milp_data \
  --output results/milp_data \
  --episodes 20 \
  --arrival-rounds 4 \
  --requests-per-batch 5 \
  --decision-interval 4 \
  --nodes 64 \
  --min-hops 4 \
  --max-hops 4 \
  --paths 4 \
  --construction-plans 5
```

在线工作负载默认兼容旧的固定总请求数模式；正式持续流量实验使用
`--arrival-rounds R --requests-per-batch m`，表示固定运行 `R` 个到达轮次，
每轮产生 `m` 个新请求。请求总数由 `R × m` 自动推导，未完成请求保留到
TTL，到达结束后自动留出一个 TTL 排空窗口。该组织方式与 Q-CAST 官方实验
按固定轮次、固定每轮请求数评估吞吐的方式一致，但本项目保留跨轮次积压和
请求完成延迟。

训练自回归 GNN：

```bash
python -m algorithms.telgen.train_online_milp_gnn \
  --dataset results/milp_data/online_milp_dataset.json \
  --output results/gnn_model \
  --device auto
```

正式方法集合固定为 GNN、MILP、Q-PASS 和 Greedy。固定构造版本作为同一
GNN 的独立配对消融，不混入主路由方法表；Q-CAST 不进入正式对比。

在相同 EpisodeSpec 和独立同配置 SeQUeNCe 执行器上运行主对比：

```bash
python -m algorithms.telgen.compare_online_gnn \
  --checkpoint results/gnn_model/online_milp_gnn.pt \
  --output results/online_comparison \
  --comparison-profile formal \
  --arrival-rounds 4 \
  --requests-per-batch 5 \
  --seeds 20 \
  --seed-start 30000
```

`formal` 固定运行 GNN、MILP、Q-PASS 和 Greedy；`scalable` 固定省略
MILP；`construction_ablation` 只运行 GNN，用于自适应构造与固定构造的
同实例消融。入口不再提供任意增删方法的 `skip` 组合。

先运行不需要训练的基线：

```bash
python -m algorithms.baselines.run_online \
  --algorithm all \
  --output results/baselines_online
```

该命令不会加载 GNN，也不会调用 LP/MILP；所有方法仍通过独立同配置的
SeQUeNCe 持久执行器完成物理仿真。

验证构造方式选择本身是否优于固定交换树：

```bash
python -m algorithms.telgen.validate_construction_milp \
  --output results/construction_milp

python -m algorithms.telgen.validate_construction_physics \
  --output results/construction_physics
```

论文实验图由 `algorithms/telgen/plot_paper_figures.py` 统一生成，视觉规范参考
`D:/codes/qnet_sim`：Times 系字体、双栏小画布、四周坐标框、向内刻度、浅色
点状网格、顶部图例，以及颜色与线型/标记的双重编码。PDF 和 SVG 为正式
矢量输出，PNG 只用于预览。

服务器固定任务已随 IPM 范式删除而停用：`server_training_job.sh` 保留但标记失效，不再启动训练。

实验协议见 [`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md)。
