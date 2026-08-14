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
与 Q-CAST 在线基线比较
```

研究对象是多请求场景下的联合路径与交换构造计划选择。MILP 先最大化
期望完成量，再在最优完成量下最小化期望完成延迟；GNN 学习 MILP 的离散
选择集合，在线推理时不调用 LP/MILP，也不使用事后硬解码或局部搜索。

## 目录

- `algorithms/telgen/`：候选展开、MILP、约束图、自回归 GNN、训练与评估；
- `algorithms/qcast/`：共享在线环境中的 Q-CAST 路径基线；
- `qnet_core/`：规划层与 SeQUeNCe 之间的中性接口和持久执行器；
- `QCAST/`：Q-CAST 上游源码参考，不参与物理仿真；
- `results/`：实验数据、模型与评估结果；
- `server_training_job.sh`：服务器固定配置训练任务。

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
  --requests 20 \
  --requests-per-batch 5 \
  --decision-interval 4 \
  --nodes 64 \
  --min-hops 4 \
  --max-hops 4 \
  --paths 4 \
  --construction-plans 5
```

训练自回归 GNN：

```bash
python -m algorithms.telgen.train_online_milp_gnn \
  --dataset results/milp_data/online_milp_dataset.json \
  --output results/gnn_model \
  --device auto
```

在相同 EpisodeSpec 和独立同配置 SeQUeNCe 执行器上比较 GNN、MILP 与
Q-CAST：

```bash
python -m algorithms.telgen.compare_online_gnn \
  --checkpoint results/gnn_model/online_milp_gnn.pt \
  --output results/online_comparison \
  --seeds 20 \
  --seed-start 30000
```

验证构造方式选择本身是否优于固定交换树：

```bash
python -m algorithms.telgen.validate_construction_milp \
  --output results/construction_milp

python -m algorithms.telgen.validate_construction_physics \
  --output results/construction_physics
```

服务器固定任务：

```bash
bash server_training_job.sh check
bash server_training_job.sh start
bash server_training_job.sh status
bash server_training_job.sh log
```

实验协议见 [`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md)。
