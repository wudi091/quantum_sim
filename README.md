# 构造感知量子路由仿真

当前仓库只保留一条研究主链：

```text
LP 教师样本
        ↓
TELGEN IPM 轨迹 GNN
        ↓
可行的连续规划解
        ↓
共享容量舍入
        ↓
SeQUeNCe 在线执行与基线比较
```

研究核心是可泛化的量子网络规划。GNN 在一种拓扑族上学习单阶段期望截尾
完成时延 LP 教师的规划规律，测试时迁移到未见拓扑和更大规模，并在在线
推理时不调用 LP/MILP。LP 与 MILP 使用同一个目标，MILP 仅作为离散精确
参考，不再执行第二次目标优化。连续输出通过与教师一致的容量安全舍入
转换为可执行计划。

## 目录

- `algorithms/telgen/`：候选展开、LP/MILP 教师、约束图、IPM 轨迹 GNN、训练与评估；
- `algorithms/qcast/`：共享在线环境中的 Q-CAST 路径基线；
- `qnet_core/`：规划层与 SeQUeNCe 之间的中性接口和持久执行器；
- `QCAST/`：Q-CAST 上游源码参考，不参与物理仿真；
- `results/`：实验数据、模型与评估结果；
- `experiments/`：围绕论文核心价值的固定实验协议和可扩展配置；
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

## 核心实验

实验协议集中在 `experiments/core_value_config.json`。新增拓扑、规模或负载
只需增加配置项，不需要修改规划器和物理层。

先检查协议：

```bash
python -m experiments.run_core_value --dry-run --case all
```

运行 LP 学习质量和拓扑泛化：

```bash
python -m experiments.run_core_value --case quality
python -m experiments.run_core_value --case generalization
```

运行在线 GNN、MILP、Q-CAST 配对比较：

```bash
python -m experiments.run_core_value --case online
```

完整运行：

```bash
python -m experiments.run_core_value --case all
```

服务器固定任务：

```bash
bash server_training_job.sh check
bash server_training_job.sh start
bash server_training_job.sh status
bash server_training_job.sh log
```

实验协议见 [`experiments/core_value_config.json`](experiments/core_value_config.json)，
说明见 [`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md)。
