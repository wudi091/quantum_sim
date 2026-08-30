# SeQUeNCe 执行核心

`qnet_core` 是规划算法与 SeQUeNCe 物理仿真之间的中性边界。这里不包含
第二套量子网络模拟器。

## 主要模块

- `planning_spec.py`、`spec.py`：拓扑、请求和物理配置；
- `workload.py`：固定总请求数与固定到达轮数的统一解析和排空窗口推导；
- `construction_api.py`：构造操作、DAG、快照和事件合同；
- `construction_catalog.py`、`construction_plans.py`：路径与交换树候选；
- `capacity_feasibility.py`：并发操作的资源容量与输入独占检查；
- `resource_catalog.py`：规划层可见的资源容量；
- `fidelity_estimation.py`：不读取未来随机结果的保真度下界；
- `scheduled_execution.py`：中性时隙计划、持久在线调度器和通用条件后缀接口；
- `sequence_backend.py`：SeQUeNCe 节点、信道、内存和协议适配；
- `sequence_construction_executor.py`：事件驱动的构造执行器；
- `runtime.py`：组装 SeQUeNCe 执行环境。

## 调用方向

```text
规划算法
  ↓ 中性候选与离散计划
qnet_core 调度边界
  ↓
SeQUeNCe 物理事件
  ↓ 中性事件与请求结果
规划评估
```

规划层不能直接操作量子内存或 Bell 态。纠缠生成、交换、纯化、随机失败、
退相干、过期和物理时间推进全部由 SeQUeNCe 完成。

需要根据物理事件调整计划时，算法层只能返回“继续、接受已建立的端到端段、
失败或替换尚未执行的 DAG 后缀”这四类中性响应。恢复路径如何选择属于算法；
核心层只验证端点、保真度、资源、依赖与时隙并调用 SeQUeNCe。

## 检查

```bash
python -m qnet_core.sequence_smoke
python -m pytest -q qnet_core/tests
```
