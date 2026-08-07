# Construction-Aware RL for Batch Quantum Entanglement Routing

Implementation note: the current `no_capacity_context` ablation removes the
capacity feature from the policy observation while retaining the hard
capacity-feasibility mask. Removing the mask itself requires an explicit
action-rejection transition and is a future experiment, not a current result.

本文档记录经过多轮自审查后的研究与实现方案。它是方法规划文档，不是已经完成的实验结果，也不代表已经完成最新文献查新。

## 1. 审查结论

研究方向成立，但独立审计确认原始方案不能直接作为实现或论文定稿。审计发现的阻断点已在本文档中改写为明确的前置契约和受限定理：

1. `flow-time` 的严格等价只适用于固定窗口的无折扣目标，不能与任意 `gamma < 1` 的时间折扣同时宣称完全等价。
2. `maximal independent set`、后置 projection 和 `STOP` 动作语义互相冲突，应统一为“可行 construction antichain”，不强制 maximal，也不依赖事后投影。
3. 每次固定选择完整 `(P, C)` 会削弱随机失败后的在线重规划。构造计划必须是可扩展、可修复的部分 DAG。
4. 当前 `PlanDescriptor` 和 `commit` 仍表达原子 multi-hop 计划，不能证明不同构造计划产生不同物理时间轨迹，必须先引入事件驱动 construction executor。

修订后的方案保留一个主贡献。当前仓库已经实现 neutral DTO、事件驱动 executor、SeQUeNCe 物理适配器、NumPy reference CAAPPO、可运行的 PyTorch policy heads、bounded nominal oracle 和 seeded sanity harness；仍未完成的部分继续作为 paper-complete gates，不把当前实现写成完整 CCFA 系统：

> 将批量量子纠缠路由建模为 construction-aware、事件驱动、受约束的 SMDP，并把动作定义为 construction DAG precedence frontier 与资源可行 concurrent-set family 的交集。

当前环境检查结果：

- `python -m compileall -q algorithms qnet_core QCAST` 通过；
- `qnet_core`、`algorithms`、`qcast_paper` 测试集均通过；
- 未发现已撤销的 `AllocationRequest`、`AllocationResult` 或 `generate_allocation_batch` 残留引用。

这些结果证明当前 construction-aware event foundation 可以运行，并且 SeQUeNCe 是唯一物理后端；多 demand-pair 交付、到达/deadline/expiration 的事件边界、同路径 retry 和 catalogue-bounded reroute repair 已有可运行实现，但这仍不等价于收敛的生产级 PPO、动态全路径 repair 或已完成论文实验。

## 2. Problem Anchor

### Bottom-line problem

给定一批同时存在的端到端纠缠请求，联合选择每个请求的路径和物理构造计划，以提高 batch throughput 并降低 request completion latency。

### Must-solve bottleneck

Path-only routing 隐藏了 elementary EPR 生成批次、swap 依赖、swap 并行关系和中间 memory lifetime。同一路径的不同 construction plan 会产生不同的资源占用和完成时间，因此路径选择不能独立于构造计划。

### Non-goals

- 不替换 SeQUeNCe 的 generation、memory、channel、BSM、swap、decoherence 或 fidelity 模型。
- 不把 RL 逻辑放入 `qnet_core`。
- 不把 Q-DDCA 或 Q-CAST 作为新方法的一部分；它们只作为 path-level baselines。
- 不声称复现其他模拟器的数值结果。

### Constraints

有限 quantum memory、link/BSM contention、stochastic generation and swap failure、memory decoherence、fidelity constraint，以及规划层与物理层的单向调用关系。

### Success condition

预注册 primary endpoint 为 batch window 末的 completed-request count `D_H`，secondary endpoints 为 total flow-time、mean/p95 completion latency 和 `C_{risk}`。只有在固定场景分层、多个 seed 和置信区间下，CAAPPO 的归一化目标有统计支持，且 `C_{risk}` 不恶化时，才声称相对于 path-only/fixed-construction baselines 有改善；不使用“throughput 或 latency 任意一个变好”的选择性标准。

### Independent audit remediation

| Audit issue | Resolution in this document | Status |
|---|---|---|
| Pairwise conflict cannot express finite capacity | Resource-demand vector, residual capacity, slot assignment and hyperedge semantics | Implemented for additive demands; physical scheduler remains conservative |
| Markov state was underspecified | Full `ConstructionSnapshot`/`Z_k` sufficient-statistic contract | Implemented as a simulator-neutral reference snapshot |
| Route/repair/STOP transition was open | Event-typed route action, immutable prefix, explicit release and progress rules | Implemented for fixed catalogue admission and structured retry/reroute/drop |
| Flow-time and discount were mixed | Finite-horizon undiscounted theorem; discounted variant separated | Fixed |
| DROP conflicted with completion latency | Failed requests use horizon-censored completion time and receive a lump remaining-horizon penalty at settlement | Fixed |
| Horizon-active requests could miss `C_risk` | Force timeout settlement at `T_i=H` and include them in terminal `F_K` | Fixed |
| State symbols and snapshot names diverged | Unified full state as `Z_k` and DTO as `ConstructionSnapshot` | Fixed |
| Capacity decoder completeness was underspecified | Hereditary feasible family and explicit incremental feasibility mask | Implemented for the additive resource oracle |
| Empty feasible set conflicted with illegal `STOP` | Completeness is for nonempty sets; empty action is representable only when `stop_legal=1` | Fixed |
| Potential invariance was overclaimed | Terminal `Phi=0`, fixed state-only estimator, discounted case restricted | Fixed |
| Current runtime was atomic-slot | Explicit implementation gate and migration order | Event-driven SeQUeNCe construction path implemented |

## 3. 最终方法

论文工作名：**CAAPPO: Construction-Aware Precedence-Antichain Policy Optimization**。

这里的 antichain 只指 construction DAG precedence frontier；实际动作是该 frontier 与 capacity/resource-feasible concurrent-set family 的交集。环境形式化为 construction-aware SMDP；CAAPPO 是其上的 centralized graph actor-critic 实现。

### 3.1 Construction DAG

请求 `i` 的 construction plan 是一个有向无环图：

\[
G_i^C=(V_i,E_i).
\]

每个 operation 包含：

```text
operation_id
request_id
kind: GEN | SWAP | RELEASE
predecessors
input logical segments / pair bindings
output logical segment
required resources
duration model
retry policy
fidelity rule
```

`GEN(u,v)` 产生 elementary pair；`SWAP` 消耗两个相邻纠缠段并产生更长 logical segment。一个请求最终使用的 generation edges 和 swap merge 关系共同诱导路径 `P_i` 与构造计划 `C_i`。

为控制初始动作空间，第一版可以从 K-shortest route skeletons 构造 operation universe。K-shortest 只是候选生成边界，不是固定 construction procedure；策略仍然决定 DAG 的生成顺序、merge tree、并行度和重规划。

### 3.2 State

事件时刻 `t_k` 的完整执行快照为 `Z_k`，策略观测是该快照的结构化序列化：

\[
Z_k=(X_k,Q_k,G_k^{front},M_k,B_k,t_k),
\]

其中：

- `X_k`：节点、链路、BSM 和 memory 资源状态；
- `Q_k`：pair ID、endpoint、fidelity、age、owner 和 in-flight 状态；
- `G_k^{front}`：所有请求 construction DAG 的 completed、ready 和 blocked operation；
- `M_k`：请求的 source、destination、deadline、已完成进度和当前 logical segments；
- `B_k`：物理 backend 的反馈、失败原因、完整 pending event records、attempt/retry/cancel 状态、arrival queue 和 random-hazard state；若 hazard memoryless，必须在 backend contract 中显式写出该假设。

节点 ID 只作为结构索引，不作为策略语义特征。图编码器使用 relation-aware message passing，关系包括 `connected_to`、`depends_on`、`consumes`、`owns` 和 `conflicts_with`。

### 3.3 Joint route and construction decision

路径和构造计划是一个按事件类型条件化的联合动作，而不是两个没有闭合语义的独立 head：

\[
\pi_\theta(a_k\mid Z_k)=
\pi_{\mathrm{route}}(p_k\mid Z_k,e_k)\,
\pi_{\mathrm{set}}(A_k\mid Z_k,e_k,p_k).
\]

`e_k` 是事件类型：`ADMISSION`、`EXECUTION`、`REPAIR`、`TERMINAL`。

- `ADMISSION`：按固定 canonical request order 自回归选择当前到达请求的 route skeleton，形成 route vector `p=(p_i)`；不开始物理 operation。
- `EXECUTION`：route head 输出 `NOOP`，operation head 在已提交 route 和现有 logical segments 上选择 ready set。
- `REPAIR`：只能保留已完成或已启动 operation 形成的不可撤销前缀；in-flight operation 在第一版不允许被策略取消，必须等待其 terminal event；失败分支被标记为 dead，释放的 pair/resource 经过 executor 反馈后才能重新使用。`RETRY` 重建缺失前缀，`REROUTE` 从固定 catalogue 或 bounded topology-generated repair catalogue 选择替代 `(P,C)`，用显式 RELEASE 前缀释放旧 segment，废弃旧 DAG 的未提交后缀，并把替代计划重编号到单调递增的新 DAG version。每个 request 维护已尝试的 `(route, construction)` lineage，后续 reroute 不重复尝试同一计划，但允许同一路径的另一构造计划；任意无界路径合成仍是后续 gate。
- `TERMINAL`：不再采样 action。

route skeleton 只限制候选路径空间。construction DAG 在 admission 后仍可扩展和修复，不被冻结为完整线性程序。每次新增 operation 只引用已存在的 segment，并以单调递增的 DAG version 写入，因此保持无环和路径一致性。`DROP` 是 request-level settlement，不结束整个 episode；只有所有请求 settled 或达到 `H` 才产生 episode-level `TERMINAL`。

### 3.4 Action: resource-feasible ready set

ready set 定义为：

\[
\mathcal R(Z_k)=\{o: pred(o)\subseteq done_k,\ input(o)\subseteq available_k\}.
\]

这里需要区分两个概念：`R(Z_k)` 在 construction DAG 上形成 precedence frontier；真正的并行动作是其中满足资源 packing 约束的 concurrent set，而不把任意资源关系误称为 DAG antichain。

对 operation `o` 定义资源需求向量 `d_r(o,Z_k)`，对每个容量资源 `r` 要求：

\[
\sum_{o\in A_k}d_r(o,Z_k)
\leq cap_r^{free}(Z_k).
\]

pair 独占、输入消费和不可共享的 BSM/event queue 资源使用 hyperedge 或 explicit slot assignment 表示，而不是强行转换为两两冲突边。由此定义：

\[
\mathcal F(Z_k)=\{A_k\subseteq\mathcal R(Z_k):
\text{dependency, packing, pair-consumption and route constraints hold}\}.
\]

第一版要求 `F(Z_k)` 是 hereditary：若 `A` 可行，则 `A'` 为 `A` 的任意子集时也可行。route consistency 必须编译为 operation-level local preconditions；不能把一个非 hereditary 的全局路径 predicate 直接塞进逐步 decoder。对候选 operation `o` 和当前前缀 `A_{<m}`，增量 mask 定义为：

\[
M(o\mid A_{<m},Z_k)=
\mathbf 1\{o\in\mathcal R(Z_k)\land
A_{<m}\cup\{o\}\in\mathcal F(Z_k)\}.
\]

向量 packing 的检查是线性的资源剩余容量检查；如果采用一般 slot assignment 或 hypergraph oracle，必须报告该 oracle 的复杂度，并把 Theorem 1 的适用条件写成“exact feasibility oracle available”。

解码器逐个选择 operation 或 `STOP`，只使用动态 legality mask，不使用事后 projection：

```text
ready operations
  -> dependency mask
  -> residual-capacity / slot-assignment mask
  -> pair-consumption / event-resource mask
  -> canonical-order mask
  -> operation or STOP
```

`STOP` 只有在存在 in-flight operation 或下一个 arrival/deadline/expiration event 时才合法；没有任何可推进事件时必须终止或执行明确的 `DROP`，禁止零物理时间自循环。

为避免同一集合的不同排列造成重复 credit assignment，使用稳定、全局唯一且与结构状态无关的 injective canonical key `kappa(o)`，例如由 `(request_id, DAG version, local operation ordinal, kind, endpoints)` 的结构化 tuple 派生，而不是由 route catalogue 枚举顺序派生。只允许按递增 key 解码；online repair 新增 operation 使用新的 DAG version 和 local ordinal。该 key 只用于动作表示，不输入图编码器，也不携带路由偏好。

### 3.5 Event-driven SMDP transition

策略提交 `A_k` 后，`ConstructionDAGExecutor` 原子地预留资源并启动可并行 operations。executor 聚合同一物理时间戳的 arrival、generation、swap、expiration 和 deadline events，然后推进 SeQUeNCe 到下一个会改变完整 `Z_k`、累计 reward 或未来 transition kernel 的时间戳；只有经过 backend 证明完全不影响这三者的内部事件才允许跳过。定义：

\[
(Z_{k+1},\bar\tau_k,y_k)\sim
P(\cdot\mid Z_k,a_k),
\qquad
\bar\tau_k=\min(\tau_k,H-t_k).
\]

`y_k` 至少包含 event kind、physical timestamp、success/failure、attempt ID、consumed/surviving pair IDs、fidelity、expiration、resource release 和 in-flight status。

executor 必须维护完整 in-flight queue。失败时不隐式清空整个请求：已完成的 prefix 保留，失败 branch 标记为 dead，消耗的 pair 由 backend 明确释放，新的 repair operation 只能接在 surviving segments 上。规划层不能直接修改 SeQUeNCe memory 或 pair inventory。

如果同一 timestamp 有多个 event，先按 `(physical_timestamp, event_priority, event_id)` 聚合和排序；如果 `STOP` 后不存在未来事件，则直接终止。采用 non-Zeno 假设：任意有限 horizon 内，改变 `Z_k` 的事件数有限，且 retry/release 不形成无限零时长循环。该定义保证每次非终止 transition 都有 `bar_tau_k > 0` 或只消耗一个有限的同时间戳事件集合。

### 3.6 Objective and reward

对固定 batch window `[0,H]`，请求 `i` 的 arrival time 为 `A_i`，request-level settlement time 为 `T_i`。令 `S_i=1` 表示请求在 `H` 前成功交付并通过 fidelity gate；`DROP`、expiration 和 horizon 内未成功均令 `S_i=0`。用于完成延迟评价的 horizon-censored completion time 定义为：

\[
\widetilde C_i=
\begin{cases}
T_i, & S_i=1,\\
H, & S_i=0.
\end{cases}
\]

令 `N^{pending}(t)` 为时间 `t` 已到达但尚未 request-level settled 的请求数；令 `F_k` 为事件 `k` 中 DROP、expiration 或最终 fidelity failure 的请求集合。事件区间固定为 `[t_k,t_{k+1})`，区间终点事件进入 `Z_{k+1}` 的反馈；所有 arrival/success/failure/deadline/expiration 都必须成为 event boundary。`D_k` 为区间内成功交付数，`N_k^{pending}=N^{pending}(t_k^+)`。

在 horizon event `K`，所有仍 active 的请求组成 `U_H`，强制以 timeout failure 在 `T_i=H` 结算并加入 `F_K`：

\[
F_K\leftarrow F_K\cup U_H,
\qquad
T_i=H\ \text{for }i\in U_H.
\]

因此每个失败请求恰好属于一个 `F_k`，并有 `C_{risk}=\sum_k|F_k|`。horizon timeout 的剩余 latency 补偿为 `H-T_i=0`，不会重复计算。

\[
J(\pi)=
\alpha\,\mathbb E[D_H]
-\beta\,\mathbb E\left[\sum_i(\widetilde C_i-A_i)_+\right]
-\chi\,\mathbb E[C_{risk}],
\]

\[
r_k=\alpha D_k
-\beta\left(
N_k^{pending}\bar\tau_k
+\sum_{i\in F_k}(H-T_i)
\right)
-\chi |F_k|.
\]

在上述事件划分和 horizon clipping 条件下：

\[
\sum_k\left(
N_k^{pending}\bar\tau_k
+\sum_{i\in F_k}(H-T_i)
\right)
=\sum_i(\widetilde C_i-A_i)_+.
\]

因此提前 DROP 不能通过停止 holding cost 获利：DROP 会立即补足 `(H-T_i)` 的剩余 horizon latency。`H-censored latency` 对所有请求使用 `(\widetilde C_i-A_i)_+`，并作为主要 latency summary；`conditional p95` 只在已成功交付请求上计算，必须与 completion rate 同时报告，不能单独用于胜负判断。`D_H`、censored completion latency 和 `C_{risk}` 分别按 batch size、`|B|H` 和 batch size 归一化；`alpha/beta/chi` 在训练前固定，不根据测试集调节。

主理论和主实验使用 finite-horizon undiscounted objective，即 `Gamma_k=1`。如果额外研究 physical-time discount，则明确使用另一个目标：

\[
\Gamma_k=\exp(-\rho\bar\tau_k),
\]

不能将其宣称为无折扣 flow-time 的严格等价形式。

### 3.7 Constrained actor-critic

fidelity/expiration/intentional-drop 约束定义为 trajectory-level cost，而不是按 event 次数计数：

\[
C_{risk}=\sum_i\mathbf 1\{
\text{request }i\text{ settles without a successful delivery meeting }F_i^{req}\}.
\]

在 horizon active requests 已并入 `F_K` 的约定下：

\[
C_{risk}=\sum_k|F_k|.
\]

如果研究机会约束，则另外定义 per-request violation probability；如果研究硬约束，则由 executor 在交付时拒绝低于阈值的 pair，发出一次 `fidelity_reject` event，释放该 pair 并保留请求的 surviving prefix 继续 repair。一次请求只产生一个 terminal violation indicator，后续 retry 不重复计数。双变量只优化期望约束，不提供严格可行性保证：

\[
\max_\pi J(\pi)
\quad\text{s.t.}\quad
\mathbb E_\pi[C_{risk}]\le d,
\]

\[
\lambda\leftarrow[\lambda+\eta(\widehat C_{risk}-d)]_+.
\]

使用共享异构图 encoder、route actor、operation-set actor、reward critic 和 constraint critic。PPO 只优化 `F(Z_k)` 中的合法 route/ready-set action distribution。

### 3.8 Potential shaping

定义一个固定、bounded、与 policy 参数无关的 critical-path estimator；关键路径长度先按 `H` 归一化，`kappa` 在训练前固定：

\[
\Phi(Z)=-\kappa\sum_i\widehat{L}^{crit}_i(Z).
\]

主目标使用：

\[
F(Z_k,a_k,Z_{k+1})=\Phi(Z_{k+1})-\Phi(Z_k).
\]

所有 episode-level terminal states，包括 horizon truncation、所有请求 settled 和正常完成，都令全局 `Phi=0`。request-level `DROP`/expiration 不结束 episode，只把该请求从 critical-path sum 中移除，并通过一次性 `C_risk` 计费。因此 finite-horizon shaping 项望远镜相消，不改变真实 constrained objective 的最优可行策略。若使用 discounted variant，只在 `Phi(Z_T)=0` 且使用 `Gamma_k*Phi(Z_{k+1})-Phi(Z_k)` 时讨论不变性。

## 4. 物理执行契约

下面的契约是 CAAPPO 实现的边界。当前仓库已经实现这些 DTO 和两类 executor；其中 SeQUeNCe executor 是真实物理路径，deterministic executor 只作为 contract oracle。

```text
ConstructionOperation
  op_id, request_id, kind, predecessors, inputs, outputs
  resource_demand, duration_model, retry_policy, fidelity_rule

ConstructionDAG
  version, immutable committed prefix, ready frontier, dead branches

ConstructionSnapshot
  physical_time_ps, arrivals, deadlines, pair age/fidelity/owner
  memory occupancy, reservations, in-flight operations
  complete pending event records or an explicitly equivalent sufficient summary
  backend random-hazard state (or an explicit memoryless-hazard assumption)

ExecutionEvent
  event_id, operation_id, physical_time_ps, event_kind
  success/failure, failure_cause, consumed/surviving pair IDs
  fidelity, released resources, remaining in-flight operations
```

`ConstructionDAGExecutor` 必须提供：

```text
snapshot() -> ConstructionSnapshot      # read-only, no physical side effect
launch(feasible_set) -> reservation     # atomic resource reservation
advance_to_next_event() -> ExecutionEventBatch
repair/close/terminate(...)              # explicit state transition
```

`snapshot()` 不得隐式生成 EPR、推进 SeQUeNCe timeline 或修改 pair inventory。所有 generation、swap、release 和时间推进都必须发生在 executor transition 中。

物理时间统一使用 SeQUeNCe timeline 的 ps 或一个明确的物理时间单位；logical routing slot 只能作为日志字段，不能用于 latency、TTL、pair age 或 flow-time 指标。

`WAIT` 不是 DAG operation，而是 `STOP` 后的 next-event advance；它不占用 memory/link/BSM，不产生新的 zero-duration loop。`RELEASE` 是一次性 resource-release event，必须记录释放对象，不能重复选择。每个 `GEN`/`SWAP` operation 都有 duration model、attempt index、retry limit、resource reservation 和 fidelity update；retry 是新 attempt ID，但沿用同一 DAG operation。

## 5. 理论结果与边界

论文采用以下显式建模假设并证明受限定结果，不声称 RL 全局收敛或全局最优。

### Assumption 1: Snapshot sufficiency contract

定义 `Z_k` 为包含完整 committed DAG version、surviving prefix、reservations、in-flight event queue、retry/attempt/cancel 状态、arrival/deadline queue、pair age/fidelity 和 backend random-hazard state 的执行快照。只有在 backend 明确保证这些信息对策略可观测，或给出一个等价的 sufficient summary 时，才可写：

\[
P(Z_{k+1}\mid h_k,a_k)=P(Z_{k+1}\mid Z_k,a_k).
\]

path 和 frontier-only state 的非 Markov 性作为 motivating counterexample；不把“加入若干摘要字段”直接当作无条件定理。

### Assumption 2: Non-Zeno event process

对任意有限 `H`，改变 `Z_k`、累计 reward 或 transition kernel 的事件数有限；同一 timestamp 的内部事件可有限聚合，retry/release 不产生无限零时长循环。否则 event-driven SMDP 不成立。

### Theorem 1: Relative resource-feasible ready-set encoding

对每一个决策 epoch，给定当前固定 route skeleton、当前 epoch 内固定且完整的 operation universe、hereditary feasible family `F(Z_k)` 和 exact incremental feasibility oracle，canonical decoder 的每个非空输出都属于 `F(Z_k)`；任意非空 `A\in F(Z_k)` 都有唯一 canonical decoding 序列。空集合只作为内部初始前缀存在，只有在 `stop_legal(Z_k)=1` 时才可作为环境动作输出。online repair 新增 operation 只在下一 epoch 重新建立 universe，不属于本 epoch 的 completeness 声明。

该定理对所有非空 feasible sets 给出相对于候选 operation universe 的 soundness/completeness；对空动作只给出条件表示性。它不覆盖 K-shortest catalogue 之外的网络路径，也不把资源 packing 误写成普通 pairwise independent set。若实现使用 greedy slot assignment，只保留 soundness，不再声明 completeness。全局候选截断误差必须通过 catalogue coverage 和 oracle gap 单独报告。

### Proposition 1: Censored completion-time equivalence

若 event stream 用 `[t_k,t_{k+1})` 完整划分 `[0,H]`，包含所有 arrival/success/failure/deadline/expiration events，事件区间内 `N^{pending}(t)` 恒定，使用 `bar_tau_k=min(tau_k,H-t_k)`，将 horizon active requests 以 `T_i=H` 加入 `F_K`，并对每个 failure settlement 加入 `(H-T_i)`，则：

\[
\sum_k\left(
N_k^{pending}\bar\tau_k
+\sum_{i\in F_k}(H-T_i)
\right)
=\sum_i(\widetilde C_i-A_i)_+.
\]

该恒等式对应成功请求的真实 completion latency 和失败请求的 horizon-censored latency，不再把 DROP time 错当成 completion time。

### Proposition 2: Undiscounted potential invariance

对 finite-horizon 主目标，若所有 episode-level terminal states，包括 horizon truncation、所有请求 settled 和正常完成，都令全局 `Phi=0`，且 request-level DROP/expiration 只移除对应请求的 potential contribution、不修改 shaping 规则，则：

\[
F_k=\Phi(Z_{k+1})-\Phi(Z_k)
\]

只改变学习信号密度，不改变真实 constrained objective 的最优可行策略。discounted variant 另行分析；不能用 terminal states 同值但非零的条件替代 `Phi_T=0`。

## 6. 规划层与物理层边界

`qnet_core` 只提供中立 DTO 和执行边界：

```text
ConstructionOperation
ConstructionDAG
ConstructionSnapshot
ExecutionEvent
ConstructionDAGExecutor
```

算法包负责：

```text
graph encoder
resource-feasible ready-set decoder
actor / critics
PPO update
dual multiplier
training loop
```

SeQUeNCe 负责全部 physical effects。executor 只把中立 operation 转成 backend 调用，并把真实事件反馈转换回 `ExecutionEvent`。规划层不导入 SeQUeNCe 类，也不直接修改 pair inventory。

当前代码状态是明确的 implementation gate：现有 `PlanDescriptor`、`commit`、logical-slot clock 和 `snapshot()` 副作用不能支撑 CAAPPO。必须完成下列契约后，才允许进入 RL 训练：

- `snapshot()` 纯读，不触发 generation 或 timeline advance；
- backend 暴露物理时间、next-event、launch、release 和 in-flight 状态；
- memory、link、BSM、generation slot 使用 capacity-aware reservation；
- delivery 处检查 required fidelity，并记录一次性 trajectory-level violation；
- failure 保留 completed prefix 和 surviving segments，repair 不得隐式重置整个请求；
- event log 记录 physical timestamp、operation、attempt、pair、fidelity、资源和随机数派生信息。

## 7. 实现顺序

1. 先建立 deterministic toy backend 和上述 neutral DTO，验证 DAG dependency、capacity packing、canonical ready-set decoding、repair 和 reward identity。
2. 将 SeQUeNCe timeline timestamp 接入 backend contract，统一 physical time、TTL、pair age、completion latency 和 makespan。
3. 把 generation、swap、release 从 `snapshot()` 和 atomic `commit` 移到 `ConstructionDAGExecutor`，加入 in-flight event aggregation 和 `tau>0`/terminal progress 规则。
4. 显式建模 BSM、channel、generation slot 和 memory reservations；对独立 operations 做真实并发测试，对冲突 operations 做 rejection 测试。
5. 在固定单一路径上实现 left-deep 与 balanced 两个 DAG，验证相同路径产生不同 physical event trace 和 completion time。
6. 加入 partial-prefix failure repair、required-fidelity delivery gate、risk/event metrics 和 common-random-number seed derivation。
7. 加入 K-shortest route skeleton selection 和 catalogue coverage logging；construction DAG 不在 admission 时冻结为不可修复的完整计划。
8. 先用规则策略验证 executor，再实现 CAAPPO 和 CMDP dual update。
9. 最后进行随机 generation、swap failure、decoherence 和 memory pressure 实验。

## 8. 最小实验闭环

### Experiment A: Same path, different construction

固定拓扑、请求和路径，比较 left-deep、balanced、center-out、memory-aware construction。报告 physical completion trace、makespan、mean/p95 flow-time、peak memory、expiration 和 fidelity violation。

### Experiment B: Joint routing and construction

比较：

- shortest path + fixed construction；
- K-shortest path + fixed construction；
- atomic-plan PPO；
- CAAPPO；
- deterministic CP-SAT/MILP oracle on small instances。

### Experiment C: Mechanism ablation

在同一个 event-driven executor、route catalogue、seed protocol 和 action budget 下，分别去掉 DAG state、capacity context features、flow-time reward、dual constraint 和 potential shaping，测 throughput、latency、constraint violation、mask rejection 和 sample efficiency。capacity-safety ready-set mask remains mandatory in every policy variant; `no_capacity_context` removes only the learned capacity context features, so it is not a no-mask experiment. event-driven 与 atomic-slot 的比较单独作为环境语义对照，不与其他 ablation 混合解释。

oracle 只用于小规模 deterministic nominal instances；随机 SeQUeNCe 结果不能被表述为 exact optimum。
所有实验都记录 event trace、p95、peak memory、fidelity violation、expiration、mask rejection、executor rejection 和 stochastic physical failure。独立 evaluation seed 是主要统计单位；同一 evaluation seed 上的 training replicas 先求平均，再计算 CI。operation/attempt 派生的 common-random-number stream 只作为 paired variance-reduction 分析，并明确标注其策略间耦合。

## 9. 审稿风险与必须补齐的证据

- 必须查新：construction-aware entanglement routing、swap scheduling、memory-aware quantum routing、quantum network RL。
- 必须继续扩大证据，证明 executor 在更广泛的独立 operation、arrival、expiration 和 repair 组合下仍能返回真实事件时间；当前回归已覆盖固定候选下的并发 launch 与时间一致性。
- 必须报告 K-shortest route catalogue 的可达总数、bounded repair route 的覆盖率和小规模全路径 oracle gap。
- 必须区分 mask rejection、executor rejection 和 stochastic physical failure；mask soundness 不等于物理成功率为零。
- 必须固定 primary metric、`alpha/beta/chi` 的归一化和 seed/置信区间协议，不能用“throughput 或 latency 有改善”选择性报告。
- 不能使用“first”“optimal”“solves”这类超出证据的表述，除非完成检索或小规模最优性验证。

## 10. 最终判断

修订后的文档已经消除了原审计中的理论过度声称和接口歧义。代码已经是可运行的 construction-aware CAAPPO reference 环境，但仍不能提前保证 CCFA 级别。论文贡献应谨慎写成：

1. construction-aware batch routing 的问题与 SMDP 形式化；
2. 在固定候选 operation universe 和容量模型下的 resource-feasible ready-set 表示及其相对 soundness/completeness 证明；
3. 在 SeQUeNCe 真实物理执行下，对 throughput、flow-time 和 fidelity constraints 的联合验证。

只有在第 7 节的 remaining implementation gates 全部通过、完成最新查新、并用强组合调度 baselines 与小规模 oracle 证明增益后，才可以把它作为 CCFA 投稿候选。当前实现已经跨过 event-DAG/物理后端的基础 gate，并有 Torch heads、有限 retry lineage、bounded oracle 与 sanity harness，但仍不能把它表述为收敛的完整 CCFA 方案。
