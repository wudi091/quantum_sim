# 量子网络路由创新点：多轮检索与审计资料总档

更新时间：2026-07-27

## 1. 本档案的用途

本文件汇总当前工作区多轮 `idea-spark` 检索、全文抓取、候选生成和碰撞审计结果。目的不是继续提出新点，而是保存已经查过的材料、已经否决的方向及其原因，避免后续重复检索和重复论证。

用户约束以最后一次收敛版本为准：研究者是计算机与路由算法背景；主要考察有限同质内存槽以及逻辑 `Create / Swap / Consume`；不把物理噪声、保真度、退相干、测量结果、纯化、应用工作流或网络协议作为论文主线。经典问题迁移到量子网络可以接受，但必须明确量子网络中的新状态语义、现有量子路由为何没有解决，以及 DRL 相对经典基线承担什么职责。

完整去重索引见 [quantum_routing_literature_index.csv](quantum_routing_literature_index.csv)。该索引由 13 份 `lit_table.md` 自动合并：原始 424 条记录，按标题归并为 272 条。索引中的 `category` 是关键词启发式分类，不能替代人工相关性判断；其中保留了检索噪声，便于追溯每轮为什么召回某篇论文。

## 2. 检索覆盖与证据强度

- 使用过的检索源：arXiv、OpenAlex、Semantic Scholar。
- OpenReview 因缺少账号凭据未覆盖，因此新近在审论文的 scoop-check 不完整。
- 自动检索主要覆盖 2024–2026 年；另人工加入 2021–2025 年正式锚点论文。
- 最近一次“逻辑内存死锁”检索池为 32 篇，其中 17 篇成功抓取全文或方法段，15 篇只能依据摘要和已有文献表判断。
- 已抓取并阅读方法段的关键邻近工作包括：Dynamic Scheduling、Asynchronous Routing、RELiQ；Connection-Oriented、Fragmentation-Aware、Swapping-Based Congestion Mitigation 等部分 IEEE 论文在当前环境中全文抓取失败，相关残差只能标为摘要级判断。
- 因此，本档案不宣称“所有相关论文都已全文读完”，也不宣称穷尽了量子路由文献。

## 3. 直接相关的正式量子路由文献

| 年份 | 论文 | 正式来源 / 标识 | 本轮关注点 | 当前证据 |
|---|---|---|---|---|
| 2021 | Effective routing design for remote entanglement generation on quantum networks | npj Quantum Information, `10.1038/s41534-020-00344-4` | 远程纠缠生成的有效路径设计；基础路由锚点 | 摘要/文献表 |
| 2021 | Fragmentation-Aware Entanglement Routing for Quantum Networks | Journal of Lightwave Technology, `10.1109/JLT.2021.3070859` | 提前分配空闲量子比特时的碎片化与共享 | 摘要级；全文抓取失败 |
| 2021 | Request Scheduling in Quantum Networks | IEEE Transactions on Quantum Engineering, `10.1109/TQE.2021.3090532` | 将请求排序与路径选择分离 | 摘要级；全文抓取失败 |
| 2022 | A Connection-Oriented Entanglement Distribution Design in Quantum Networks | IEEE Transactions on Quantum Engineering, `10.1109/TQE.2022.3176375` | 连接级资源预留；可预测但可能保守 | 摘要级；全文抓取失败 |
| 2023 | Entanglement Routing Design Over Quantum Networks | IEEE/ACM Transactions on Networking, `10.1109/TNET.2023.3282560` | 顺序整数规划联合路径与聚合纠缠资源 | 摘要/文献表 |
| 2023 | A Linear Algebraic Framework for Dynamic Scheduling Over Memory-Equipped Quantum Networks | IEEE Transactions on Quantum Engineering, `10.1109/TQE.2023.3341151`; arXiv `2307.06009` | 把 EPR 生成/使用建模为入队出队，Swap 为两次出队加一次入队；静态预计算路由 | 已抓取全文/方法段 |
| 2023 | Swapping-Based Entanglement Routing Design for Congestion Mitigation in Quantum Networks | IEEE Transactions on Network and Service Management, `10.1109/TNSM.2023.3275815` | 利用 Swap 与资源重分配缓解拥塞 | 摘要级；全文抓取失败 |
| 2024 | Asynchronous entanglement routing for the quantum internet | AVS Quantum Science, `10.1116/5.0172819`; arXiv `2312.14300` | 保留动态纠缠图并异步导航，不做全局轮次重置 | 已抓取全文/方法段 |
| 2024 | On the Concurrent Multipath Entanglement Distribution in Quantum Networks | IEEE GLOBECOM, `10.1109/GLOBECOM52923.2024.10901810` | 在线并发多路径选择与资源分配，考虑量子内存 | 摘要级；全文抓取失败 |
| 2024 | Optimal routing and end-to-end entanglement distribution in quantum networks | Scientific Reports, `10.1038/s41598-024-70114-1` | ILP 与启发式的路径和 Bell-pair 资源分配 | 摘要级 |
| 2024 | Multi-Tree Quantum Routing in Realistic Topologies | IEEE Communications Magazine, `10.1109/MCOM.006.2300851` | 异步多树、局部知识与现实拓扑 | 摘要/文献表 |
| 2024 | Resource Efficient Link-Set Configuration Based Entanglement Routing | IEEE Transactions on Communications | 链路集合配置与资源效率 | 摘要级 |
| 2024 | Q-DDCA: Decentralized Dynamic Congestion Avoid Routing in Large-Scale Quantum Networks | IEEE/ACM Transactions on Networking | 大规模量子网络的动态拥塞规避 | 摘要级 |
| 2025 | Maximizing Entanglement Routing Rate in Quantum Networks: Approximation Algorithms | IEEE Transactions on Network Science and Engineering, `10.1109/TNSE.2025.3542332` | 最大纠缠路由率及近似算法 | 部分全文/摘要 |
| 2025 | Differentiated service entanglement routing for quantum networks | Quantum Science and Technology, `10.1088/2058-9565/adc82b` | 差异化服务与整体效率 | 已抓取可访问版本 |
| 2025 | ZBR: Zone-based routing in quantum networks with efficient entanglement distribution | Journal of Network and Computer Applications, `10.1016/j.jnca.2025.104156` | 分区路由与高效纠缠分发 | 摘要级 |
| 2025 | Traffic-Aware Initial Shared State for Proactive Entanglement Routing in Quantum Networks | IEEE ICC, `10.1109/ICC52391.2025.11161265` | 面向流量的预建立共享状态 | 摘要级 |
| 2025 | Quantum Network Optimization: From Optimal Routing to Fair Resource Allocation | ACM POMACS | 最优路由与公平资源分配 | 摘要级 |
| 2025 | RELiQ: Scalable Entanglement Routing via Reinforcement Learning in Quantum Networks | IEEE Transactions on Communications, `10.1109/TCOMM.2025.3640083`; arXiv `2511.22321` | GNN + MARL，使用局部信息与迭代消息交换进行下一跳/路径构造 | 已抓取全文/方法段 |

## 4. 直接相关的学习型与新近预印本

这些论文不都满足“正式发表”要求，但属于必须检查的技术碰撞。

| 论文 | 状态 | 与本项目的关系 |
|---|---|---|
| Adaptive Entanglement Generation for Quantum Routing, arXiv `2505.08958` | 预印本 | RL 选择生成链路并主动 Swap，缓存未使用纠缠；会碰撞“缓存、主动 Swap、面向未来需求”的故事 |
| Entanglement Request Scheduling in Quantum Networks Using Deep Q-Network | IEEE ICC 2025 记录 | 学习请求进入时隙的顺序、pending/dropping、延迟与公平性；不等于 EPR 级执行控制 |
| Two-Stage Deep Q Learning Routing in Entanglement Networks | 2025 记录 | 把路由、请求调度、Swap、distillation 拆成两阶段学习；会碰撞“联合动作空间”类候选 |
| A two-stage Q-learning routing approach for quantum entanglement networks | Annals of Telecommunications 2025 | 两阶段 Q-learning 路由；需作为一般 RL 基线 |
| On Utility-optimal Entanglement Routing in Quantum Networks, arXiv `2603.01197` | 预印本 / QCNC 记录 | 放松预定路由，联合效用与路径；会碰撞“路由不应预先固定”的表述 |
| Quantum Routing Beyond Pathfinding: Multipartite Entanglement Complementation, arXiv `2604.13834` | 预印本 | 直接挑战“路由必须等于寻路”的假设，但使用 multipartite entanglement，超出当前逻辑 Bell-pair 范围 |
| SatQNet, arXiv `2604.09306` | 预印本 | 卫星量子网络、directed line-GNN、去中心化运行；主要是动态物理拓扑与局部信息 |
| Stochastic Multipath Routing for High-Throughput Entanglement Distribution, arXiv `2603.25563` | 预印本 | 随机多路径与吞吐；可作为轻量控制基线 |

## 5. 邻近的量子调度、内存和资源论文

这些论文不一定属于“纯路径选择”，但它们决定某个 B 是否已经被量子网络资源管理覆盖。

| 论文 | 正式来源 / 标识 | 主要对象 |
|---|---|---|
| Entanglement buffering with two quantum memories | Quantum 2024, `10.22331/q-2024-09-03-1458` | 两节点、双内存的缓冲可用性与消费质量；已抓取全文 |
| Purification scheduling control for throughput maximization in quantum networks | Communications Physics 2024, `10.1038/s42005-024-01796-2` | 网络级并发纯化资源调度；纯化边界资料 |
| From Entanglement Purification Scheduling to Fidelity-Constrained Entanglement Routing | IEEE ICNP 2024, `10.1109/ICNP61940.2024.10858500` | 单跳纯化调度与保真度约束路径；纯化边界资料 |
| Continuously distributing entanglement in quantum networks with regular topologies | Physical Review A 2024, `10.1103/PhysRevA.110.022429` | 连续分发及 Swap 频率对虚拟邻域的影响；已抓取可访问版本 |
| An on-demand Resource Allocation Algorithm for a Quantum Network Hub and its Performance Analysis | IEEE QCE 2024 | hub 的即时准入/阻塞资源模型；已抓取可访问版本 |
| On-Demand Resource Allocation for a Quantum Network Hub | IEEE Transactions on Quantum Engineering 2025, `10.1109/TQE.2025.3641834` | 上述 hub 资源分配的正式扩展 |
| Entanglement Request Scheduling in Quantum Datacenter Networks | IEEE Network 2025, `10.1109/MNET.2025.3532847` | 已知程序需求下的纠缠请求调度；涉及数据中心/应用边界 |
| Topology Design with Resource Allocation and Entanglement Distribution for Quantum Networks | IEEE SECON 2024 | 拓扑、资源分配与分发联合设计 |

## 6. 必须承认的经典祖先与跨领域碰撞

以下不是量子路由证据，而是避免“经典换名”拒稿时必须正面引用或设置为基线的机制族。

- Banker / safe-state deadlock avoidance：固定最大资源声明与安全完成序列。
- Deadlock avoidance with flexible or alternative resource claims：可选资源序列、替代路由和柔性制造系统。
- Resource-constrained project scheduling、job-shop scheduling、online admission control。
- Crankback、preemption、rollback、restart、reservation teardown：会碰撞“释放前缀再改路”。
- Cache replacement / learned eviction：会碰撞“主动淘汰仍有效 EPR”。
- Viability kernel、safe-RL action shielding、model-checker-guided RL：会碰撞“完成可达性标签 + RL”。
- Set packing、matching、action masking：会碰撞“并发 Swap 计划集与精确 token 冲突”。
- MDP homomorphism、state abstraction、symmetry reduction：会碰撞“聚合状态掩盖精确槽位”的表述。
- Semi-Markov decision processes、options、variable-duration credit assignment：会碰撞“内部解码不应推进物理时间”。

经典迁移可以成为论文，但至少需要三个层次：量子网络中的真实反例；经典模型不能直接表达的量子状态转换或约束；相对经典算法与现有量子算法的双重基线。

## 7. 多轮候选审计台账

| 候选方向 | 对应目录 | 最终状态 | 主要结论 |
|---|---|---|---|
| “有效前缀可能是负进度” | `negative-progress-prefix-drl` | `do_not_generate` | 相关量子文献不足以建立该失败；容易退化为经典 crankback、抢占和 restart/rollback |
| 聚合状态掩盖 exact token/slot | `logical-swap-execution-gap_2` | 两次 Phase 3 `abandon` | 第一版的 twin 由动作合法性掩码直接区分，aliasing-regret 定理无对象；第二版所谓后继死角最终只是完成时序差异和普通 lookahead |
| 请求结束后的部分链 salvage | `request-salvage-routing` | Phase 3 failed | 所有权/回收语义有场景价值，但机制接近生命周期管理、回收与经典资源 salvage，未形成硬量子区别 |
| 纯化 + 内存的路径/顺序联合 | `purification-memory-routing-audit` | Phase 3 failed | 容易退化为 RCSP、RCPSP、pebbling 或 state-expanded shortest path；且偏离“不做物理层/纯化”边界 |
| 部分纯化 bundle 持有等待 | `teleportation-purification-bundle-gap` | Phase 3 failed | 仍属于纯化资源碎片化、hold-and-wait 与取消回收，且依赖被排除的纯化主线 |
| 纯化计划的 transient live-set | `teleportation-purification-memory-gap` | Phase 3 failed | 逻辑执行态存在，但与经典 live-set/ordering 调度同构，未建立不可压缩的新结构 |
| slot-atomic 并发计划集 | `atomic-batch-quantum-routing` | skill 生成完成，但不采用 | 主要修正模拟器时间与集合动作结算；涉及 TTL、退相干、BSM 等已排除内容，且易被 SMDP/sequence-to-set 文献吸收 |
| 动态量子程序的分支谱系 | `quantum-routing-scenario-gap_2` | skill 生成完成，但不采用 | 依赖 mid-circuit measurement、动态程序分支、fidelity/expiry 和应用工作流，违反当前范围 |
| 主动 rollback / reroot | `teleportation-routing-rollback-gap` | skill 生成完成，但后来否定 | 仍是释放、回退和 deadline 驱动决策；用户认为不稳，且经典 preemption/crankback 风险高 |
| 主动淘汰仍有效 EPR | `entanglement-pair-eviction-routing-drl` | 用户明确否决 | 不再继续该方向 |
| 有限内存 safe-state / 逻辑死锁 | `logical-memory-deadlock-drl_2` | 暂停，未完成 Phase 3 | 已构造真实逻辑反例，但尚未完成经典 deadlock/safe-RL 与量子论文的最终碰撞审计；不能标为“能过” |

关键终止文件：

- [前缀负进度：do_not_generate](ideaspark_run/negative-progress-prefix-drl/do_not_generate.md)
- [exact token / reachability：phase_3_failed](ideaspark_run/logical-swap-execution-gap_2/phase_3_failed.md)
- [request salvage：phase_3_failed](ideaspark_run/request-salvage-routing/phase_3_failed.md)
- [纯化 + 内存：phase_3_failed](ideaspark_run/purification-memory-routing-audit/phase_3_failed.md)
- [纯化 bundle：phase_3_failed](ideaspark_run/teleportation-purification-bundle-gap/phase_3_failed.md)
- [纯化 transient memory：phase_3_failed](ideaspark_run/teleportation-purification-memory-gap/phase_3_failed.md)

## 8. 暂停候选：有限内存下的部分链逻辑锁死

该候选仅作为已完成工作记录，不代表最终推荐。

最小反例使用四个节点 `0-1-2-3`，节点 1 和节点 2 各有两个槽位：

- 请求 `r0` 走 `0-1-2`；请求 `r1` 走 `1-2-3`。
- 为 `r0` 在物理边 `1-2` 上创建一条 EPR，再为 `r1` 在同一边创建另一条独立 EPR。
- 两次 Create 都在执行时满足容量约束，但之后节点 1、2 均占满。
- `r0` 还需要 `0-1`，却没有节点 1 的空槽；`r1` 还需要 `2-3`，却没有节点 2 的空槽。
- 每个请求只拥有一段，不能 Swap；也没有端到端 EPR 可以 Consume。因此状态没有任何合法 `Create / Swap / Consume` 后继。
- 如果先完整执行 `r0` 再执行 `r1`，两者都能完成。由此得到：全路径联合预留会过度阻塞，而逐步接受当前合法操作可能把系统锁死。

暂定 B：有限量子内存下，现有路由的路径/容量可行性不等价于部分 EPR 状态的完成可达性。

暂停前的方案是让精确枚举器为小冲突组件产生 `y_safe(s,a)` 完成可达性标签与 witness，DRL 只在已验证安全动作中优化长期吞吐；该方案仍需防守 Banker、柔性死锁避免、viability-kernel shielding 和 model-checker-guided RL，因此不能在当前状态下声称稳过。

相关中间文件：

- [Phase 1 bottleneck](ideaspark_run/logical-memory-deadlock-drl_2/phase1/phase1_output.json)
- [Phase 2 candidate](ideaspark_run/logical-memory-deadlock-drl_2/phase2_generate/phase2_generate_output.json)
- [一致性检查与待合并修复](ideaspark_run/logical-memory-deadlock-drl_2/phase2_coherence/phase2_coherence_output.json)

## 9. 已经形成的稳定认识

1. “拓扑会被路由动作改变”不能单独作为 B。动态纠缠拓扑、异步路由、中间 ebit 队列和主动 Swap 已被多条文献覆盖。
2. “内存有限”也不能单独作为 B。碎片化、连接预留、buffer、hub 资源分配、memory-equipped scheduling 都已存在。
3. “换成 RL”不能成为贡献。RELiQ、DQN 请求调度、two-stage Q-learning、Adaptive Entanglement Generation 已覆盖多个动作层次。
4. exact token/slot 适合作为正确模拟器实现细节，但若动作掩码已经读取 exact state，它本身不是新的决策问题。
5. Release、rollback、eviction、salvage 一族必须直接面对经典资源管理祖先；此前多轮均未达到稳定新颖性。
6. 如果继续坚持纯逻辑 `Create / Swap / Consume`，最可能形成论文的方向是多请求执行与调度，而不是单请求简单路径；单条无纯化链路的 Swap 顺序通常结构过于简单。
7. 经典问题迁移到量子网络是可行的，但论文故事应写成“量子逻辑状态导致经典假设失效或需要新实例化”，而不是“把经典算法名称替换成量子术语”。

## 10. 原始资料目录

所有原始运行均保存在 [ideaspark_run](ideaspark_run/) 下。建议以后优先查看以下目录，而不是重新检索：

- `logical-swap-execution-gap_2`：exact token、动作掩码、后继可达性两次失败审计。
- `negative-progress-prefix-drl`：前缀释放方向的 Phase 1 否决。
- `request-salvage-routing`：请求生命周期与部分链回收。
- `purification-memory-routing-audit`、`teleportation-purification-*`：纯化与内存方向。
- `atomic-batch-quantum-routing`：集合动作与时隙原子结算。
- `quantum-routing-scenario-gap_2`：动态程序分支需求，当前范围外。
- `teleportation-routing-rollback-gap`：释放、重根和 deadline 回放。
- `logical-memory-deadlock-drl_2`：当前暂停的 safe-state/逻辑死锁候选。

如果以后恢复调研，最有价值的工作不是再生成候选，而是补齐五篇核心正式论文的全文：Fragmentation-Aware、Connection-Oriented、Request Scheduling、Swapping-Based Congestion Mitigation、Concurrent Multipath，并用它们逐条核对“是否已存在部分资源持有的安全状态或死锁避免语义”。
