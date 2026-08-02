# 离线交换日程候选库与在线批请求联合选择

## 0. 当前研究主张

本文研究集中式量子网络中的在线批请求控制。对给定网络拓扑，离线阶段为每个源—目的节点对计算 4 条候选路径，并为每条路径选择 4 个完整交换日程，形成每个请求最多 16 个“路径—日程”候选。在线控制周期到来时，控制器只读取候选库、附加当前资源状态和请求冲突特征，再一次性选择能够共同执行的方案。

本文不再把以下内容作为创新：

- 用 swap group 表示并行交换；
- 允许多个节点同时执行 BSM；
- 用非线性顺序替代 sequential order；
- 声称交换日程“不是 tree”；
- 单独使用 GNN 或 RL。

当前工作的两个核心创新为：

1. **不读取在线请求的拓扑级路径—完整交换日程候选库生成。**
2. **面向在线批请求的跨路径、跨请求交换日程联合选择。**

一句话概括：

> 现有 swapping-tree 方法主要为单条路径寻找一个局部较优方案；本文在部署前仅根据固定拓扑枚举合法路径与完整日程，以路径质量和资源重叠选择 4 条互补路径，再从日程的 Pareto 释放前沿中选择最多 4 个互补日程。真实请求到达后不运行 LP，而由 GNN + RL 从每个请求的最多 16 个缓存候选中联合选择。

离线生成器不读取 request ID、arrival slot 或真实请求组合。在线 MILP 只在小规模实验中作为统一评价 oracle，用来比较不同离线生成器提供的候选库；它不属于部署时的 CON 决策流程。

---

## 1. 研究问题

量子网络表示为：

$$
G=(V,E),
\qquad
M_v=\text{节点 }v\text{ 的量子内存容量}.
$$

在控制周期 $t$，等待处理的请求集合为：

$$
\mathcal R_t
=
\{r\mid r\text{ 已到达、尚未完成且未过期}\}.
$$

每个源—目的节点对离线保存 4 条候选路径，因此请求 $r$ 具有：

$$
\mathcal P_r=\{p_{r,1},p_{r,2},p_{r,3},p_{r,4}\}.
$$

每条路径离线保存 4 个完整交换日程：

$$
\mathcal S(p_{r,i})
=
\{\Pi_{r,i}^{(1)},\Pi_{r,i}^{(2)},\Pi_{r,i}^{(3)},\Pi_{r,i}^{(4)}\}.
$$

所以每个请求最多具有：

$$
4\times4=16
$$

个“路径—完整交换日程”候选。相同路径采用不同日程会改变：

- 哪些 BSM 可以同时执行；
- 中间纠缠在各节点停留多久；
- 哪个共享节点先释放内存；
- 其他请求何时能够启动新的 elementary-entanglement generation；
- 多条交织路径能否在同一控制周期内共同推进。

离线保存多个日程并不是为了展示“交换顺序很多”，而是为了给在线批请求控制器保留不同的资源释放方式。路径本身只说明请求经过哪些节点，并没有说明共享节点何时释放；因此仅选择路径不足以决定多个请求能否在同一控制周期内共同执行。

因此，本文优化的对象不是单独的 routing path，也不是单独的下一次 swap，而是：

> 当前网络快照下的“请求—路径—完整交换日程”组合。

主目标为最大化整个 episode 中完成的唯一请求数：

$$
\max_\pi
\quad
\mathbb E_\pi
\left[
\sum_{t=0}^{H-1}
N_t^{\mathrm{complete}}
\right].
$$

---

## 2. 对现有交换顺序生成方法的功能性批评

创新点 1 不能建立在“交换树无法表示本文日程”这一说法上。对一条线性路径，每次交换都将两个相邻纠缠段合并；把 elementary links 作为叶节点、swap 作为内部节点，任何合法完整日程都可以还原成一棵带执行轮次标签的二叉交换树。反过来，也可以从带轮次的树得到：

$$
\Pi=(G_1,G_2,\ldots,G_K).
$$

因此，tree 与 ordered swap groups 在加入执行轮次后可以描述同一类单路径方案。本文不声称 group representation 比 tree 具有更大的可行解空间。

真正的功能缺口是生成目标不同。现有方法主要回答：

> 给定一条路径，哪一棵交换树或哪一个交换顺序对该路径本身较好？

本文需要回答：

> 在每个请求只能保留 4 条路径、每条路径只能保留 4 个日程时，离线数据库应当保存哪一组候选，才能让未来在线批请求最多地完成？

| 方法类别 | 代表工作 | 已提供的功能 | 本文所需但尚未直接提供的功能 |
|---|---|---|---|
| 单路径 tree/order 生成 | Order Matters、QNR、Greedy Order | 为给定路径生成一个或若干完整 tree/order | 把有限候选库作为一个整体，直接按下游批请求完成数选择互补日程 |
| 在线下一步决策 | Iñesta MDP、Real-Time Ordering、RL Swapping | 根据当前状态选择下一次 swap 或 swap-set | 离线构建可被在线控制器直接读取的完整日程目录 |
| 固定并发策略 | swap-asap、Haldar、Concurrent Swapping | 支持并发交换或固定 pairing 规则 | 在相同候选预算下系统选择资源释放行为互补的日程组合 |
| 多用户触发执行 | M-PSES | 跨路径并行、资源锁定/释放和 trigger signal | 不依赖固定路径优先级和唯一 Layer/Segment Greedy 日程，而是在线联合选择离线保存的路径—日程候选 |

### 2.1 树能表示，但“单个最好”不等于“候选组合最好”

如果只处理一条路径，交换树已经足够描述依赖关系、顺序和并行层次。本文提出新方法的理由不是替换树这一数据结构，而是把优化对象从一个日程改为有限候选组合：

$$
\text{现有：}\quad p\longrightarrow T^*(p),
$$

$$
\text{本文：}\quad
(\mathcal D,\Omega,B)
\longrightarrow
\mathcal L^*.
$$

其中 $\mathcal D$ 是离线请求场景集，$\Omega$ 是合法路径—日程候选池，$B$ 是“4 条路径、每条路径 4 个日程”的候选预算，$\mathcal L^*$ 是最终保存的离线候选库。

四个单独指标最好的日程可能具有近似相同的资源释放行为。本文关心的是这四个日程作为一个集合，是否能在不同请求场景中提供互补的联合调度选择。

### 2.2 在线局部决策仍不是离线完整候选库

Iñesta、Real-Time Ordering 和相关 RL 方法可以根据当前状态选择 swap 或 swap-set，但它们的直接输出是“当前执行什么”，而不是供后续在线批控制反复读取的完整候选库。

本文离线保存的原子模板为：

$$
\text{path}
+
\text{complete schedule}
+
\text{time-indexed resource footprint}.
$$

在线请求到达后，再把 request ID 和当前资源特征附加到模板上。

### 2.3 最接近的 M-PSES

M-PSES 已经研究多用户路径交织、节点资源提前释放和跨路径并行，不能再声称本文首次发现这些现象。

本文与 M-PSES 的功能区别表述为：

> M-PSES 在给定路径优先级和 Layer/Segment Greedy 规则下执行 trigger 与资源锁定；本文离线构建多个完整路径—日程候选，在线再根据当前批次联合决定请求、路径和日程组合。

可写入引言的批评句：

> Existing swapping-tree methods optimize schedules mainly at the level of an individual path. They do not directly optimize which fixed-size portfolio of complete path–schedule candidates should be retained to maximize the downstream number of completed requests across batch scenarios.

---

## 3. 创新点 1：拓扑级路径—交换日程候选库生成

离线阶段的输入只有固定拓扑及其静态资源参数：

$$
G=(V,E),\quad M_v,\quad c_e,\quad p_e.
$$

离线生成器不读取真实请求、arrival slot、在线 memory snapshot 或未来请求组合。它只负责为每个无序节点对准备最多 16 个可执行候选，并将结果缓存到磁盘。

### 3.1 路径池与路径组合选择

对每个节点对 $(s,d)$，先使用 shortest-simple-path enumeration 生成最多 8 条路径池。路径 $P$ 的拓扑质量包括：

$$
H(P)=|P|-1,
$$

$$
R(P)=\sum_{e\in P}-\log(\max(p_e,\epsilon)),
$$

$$
E(P)=\sum_{e\in P}\frac{1}{c_e},
$$

$$
N(P)=\sum_{v\in P^\circ}\frac{1+b_v}{M_v},
$$

其中 $b_v$ 是 topology betweenness。对同一节点对内的指标归一化后，路径成本为：

$$
C(P)=
0.35\hat H+0.30\hat R+0.15\hat E+0.10\hat N+0.10\hat B.
$$

$\hat B$ 表示链路容量和中间节点内存的瓶颈项。

路径重叠同时考虑共享边和共享内部节点；低容量边与低内存节点具有更高重叠权重。因为最多只是 8 选 4，每个节点对最多枚举：

$$
\binom{8}{4}=70
$$

个路径组合，因此直接穷举结构目标，不需要 LP。组合目标同时惩罚平均路径成本、最差路径成本、平均资源重叠和最大资源重叠。

### 3.2 完整交换日程特征

对路径 $P=(v_0,v_1,\ldots,v_{n+1})$，完整日程为：

$$
\Pi=(G_1,G_2,\ldots,G_K).
$$

定义内部节点 $v_i$ 的释放轮次：

$$
r_i(\Pi)=k\iff v_i\in G_k.
$$

日程的 topology-only 资源代理包括：

$$
K(\Pi)=|\Pi|,
$$

$$
M(\Pi)=2\sum_i r_i(\Pi)+2K(\Pi),
$$

以及同轮执行的节点对集合和加权 memory-time。这里的指标只描述固定日程结构，不使用在线请求冲突。

### 3.3 Pareto 释放前沿

若日程 $\Pi_a$ 满足：

- 交换轮数不多于 $\Pi_b$；
- memory-time 不大于 $\Pi_b$；
- 并行关系不少于 $\Pi_b$；
- 每个内部节点的释放轮次都不晚于 $\Pi_b$；
- 至少一项严格更好；

则称 $\Pi_a$ 支配 $\Pi_b$。被支配日程不进入正式候选库。

在当前最多 5 个内部节点的设置下，合法日程数与 Pareto 数分别为：

| 内部节点数 | 0 | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|---:|
| 合法完整日程数 | 1 | 1 | 2 | 7 | 34 | 214 |
| Pareto 日程数 | 1 | 1 | 2 | 3 | 6 | 12 |

正式默认生成器从 Pareto 前沿中选择：

1. 一个低轮数、低 memory-time 的质量锚点；
2. 能最大幅度提前尚未覆盖节点释放时间的日程；
3. 与已选释放向量差异最大的非支配日程；
4. 重复上述边际覆盖，直到达到 4 个或 Pareto 前沿耗尽。

短路径不会复制日程补足 4 个槽位，剩余位置由 `valid_mask` 标记为不可用。

### 3.4 已实现的生成器消融

当前实现并比较：

1. `canonical`：最短路径 + 低轮数日程；
2. `quality`：路径质量 + 最低加权 memory-time；
3. `banded`：质量带内 max-min release diversity；
4. `pareto`：精确路径组合 + Pareto release coverage；
5. `exact_kcenter`：精确路径组合 + Pareto 前沿 anchored k-center。

生成完成后，每个无序节点对保存一个固定 $4\times4$ 网格：

$$
\text{slot}=4\times\text{path slot}+\text{schedule slot}.
$$

反向请求复用同一缓存 entry，只反转路径方向并重新构造 dependency tree；template ID、slot 和 mask 保持不变。

### 3.5 评价协议

不同离线生成器不观察评价请求。对每个固定 topology seed：

1. 各生成器分别产生并冻结候选库；
2. 使用完全相同的在线请求 trace；
3. 使用完全相同的物理随机数；
4. 在线 MILP oracle 只在当前候选库中选择请求—路径—日程组合；
5. 统一事件环境执行所选方案。

在线 MILP 只用于小规模评价，不属于实际部署算法。正式在线算法仍为 GNN + RL。

<!-- 以下为已废弃的“离线场景 MILP 生成库”设计，保留在源码中仅供历史追踪。

## 3-legacy. 离线场景 MILP 候选库

### 3.1 离线候选库

对源—目的节点对 $(s,d)$，离线阶段保存：

$$
\mathcal P_{sd}
=
\{p_{sd}^{(1)},p_{sd}^{(2)},p_{sd}^{(3)},p_{sd}^{(4)}\}.
$$

每条路径保存 4 个完整日程：

$$
\mathcal S(p_{sd}^{(i)})
=
\{\Pi_{i,1},\Pi_{i,2},\Pi_{i,3},\Pi_{i,4}\}.
$$

其中：

$$
\Pi_{i,j}
=
\left(
G_1^{(i,j)},
G_2^{(i,j)},
\ldots,
G_{K_{i,j}}^{(i,j)}
\right),
$$

$G_k^{(i,j)}$ 是第 $k$ 个可并行执行的 swap group。一个离线模板记录：

$$
c=(p,\Pi,\rho,\chi),
$$

其中 $\rho$ 是节点级资源占用与释放轨迹，$\chi$ 是路径、交换组和资源冲突签名。请求 $r=(s,d)$ 在线到达后，直接实例化：

$$
\mathcal C_r
=
\{(r,p_{sd}^{(i)},\Pi_{i,j})\mid i,j=1,\ldots,4\},
$$

因此每个请求最多有 16 个候选。

### 3.2 “离线最优”的严格定义

离线生成请求与资源场景集：

$$
\mathcal D
=
\{\omega_1,\omega_2,\ldots,\omega_N\}.
$$

每个场景使用与正式环境相同的拓扑、请求到达逻辑、容量约束和确定性 planning model。设候选库为 $\mathcal L$，在场景 $\omega$ 中使用该候选库能够完成的最多请求数为：

$$
F(\omega,\mathcal L).
$$

离线最优候选库定义为：

$$
\mathcal L^*
=
\arg\max_{\mathcal L}
\sum_{\omega\in\mathcal D}
q_\omega F(\omega,\mathcal L),
$$

约束为：

$$
\text{每个 }(s,d)\text{ 保存 4 条路径},
$$

$$
\text{每条保存路径保留 4 个完整交换日程}.
$$

如果场景等权，则 $q_\omega=1/N$。这里的“最优”不是指每个日程单独最短，而是指整套固定候选库在离线场景集上支持完成的请求总数最大。

由于路径和日程选择都是离散变量，准确模型是 0-1 MILP，而不是连续 LP。普通 LP 只能作为松弛上界，不能直接输出可执行候选库。

该最优性只相对于以下内容成立：

- 给定的合法路径—日程候选池；
- 给定的离线场景集 $\mathcal D$；
- 给定的“4 路径 $\times$ 4 日程”预算；
- 给定的确定性 planning model。

因此不能把 $\mathcal L^*$ 称为所有未来随机物理实现上的无条件全局最优。

### 3.3 离线 MILP

设 $u_{sd,p}\in\{0,1\}$ 表示是否为 $(s,d)$ 保存路径 $p$：

$$
\sum_{p\in\mathcal P_{sd}^{\mathrm{pool}}}
u_{sd,p}
=
\min\left(4,\left|\mathcal P_{sd}^{\mathrm{pool}}\right|\right).
$$

设 $z_{sd,p,\Pi}\in\{0,1\}$ 表示是否把完整日程 $\Pi$ 保存到路径 $p$ 下：

$$
\sum_{\Pi\in\Omega(p)}
z_{sd,p,\Pi}
=
\min\left(4,|\Omega(p)|\right)u_{sd,p}.
$$

这里必须使用有效预算 $\min(4,|\Omega(p)|)$。短路径可能只有
1 个或 2 个合法完整日程，不能复制相同日程来补足 4 个候选。因此本文
统一表述为“每个源—目的节点对最多保存 4 条路径、每条路径最多保存
4 个唯一日程、每个在线请求最多具有 16 个候选”。

其中 $\Omega(p)$ 是路径 $p$ 的全部或大规模合法完整日程池。

对离线场景 $\omega$，设 $x_{\omega,r,p,\Pi}\in\{0,1\}$ 表示场景中的请求 $r$ 是否使用候选 $(p,\Pi)$。在线场景模拟只能使用已保存候选：

$$
x_{\omega,r,p,\Pi}
\le
z_{s(r)d(r),p,\Pi}.
$$

每个请求最多使用一个候选：

$$
\sum_{p,\Pi}
x_{\omega,r,p,\Pi}
\le1.
$$

在确定性 planning model 中，定义：

$$
y_{\omega,r}
=
\sum_{p,\Pi}
x_{\omega,r,p,\Pi},
$$

表示请求 $r$ 是否被该场景的最优选择器完成。

再加入每个场景中的节点—阶段容量约束、路径和完整日程可行性约束。目标为：

$$
\max
\quad
\sum_{\omega\in\mathcal D}
q_\omega
\sum_{r\in\mathcal R_\omega}
y_{\omega,r}.
$$

变量 $u,z$ 决定离线数据库保存什么；变量 $x,y$ 模拟未来各场景中的最优在线选择。因为 $u,z$ 在所有场景之间共享，该 MILP 会选择一套能够跨场景支持最多请求的固定候选库。

若第一阶段不想同时优化路径，可以先用 k-shortest、Q-CAST 或其他固定方法得到 4 条路径，把 $u$ 固定，只用 MILP 为每条路径选择 4 个日程。

### 3.4 离线求解流程

推荐采用“领域算法生成合法池 + MILP 选择最优组合”：

1. 对每个 $(s,d)$ 生成候选路径池；
2. 对每条路径使用 tree enumeration、递归搜索或动态规划枚举合法完整 swap-group schedules；
3. 用统一执行模型验证每个日程，并计算资源占用与释放轨迹；
4. 用与正式环境相同的请求分布生成离线场景集 $\mathcal D$；
5. 求解上述 0-1 MILP，选择 4 条路径及每条路径的 4 个日程；
6. 将候选库保存到磁盘；
7. 在线阶段只按 $(s,d)$ 读取 16 个候选，不再运行生成 MILP。

这样做比让 MILP 从零描述所有纠缠段状态转移更容易验证正确性，也避免用同一个局部目标独立求解 4 次而得到近似重复的日程。

### 3.5 为什么生成方式仍然重要

设当前批次在完整合法候选空间中的最优完成数为：

$$
\operatorname{OPT}(\Omega_t),
$$

使用离线候选库实例化出的有限候选集合时为：

$$
\operatorname{OPT}(\mathcal C_t).
$$

GNN + RL 不能创造离线库之外的新路径或交换顺序，因此总损失可以分为：

$$
\underbrace{
\operatorname{OPT}(\Omega_t)
-
\operatorname{OPT}(\mathcal C_t)
}_{\text{离线候选库损失}}
+
\underbrace{
\operatorname{OPT}(\mathcal C_t)
-
J_{\mathrm{RL}}
}_{\text{在线选择损失}}.
$$

离线 MILP 负责减小第一项；GNN + RL 负责减小第二项。如果已经把全部合法日程都提供给在线控制器，则候选生成方式不再影响候选空间最优值，创新点 1 也应删除。

### 3.6 一个具体例子

考虑路径：

$$
A-B-C-D-E.
$$

离线候选库中可以同时保存：

$$
\Pi_1=(\{B,D\},\{C\}),
$$

以及：

$$
\Pi_2=(\{C\},\{B\},\{D\}).
$$

如果未来某一批请求主要竞争节点 $C$，在线控制器可能选择先释放 $C$ 的 $\Pi_2$；如果其他节点更紧张，则可能选择 $\Pi_1$ 或另外两个候选。离线阶段不需要知道未来具体的 request ID，而是通过大量场景上的完成请求数，判断哪些日程组合值得长期保留。

### 3.7 创新边界

本文不声称 group-based schedule representation 首次出现，也不声称交换树无法表示这些日程。交换树、递归搜索或动态规划都可以作为合法日程枚举工具。

本文声称的是：

> 将完整交换日程的候选构建定义为一个固定预算下的离线场景优化问题，并直接以离线场景中可完成的请求数选择 4 路径 $\times$ 4 日程候选库，而不是为每条路径独立保存若干局部指标最好的 tree/order。

---

-->

## 4. 创新点 2：在线批请求的交换日程联合选择

### 4.1 候选冲突图

在控制周期 $t$，全部候选为：

$$
\mathcal C_t
=
\bigcup_{r\in\mathcal R_t}
\left(
\{r\}\times\mathcal L_{s(r)d(r)}
\right),
$$

其中 $\mathcal L_{s(r)d(r)}$ 是根据请求端点从离线数据库读取并实例化的 16 个候选。在线阶段只更新候选的当前 EPR、memory、age、deadline 和跨请求冲突特征，不改变其路径与完整交换日程。

构造候选冲突图或异构图：

- 物理节点表示 repeater 和 memory 状态；
- 请求节点表示 deadline、等待时间和 QoS；
- 候选节点表示 path 与完整 schedule；
- 路径边描述候选经过哪些物理节点；
- 冲突边描述候选在时间和空间上的资源竞争；
- 兼容关系描述某个候选能够与哪些其他请求的候选共同执行；
- 依赖边描述 swap groups 的先后关系。

### 4.2 GNN + RL 决策

GNN 编码：

- 任意拓扑；
- 当前 memory/EPR 状态；
- 请求与候选之间的归属关系；
- 候选之间的共享节点和资源冲突；
- 候选的 memory-release profile；
- 候选与当前批次其他请求之间的兼容和冲突关系。

RL 在一个控制周期只提交一次组合动作：

$$
A_t\subseteq\mathcal C_t.
$$

这里 GNN + RL 学习的是“从离线数据库给定的候选中选择哪一组”，而不是自行生成新的 path 或 swap order。若希望 RL 逐步创造交换顺序，则动作必须改为逐组生成 $G_1,G_2,\ldots$；这属于另一套序列决策模型，不是本文当前采用的离线建库—在线选择结构。

约束包括：

$$
\sum_{c:r(c)=r}x_c\le 1,
\qquad
\forall r\in\mathcal R_t,
$$

以及当前可以确定的 EPR、memory、BSM 和结构冲突约束。

物理随机性由环境执行，不由 RL 伪装成确定性 mask。

### 4.3 与现有多用户方法的区别

QNR 已经能够为多个 S-D pair 选择 swapping trees，M-PSES 已经能够在多条路径之间提前释放资源。因此创新点 2 不能只写“多请求”或“共享内存”。

准确区别是：

> 在每个在线控制周期，根据当前 realized snapshot，从离线固定的 4 路径 $\times$ 4 日程目录中联合决定请求接纳、候选路径和完整交换日程，而不是为预先给定的唯一路径、树或优先级执行固定规则。

### 4.4 在线选择 MILP oracle

在一个 slot 的 planning snapshot 上，MILP 读取当前已到达且未过期的请求、每个请求缓存的完整“路径 + 交换日程”候选、节点 memory、当前库存 EPR、链路容量与生成概率、swap 概率以及物理时间参数。

模型输出两件事：选择哪些请求，以及每个请求采用哪个完整候选。模型内部还会安排候选的可行开始时刻和 EPR 来源，但这些只是可行性证书，不是控制器新增的动作；环境仍然只接收一次候选集合决策。

新 oracle 不再使用旧的“整槽静态资源总量”松弛。它在离散物理时间上直接检查：

- 当前库存 EPR 只能分配一次，未使用时继续占用两端 memory 和链路 buffer；
- 链路在各生成时刻能够提供多少新 EPR，由二项分布的单边可靠下界确定，而不是固定 seed 抽样；
- 新 EPR 从可靠到达时刻起就占用链路 buffer 和两端 memory；
- 完整 swap 日程决定每个内部节点何时释放两份 memory；
- 同一时刻的节点 memory、链路 buffer 和 BSM 均不能超容量。

优化顺序为：先最大化模型中可完成的请求数；请求数相同时，优先名义成功概率更高的候选；仍相同时，再减少 memory-time 和完成时间。因此它给出的是当前候选库在这套“可靠供给 + 时序资源”确定性模型下的严格最优解，而不是隐藏随机物理结果的事后最优解。

可靠度参数按物理链路分别解释。例如一条链路有 4 次自动尝试、单次成功率为 0.6 时，90% 可靠度下计入 1 份可靠 EPR，80% 可靠度下计入 2 份。该参数会写入实验结果 manifest；它不是所有链路同时成功的全局概率保证。

MILP 用于：

- 给出当前有限候选空间中的最优 baseline；
- 为 GNN + RL 提供训练标签或性能上界；
- 验证提升来自 schedule selection，而不是不同物理环境。

需要区分两个 MILP：

- **离线建库 MILP：**跨多个离线场景共享路径与日程保存变量，决定数据库中的 4 路径 $\times$ 4 日程；
- **在线选择 MILP oracle：**固定离线候选库，只对一个当前 snapshot 求解最大完成请求数，用于评估 GNN + RL。

真实随机执行结果仍由统一事件环境单独统计。MILP objective 应称为“可靠时序模型中的最优可完成请求数”，不能称为随机物理层的 clairvoyant optimum。旧的静态 MILP 仅保留为明确标注的 relaxation baseline；固定 planning seeds 并逐个调用物理执行器的 exact-scenario oracle 仅用于极小 snapshot。

---

## 5. 时间模型

### 5.1 文献依据

量子网络文献中存在两种常见执行模式。

**同步模式：**

- 一个 time slot 包含 elementary-entanglement generation phase；
- 随后进入 swapping/internal phase；
- Iñesta、Concurrent Entanglement Routing 和路由综述都采用或总结了这种模型；
- 一个 generation phase 可以聚合多次物理尝试形成整体链路成功概率。

**异步模式：**

- memory 可用时自动开始 elementary-entanglement generation；
- 纠缠 ready 且 swap 条件满足时立即执行；
- swap 失败后只重新生成受影响的纠缠；
- 其他链路生成和交换继续运行；
- 综述明确描述了该模式，Real-Time Ordering 也采用 generation、destruction 和时间事件触发。

本文的“提前释放 memory 后服务其他请求”需要异步模式。

### 5.2 控制周期不是物理生成周期

环境中的一个 step 定义为固定长度的 control epoch：

$$
[t_k,t_{k+1}),
\qquad
t_{k+1}-t_k=T_{\mathrm{ctrl}}.
$$

在 $t_k$：

1. 收集从上一个边界以来到达的请求；
2. 观察当前 EPR、memory、age 和事件状态；
3. 按每个请求的 $(s,d)$ 从离线数据库读取 16 个路径—日程候选；
4. 根据当前 snapshot 更新候选特征、可行 mask 和跨请求冲突关系；
5. GNN + RL 只决策一次；
6. 将所选日程安装到本地 trigger executor。

在 epoch 内，RL 不再决策，环境自动处理：

$$
\text{free memory}
\rightarrow
\text{generation attempt}
\rightarrow
\text{heralding}
\rightarrow
\text{EPR ready}
\rightarrow
\text{enabled swap group}
\rightarrow
\text{BSM/reset}.
$$

一次 control epoch 可以包含多个物理 generation、heralding 和 BSM 事件，但它们不是新的 RL step。

### 5.3 禁止“即时补充 EPR”

memory 被 swap 释放后，不会立即出现新的 EPR。正确过程是：

1. BSM 完成；
2. memory 经过 reset delay；
3. 链路层检测到 memory 可用；
4. 启动新的 generation attempt；
5. 经过传播和 heralding delay；
6. 以 $p_{uv}^{\mathrm{gen}}$ 概率得到新的 elementary EPR。

因此应写：

> 释放后的 memory 可以立即进入新的生成流程，但新的 EPR 只能在物理生成和 heralding 延迟之后概率性地建立。

### 5.4 swap failure

若 swap 失败：

- 输入纠缠被消耗；
- 仅相关 elementary/virtual links 重新进入自动生成流程；
- 已完成且未受影响的其他 swap groups 保留；
- 若本 epoch 仍有时间，固定日程可以继续等待和重试；
- epoch 结束后仍未完成的请求保留到下一周期重新规划。

RL 不负责发起重试。

### 5.5 MDP 语义

由于 control epoch 长度固定，一个 step 虽然聚合多个内部物理事件，仍可以建模为离散时间 MDP：

$$
S_{t+1}
\sim
\mathcal P(\cdot\mid S_t,A_t).
$$

如果未来改为“动作执行完成才结束 step”的可变时长模型，则应改称 semi-Markov decision process；第一阶段不采用该定义。

---

## 6. 状态、动作与奖励

### 6.1 状态

$S_t$ 至少包含：

- 网络拓扑、链路参数和节点 memory capacity；
- 每个 memory 的 free、occupied、reserved、resetting 状态；
- 当前 EPR 的端点、年龄、保真度和归属；
- 正在运行的 generation、heralding、BSM 和 reset 事件摘要；
- 全部已到达、未完成、未过期请求；
- 每个请求从离线库读取的 4 条候选路径；
- 每条路径的 4 个离线完整日程及其当前资源释放特征。

### 6.2 动作

动作是对当前候选目录的组合选择：

$$
x_c\in\{0,1\},
\qquad
c\in\mathcal C_t.
$$

每个请求最多选择一个候选。未选择的请求继续等待，只要没有超过 deadline。

策略内部可以用 masked autoregressive decoder 和 STOP token 构造集合，但整个集合只形成一次环境 action。

### 6.3 奖励

主奖励只使用本 epoch 新完成的唯一请求数：

$$
R_t=N_t^{\mathrm{complete}}.
$$

建议正式训练使用：

$$
\gamma=1,
$$

使目标与有限 horizon 内的总完成请求数一致。

以下作为解释指标，不与主目标混成任意加权和：

- hotspot memory occupancy time；
- memory-blocked generation time；
- network memory-time；
- request completion latency；
- deadline completion ratio；
- EPR utilization；
- fidelity 和物理成功率；
- 策略决策时间。

---

## 7. 实验设置

第一阶段统一环境：

- Waxman-like 随机拓扑；
- 20 个网络节点；
- 一个 episode 包含 30 个 control epochs；
- 每个 episode 恰好 100 个请求；
- 请求到达时刻来自在 30 个 epoch 上、给定总数为 100 的齐次泊松过程；
- 一个 epoch 的请求数量不设置固定 batch size；
- 每个 $(s,d)$ 离线保存 4 条候选路径；
- 每条路径离线保存 4 个完整交换日程；
- 每个请求在线最多读取 16 个路径—日程候选；
- 所有算法共享相同拓扑、请求流、物理参数和外生随机数。

候选库以固定拓扑为单位离线求解；同一拓扑上的所有在线 episodes 共用该候选库。离线建库场景与最终评估请求流必须分离：二者使用相同的请求分布和物理模型，但最终报告使用未参与候选库优化的新请求序列。

### 7.1 对比方法

至少包括：

1. Q-DDCA 路径选择 + 固定交换日程；
2. Q-CAST 路径选择 + 固定交换日程；
3. Sequential order；
4. Balanced/tree order；
5. Greedy Order；
6. swap-asap 或 simultaneous baseline；
7. M-PSES-like fixed-priority trigger baseline；
8. MILP Path-only；
9. MILP Path + Schedule；
10. GNN + RL。

对比时，路由算法只改变路径选择；物理生成、swap executor、memory、随机数和请求流必须完全相同。

### 7.2 创新点 1 的消融

- 每条路径保存单个局部最优 tree/order；
- 每条路径随机保存 4 个合法日程；
- 保存 sequential、balanced、left-first、right-first 四个固定模板；
- `canonical`：最短路径 + 低轮数日程；
- `quality`：路径质量 + 最低 memory-time；
- `banded`：质量带内 max-min release diversity；
- `pareto`：精确路径组合 + Pareto release coverage；
- `exact_kcenter`：Pareto 前沿的精确代表集；
- 每条路径只保留一个日程 vs. 多个完整日程；
- 固定同样 4 条路径，只改变每条路径保存的 4 个日程；
- 固定同样日程池，只比较不同 topology-only 生成器；
- 使用相同候选数量，避免把收益归因于候选更多。

需要回答：

> 在完全相同的 4 路径 $\times$ 4 日程预算下，Pareto/多样性生成器是否让同一个在线选择 MILP oracle 在未见请求 trace 上完成更多请求？

创新点 1 应单独报告：

- topology-only 路径质量、路径重叠和日程释放多样性；
- 未参与生成的新 topology seeds 和请求 traces 上的在线 oracle 完成请求数；
- 离线生成时间与在线 oracle 求解时间；
- 候选资源轨迹的重复率；
- 相对于完整日程池的候选库损失。

这组实验固定选择器为同一个 MILP，专门测量：

$$
\operatorname{OPT}(\Omega_t)-\operatorname{OPT}(\mathcal C_t)
$$

在可计算小规模完整空间的场景中，可以枚举 $\Omega_t$ 得到候选库损失；在较大场景中，只比较相同候选预算下不同离线建库方法的在线 MILP 结果。

### 7.3 创新点 2 的消融

- 独立为每个请求选择局部最佳日程；
- 固定路径优先级；
- greedy conflict resolution；
- 不使用候选冲突图；
- GNN 编码替换为扁平 MLP；
- MILP 最优组合。

需要回答：

> 联合选择是否优于把每个请求的局部最优日程简单拼接？

这组实验必须固定完全相同的离线候选库和当前候选集合 $\mathcal C_t$，比较 greedy、GNN + RL 和在线选择 MILP，专门测量候选空间内的选择能力，避免把离线建库差异误算成 RL 提升。

### 7.4 时间模型消融

- 异步生成与触发；
- 同步 generation-then-swap；
- 禁止本 epoch 内释放后的 memory 重新发起 generation；
- 增大 heralding/reset delay；
- 无限 memory；
- 不同 control epoch 长度。

若禁止 epoch 内资源复用后优势仍完全不变，则不能把提升解释为提前释放共享 memory。

---

## 8. 核心否证条件

出现以下情况时，应缩小或否定相应创新：

1. 在相同 4 路径 $\times$ 4 日程预算下，Pareto/多样性生成器与最短路径、固定模板结果相同；
2. 拓扑级生成器在未参与设计的 topology seeds 和请求 traces 上不能提高在线 oracle 完成请求数；
3. 四个日程的资源轨迹高度重复，删除其中多个候选后结果不变；
4. 独立局部选择与跨请求联合选择结果相同；
5. 优化路径优先级后的 M-PSES 与本文方法功能和结果等价；
6. 使用现实 generation、heralding、BSM 和 reset delay 后，提前释放的 memory 没有可利用时间；
7. 优势仅来自候选数量增加，而不是候选功能不同；
8. GNN + RL 无法逼近小规模 MILP oracle，或简单 greedy 始终等价。
9. 若在线阶段已经提供全部合法日程 $\Omega_t$，却仍把离线候选库构建声称为主要创新。

---

## 9. 最终贡献表述

### 创新点 1

> 提出一种不读取在线请求的拓扑级路径—完整交换日程候选库生成方法。该方法对每个节点对枚举有限路径池，以路径可靠性、资源瓶颈和路径重叠穷举选择 4 条互补路径；再枚举合法完整日程，从节点释放轮次、memory-time 和并行关系的 Pareto 前沿中选择最多 4 个互补日程，为在线控制缓存固定 16 槽候选网格。

### 创新点 2

> 提出一种面向在线批请求的 GNN + RL 联合控制方法。在任意拓扑和共享 memory 约束下，从离线固定的 4 路径 $\times$ 4 日程候选库中联合选择请求、路径及完整交换日程，以最大化有限 horizon 内完成的请求数，并使用同一候选空间上的在线选择 MILP 作为小规模最优 oracle。

### 时间模型

> 每个环境 step 是固定长度的 control epoch。控制器在边界只决策一次；epoch 内 elementary-entanglement generation、heralding、swap、failure recovery 和 memory reset 均由统一的异步事件环境自动执行。RL 不控制生成，也不存在释放后即时补充 EPR。

### 最终故事

> 交换树能够表示一条路径的完整交换顺序，但现有 tree/order 方法主要输出单条路径上的一个局部较优方案。本文在部署前仅根据固定拓扑生成一组质量受控且资源释放行为互补的 4 路径 $\times$ 4 日程候选库；真实请求到达后不在线生成日程、也不在线求解 LP，而由 GNN + RL 读取缓存并根据当前资源状态完成跨请求联合选择。在线 MILP 仅作为小规模候选库质量 oracle。
