# Q-CAST 论文复现清单

本文档只记录论文 **Concurrent Entanglement Routing for Quantum Networks: Model and Designs** 中可以核实的定义、算法和实验条件，不代表当前仓库已经实现或复现这些细节。

## 1. 文献来源

- 作者：Shouqian Shi, Chen Qian
- 会议：ACM SIGCOMM 2020，pp. 62–75
- DOI：`10.1145/3387514.3405853`
- 论文中的源码链接：`https://github.com/QianLabUCSC/QuantumRouting`
- Zotero item key：`95ZRR2FI`
- Zotero PDF attachment key：`UC9PZ4NH`
- 核对日期：2026-07-17

以下简称论文算法为 **Q-CAST**（Contention-free pAth Selection at runTime）。

原文定位：网络/时隙模型见 Sec. 3.1–3.4（pp. 64–66）；wide path 与 EXT 见 Sec. 4.1（pp. 66–67）；Q-CAST G-EDA 与 recovery 见 Sec. 4.3.1–4.3.2（pp. 69–71）；P4 xor 见 Sec. 4.3.3（p. 71）；实验设置与结果见 Sec. 5.1–5.3（pp. 71–73）；复杂度补充见 Appendix A.2（p. 75）。

## 2. 论文问题定义与一轮时序

网络是多重图 `G = <V,E,C>`：节点有有限量子存储 `Q_u`；一条 edge 可包含多个并行 quantum channels，其数量是 edge width。一个 channel 两端各绑定一个 qubit 后才是 bound channel。一次绑定在 P2 中以概率 `p_c` 建立一个 quantum link。

每个 time slot 有四阶段：

1. **P1**：所有节点通过经典网络获得本轮全部并发 S-D pairs。
2. **P2 / external phase**：基于稳定、全局已知的 topology 选择路径，并独占分配 qubits/channels；随后各 channel 尝试建立局部纠缠。
3. **P3**：交换本轮实际 link states；每个节点只获得 `k`-hop 范围内的 link state。
4. **P4 / internal phase**：节点只依据 P2 的路径和本地 `k`-hop link states 作 swapping；每次 swap 以概率 `q` 成功。

目标是最大化每个 time slot 内所有 S-D pairs 成功建立的端到端 ebits 总数。论文 throughput 单位是 **ebits per time slot (eps)**。同一 S-D pair 在一轮可得到多个 ebits。

资源约束是严格独占的：一个 qubit 不能绑定多个 channel，一个 channel 的同一端不能绑定多个 qubit，一个成功 link 不能参加多个 swapping。论文每个 time slot 结束后重置整个网络，不保留未使用的纠缠资源；论文将跨 slot 复用列为未来工作。

## 3. 宽路径与 EXT

路径记为 `<(v_0,...,v_h),W>`，即 `(W,h)`-path：有 `h` hops，且每一 hop 至少保留 `W` 条并行 channels。Q-CAST 不是只选一条“线”，而是以完整 width 同时预留每一 hop 的 channel/qubit 资源。

为保证宽路径内的 swapping 一致，每条 channel 有全局唯一 ID。P4 中，一个节点把通向前驱的最小 ID 成功 link 与通向后继的最小 ID 成功 link 配对，重复直到不能再配对。

对第 `k` hop，设单 channel 成功率为 `p_k`，该 hop 的 `W` 条 channel 恰有 `i` 条成功的概率为：

```text
Q_k^i = C(W,i) * p_k^i * (1-p_k)^(W-i)
```

设前 `k` hops 每一 hop 都至少有 `i` 条成功 link 的概率为 `P_k^i`：

```text
P_1^i = Q_1^i
P_k^i = P_(k-1)^i * sum(l=i..W) Q_k^l
        + Q_k^i * sum(l=i+1..W) P_(k-1)^l
```

论文的 expected throughput（EXT）为：

```text
E_t = q^h * sum(i=1..W) i * P_h^i
```

注意：论文明确使用 `q^h`，不是 `q^(h-1)`。复现时应先忠实遵循论文公式，再单独讨论物理建模约定。

EXT 是非加性、随路径扩展单调不增的 metric。论文据此提出 Extended Dijkstra Algorithm（EDA）。

## 4. Q-CAST P2：G-EDA major-path 分配

Q-CAST 没有 Q-PASS 的 offline candidate-path 阶段；每个 time slot 针对当轮 S-D pairs 在线计算 contention-free paths。

G-EDA 的精确外层流程：

1. 在当前 residual graph 上，对**每个** S-D pair 用 EDA 找 EXT 最大的可行路径。
2. 在这些“每对当前最优路径”中，选择 EXT 最大的一条，按完整 width 独占预留该路径上的 qubits/channels。
3. 从 residual graph 删除已预留资源。
4. 重复 1–3，直到找不到路径，或已选路径数达到/超过论文的计算上限 `200`。

该过程只优化总 throughput，不显式优化公平性。论文附录明确说明 G-EDA 是 greedy heuristic，不保证全局最优。

### 4.1 EDA 状态与停止条件

EDA 类似“最大评价值优先”的 Dijkstra：对每个节点维护当前最佳完整路径评价 `E[node]`、前驱和当前 bottleneck width；优先弹出 `E` 最大节点。扩展一条边时重新以完整路径和其 width 计算 EXT。destination 首次出队/访问时返回该路径。

### 4.2 major-path 最大 hop 限制

论文用 `h_m` 限制 EDA 搜索：

1. 新 topology 初始化时随机抽取 100 对节点；
2. 对每一对用 `h_m = infinity` 的 G-EDA 做 multipath routing；
3. 在所有 `E_t >= 1` 的输出路径中取最大 hopcount，作为之后的 `h_m`。

若没有 `E_t >= 1` 的路径时如何回退，论文未说明。

## 5. Q-CAST P2：recovery paths

major paths 选完后，Q-CAST只用 residual qubits/channels 构造 recovery paths，因此 major/recovery 以及不同 major paths 之间都没有资源冲突。

一条 recovery path 的两个端点必须都是同一 major path 上的 **switch nodes**，且沿 major path 的间距不超过 `k` hops；这是为了让 P4 的相关节点在 `k`-hop link-state 视野内作一致决定。

论文给出的搜索顺序：

1. 对 major path 上每个节点 `x`，先令 `y` 为其沿 major path 前方 1 hop 的节点，在 residual graph 中用 EDA 找至多 `R` 条 `x -> y` recovery paths；每找到一条后继续更新 residual resources。
2. 对所有节点处理完 1-hop 覆盖后，再依次处理覆盖 `l = 2,3,...,k` 个 major-path hops 的 switch-node 对。
3. 最后所有节点按预留的 major/recovery paths 分配 qubits/channels。

`R` 只被描述为 “a small constant parameter”，**论文没有给出具体值**。

## 6. Q-CAST P4：xor recovery

对一条 width 为 `W` 的 major path，P4 将它视为 `W` 条分离的 width-1 paths，逐条恢复。对其中一条：

- `E` 是该 major 1-path 上实际成功的 edges 集合；
- 每条 recovery path 与它所覆盖的 major-path segment 构成一个 recovery loop，其 edge 集合记为 `E_pj`；
- 集合 xor 定义为对称差：`E1 xor E2 = (E1 union E2) - (E1 intersection E2)`；
- 选择若干 recovery loops，使图 `<V, E xor E_p1 xor ... xor E_pK>` 中该 S-D pair 连通；
- 有并列选择时优先较短 recovery paths，因为 swap 后成功率更高。

这不是“major path 失败就整体切换到一条备份 path”。多个 recovery loops 可以组合，某条 major edge 在 xor 中出现奇数次就保留、偶数次就抵消；论文 Fig. 10 甚至允许最终连接中一条 edge 以反向次序使用。

论文 Fig. 9 还明确说明：recovery-path 内部节点即使该 recovery 最终未被 switch nodes 采用，也先沿 recovery path 做 swapping；switch node 再决定保持 major pairing 或切到 recovery pairing。

### 6.1 P4 尚未完全规格化的部分

论文没有给出“从所有 recovery loops 中寻找满足连通性的子集”的完整伪代码、复杂度、搜索顺序或 tie-break 全序。论文结论也把“find an efficient algorithm to correctly select the recovery loops for Q-CAST P4”列为未来工作。因此，仅凭论文无法 bit-for-bit 复现 P4 子集选择；实现必须明确记录额外约定，不能伪称为论文参数。

## 7. Waxman topology 与物理资源

论文仿真使用随机 topology，不使用规则 grid/ring：

- 部署区域：`100K x 100K` units 的正方形，文中称每 unit 可视为 1 km。
- 输入：节点数 `n`、平均邻居数 `E_d`、所有 channels 的平均成功率 `E_p`。
- 节点随机放置。
- edges 由 Waxman model 生成。
- topology 生成后，对 channel-distance 成功率公式中的 `alpha` 做 binary search，使实际平均 channel success rate 落在 `E_p +/- 0.01`。
- 单 channel 成功率模型：`p_c = exp(-alpha * L)`，`L` 为物理长度。
- 每节点 qubit capacity：独立均匀整数采样 `Q_u in [10,14]`。
- 每 edge width：独立均匀整数采样 `W_e in [3,7]`。

### 7.1 Waxman 未知/歧义项

- 论文只说使用 Waxman model 并以 `E_d` 控制平均 degree，**没有写出 Waxman edge probability 公式，也没有给 Waxman 常见的 alpha/beta 参数值或求解方式**。
- 文中同时用 `alpha` 表示物理 channel success `exp(-alpha L)` 的衰减系数；不能把它自动等同于 Waxman 公式中的同名参数。
- 论文 PDF 的节点最小间距公式文本提取为 “at least <= 50/sqrt(n) units”，符号自相矛盾；可确认量级文本为 `50/sqrt(n)`，但严格不等号/`K` 单位需由官方源码或 PDF 公式图进一步核实。
- topology 是否在不连通时重采样、随机坐标/edge/资源的 RNG 与 seed、S-D pair 的抽样是否排除邻接节点，论文均未说明。

## 8. 论文实验参数矩阵

每个数据点是 **10 个不同随机 network topologies** 的均值；每个 topology 仿真 **1000 个相互独立 time slots**。

| 参数 | 论文取值 | reference setting |
|---|---:|---:|
| 节点数 `n` | `{50,100,200,400,800}` | `100` |
| 平均 channel success `E_p` | 方法段写 `{0.6,0.3,0.1}` | `0.6` |
| swap success `q` | 方法段写 `{0.8,0.9,1.0}` | `0.9` |
| link-state range `k` | `{0,3,6,infinity}` | `3` |
| 平均 degree `E_d` | `{3,4,6}` | `6` |
| 并发 S-D pairs `m` | `1` 到 `10` | `10` |
| node qubits `Q_u` | iid uniform integer `[10,14]` | 同左 |
| edge width `W_e` | iid uniform integer `[3,7]` | 同左 |
| G-EDA path cap `K_m` | `200` | `200` |
| recovery cap `R` | **未知** | **未知** |
| time slots / topology | `1000` | `1000` |
| topologies / data point | `10` | `10` |

论文图与方法段存在两处需要在复现报告中显式说明的不一致：

- Fig. 17 横轴画出了 `E_p = 0.1,0.2,...,0.9`，多于方法段列举的 `{0.6,0.3,0.1}`。
- Fig. 18 横轴画出了 `q = 0.80,0.85,0.90,0.95,1.00`，多于方法段列举的 `{0.8,0.9,1.0}`。

论文未给每张图所用的随机 seeds，也未给图中曲线的原始数值表。

## 9. 对比算法、消融与指标

论文主要比较：

- Q-CAST；
- Q-PASS（后续主要用其 CR metric 版本）；
- Greedy routing；
- SLMP（single-link multipath）；
- reference-setting CDF 还分别展示 Q-PASS 的 BotCap、CR、SumDist、MultiMetric；
- recovery 消融：`Q-CAST\\R` 与 `Q-PASS\\R`，即禁用 recovery paths。

报告的量包括：总 throughput（eps）、成功 S-D pairs 数、每 S-D pair 分配 major paths 的总 width、recovery 对 throughput 的贡献、occupied channels、recovery-path width CDF、每条 major path 的 recovery-path 数量 CDF。

## 10. 必须复现的论文趋势

以下是论文文字直接陈述的趋势，可作为“趋势复现”的预注册验收项；图中没有原始数组时不应伪造精确数值：

1. reference setting 下，Q-CAST throughput CDF 优于所有基线；Q-PASS/CR 比 Greedy 约高 `2 eps`，Q-CAST 又比 CR 约高 `5 eps`；Q-CAST 很少低于 `5 eps`。
2. `k=3` 对 Q-CAST 已足够；更大的 `k` 反而略降，因为会选择更长、更不可靠且占资源的 recovery paths。
3. 降低 `E_p` 或 `q` 均降低 throughput；论文称 Q-CAST 在这些设备能力变化下仍最好。swap failure 不能像 P2 link failure 一样由 recovery 绕过。
4. 网络规模增大时，各算法 throughput 下降；论文称 Q-CAST 在 `n=800` 时仍约 `7.5 eps`，且所有规模上最好。
5. 并发请求数增大时 throughput 次线性增长；Q-CAST 相对其他算法的优势随并发数快速扩大。
6. Q-CAST 的成功 S-D pairs 数最高；所有算法随请求数次线性增长。
7. recovery 消融中，recovery 约给 Q-PASS 增加 `0.5 eps`、给 Q-CAST 增加 `1 eps`。
8. `Q-CAST\\R` 相对 Q-CAST 节省约 `25%` channels；论文文字以 Q-CAST 约占 `400 channels` 为参照。
9. S-D pairs 较少时 recovery paths 更宽；论文称单请求相对 10 并发请求多数情形约宽 `2`。
10. 网络越大，major paths 越长，单条 major path 能找到的 recovery paths 越多。

## 11. 复现判定边界

论文级 Q-CAST 复现至少必须同时具备：

- 多宽度 `(W,h)` path 和论文 EXT；
- G-EDA 对所有并发 S-D pairs 的逐轮竞争与 residual-resource 更新；
- node qubit 与 edge-channel 双重独占约束；
- major 后在 residual graph 中按 `l=1..k` 搜 recovery paths；
- P3 后使用实际成功 links；
- P4 以 recovery-loop symmetric difference/xor 判断组合连通，而不是简单 shortest detour；
- 每 slot 重置；
- 论文 Waxman/reference 参数与 10 topologies x 1000 slots 的统计规模。

下列内容不能从论文唯一确定，必须在复现报告中列为 implementation choices：Waxman 的完整生成参数、`R`、P4 recovery-loop 子集搜索细节、所有 RNG seeds、节点最小间距公式的精确符号、无 `E_t>=1` 路径时 `h_m` 的回退规则。
