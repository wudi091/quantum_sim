# Construction-Aware Quantum Routing — 实现方案

> 本方案是对 `idea-stage/construction_aware_routing_proposal.txt` 的**重定调 (reframe) + 可落地实现路径**。
> 直接依据来自 `ideaspark_run/construction-aware-routing-review/phase3_critique/phase3_critique_output.json`
> 与 `phase_3_failed.md`：原"GNN+Masked PPO 联合 (P,C)"候选被判 hard-floor abandon。

---

## 0. 为什么不能照原提案实现

评审给出的 abandon 触发点（必须逐条回避，否则任何实现都重蹈覆辙）：

| 触发 | 含义 | 对实现方案的硬约束 |
|---|---|---|
| **C13** (Encode Structure by Construction) | 组装 GIN+PPO+simulator+masked dispatch，只做 quantum relabel，无新机制 | **GNN+RL 不能作为核心贡献**；它只是可选求解器之一 |
| **C15** (Stacking auxiliary encoders) | 在"换表示"之外堆 action generator/score/tuning/controller | 核心机制必须**单一、可单独证伪**；不允许多模块捆绑出"提升" |
| **C10** (Characterize a Limit, Then Surpass It) | 声称 path-state 不可区分但从未给数值对/可判定准则/严格超越证明 | **必须产出**：具体数值的 aliased state 对、native path-only 等价类定义、严格超越定理、frontier 检查 |
| **C11** (Substituting one heuristic for another) | 换选择启发式而无"何时/为何胜过 Q-Stream"的定理 | 必须给**相对 baseline class 的可证明优势实例**，不是只跑赢实验 |
| **C02** (Synthetic-only ecological validity) | 诊断与策略都只在 simulator 上验证 | 必须有**真实拓扑/硬件参数/deployment trace** 三者至少其一 |
| 威胁论文 `arxiv:2302.02506v2` | GNN+RL 解 disjunctive-graph job-shop dispatch，机制级重叠 | 主结果**不得依赖**"evolving graph + GNN + RL dispatch"这一通用机制 |

**核心策略**：把贡献从"一个学到的策略"重构为"一个可证明的**construction-footprint 不可区分性结果 + 一个可判定的 baseline 超越实例**"。GNN/RL 退居为 scale 实验中的可选求解器，且与威胁论文的差异化由定理保证，不由文本保证。

---

## 1. 重定调后的核心主张（可答辩版）

> **主张 (C)**：在并发量子网络中，存在一条物理路径 P 与两个纠缠构造计划 C₁, C₂，使得任何仅以 P 为原子决策的路由策略族 𝔹_path 都把 (P,C₁) 与 (P,C₂) 映射到同一决策状态，而在该状态下的最优动作在 𝔹_path 内无差异，却对并发请求集合造成可量化的接受率/吞吐差异。因此，路由原子应为 (P,C) 而非 P。

这版主张刻意**非"首次"**、**非"maximally expressive"**（避开 C10 子条款），只声称"存在性 + 可判定 + 严格超越"。

---

## 2. 四阶段递进实现

### 阶段 A — 形式化与构造性反例（2–3 周，纯理论，无代码依赖）

**目标**：补齐 C10 要求的"具体数值对 + native 等价类 + 严格超越证明"。

#### A.1 native path-only 等价类

设时间离散为 slot t∈{1,…,T}。对一条路径 P=(v₀,…,vₙ)，一个 construction plan C 决定：

- 链路生成时间集合 `G(C) = {(i, t) : link (v_{i-1},v_i) generated at t}`
- swap 触发集合 `S(C) = {(k, t) : swap at v_k at t}`
- 由此导出 **memory occupancy matrix** `M(C) ∈ {0,1}^{|V|×T}`，`M[v,t]=1` 当且仅当 v 在 slot t 持有一个待用 EPR。

**定义 (path-only projection)**：`π_path(P,C) = P`。两个 plan C₁,C₂ 在 P 上 **path-indistinguishable** 当且仅当 `π_path(P,C₁)=π_path(P,C₂)`（trivially true for same P）。

**定义 (native baseline class 𝔹_path)**：一个策略 b∈𝔹_path 是"path-only"的，当且仅当其决策函数 `b: State_t → Action` 可分解为 `b = f ∘ π_path`，即动作只依赖路径，不依赖 C 的 footprint 结构。Q-CAST、Q-DDCA、real-time ordering 均属此类（需在 A.4 frontier 检查中逐一定义并验证）。

#### A.2 构造性反例（numeric pair）

直接用 `con_design.md` 第 3 节的 5 节点链 A-B-C-D-E，给出**确切数值 footprint**：

- `M(C_seq)`：C 仅在 [t_i, t_i+1] 局部占用每个中继（峰值 1）。
- `M(C_bal)`：C₁={AB,DE}@t₁, {BC,CD}@t₂, swaps@t₃ → B,C,D 在 [t₁,t₃] 全段被占（峰值 3）。
- 并发请求 R₂: X-C-Y 在 C 上需 1 个 slot：C_seq 下 R₂ 可调度，C_bal 下 R₂ timeout。

**交付物 A.2**：一张 5×4 的 0/1 占用矩阵表 + R₂ 的可行性布尔。这是 C10 所要求的"concrete numeric pair"，不是 prose。

#### A.3 严格超越定理（C11 回应）

**定理 (Strict Exceedance)**：令 `Acc(b, ℐ)` 表示策略 b 在请求实例集合 ℐ 下的接受率。存在 ℐ* 与 (P,C*) 使得：

```
max_{b ∈ 𝔹_path} Acc(b, ℐ*)  <  Acc(b*_{(P,C*)}, ℐ*)
```

其中 `b*_{(P,C*)}` 是 construction-aware 的平凡策略"在 P 上选 C*"。

证明骨架：以 A.2 的 ℐ* 为实例；𝔹_path 中任何 b 对 P 给单一动作，要么选 C_seq 要么 C_bal（或固定规则选其一），构造 ℐ* 使两种固定选择都失败（通过把 R₂ 的到达窗口设在 C_bal 的占用窗口内，同时让另一并发请求 R₃ 必须用 C_bal 的并行度才能在 TTL 内完成 —— 两难）。这是**纯组合论证**，不依赖任何学习。

#### A.4 Frontier 检查（回避威胁论文 + C10 子条款）

对每篇被 review 点名或同领域的论文，**显式判定其是否已编码 footprint 信息**：

- `arxiv:2302.02506v2` (GNN+RL job-shop)：其图边权是"操作-资源-依赖"，**不编码** "memory slot 的时序占用与并发请求的外部性"——这是定理 1 的差异化锚点。
- Q-CAST/Q-DDCA：决策是路径/宽度，**不编码** C。
- real-time ordering (Chang-Xue, Sundaram-Gupta, Mai et al.)：在**给定** C 的子结构（顺序/平衡）内选 swap order，**不跨请求**优化 footprint。
- Probabilistic cutoffs (Grimbergen et al.)：age-based 丢弃，不改变 C。

**交付物 A.4**：一张 frontier 表，每行 (paper, encodes_C?, encodes_cross_request_footprint?, subsumes_our_pair?)。这把"差异化"从散文变成可核对清单。

---

### 阶段 B — Footprint Oracle 与 baseline 形式化（2 周，Python）

**目标**：把 A 的理论实例变成**可执行、可复现的判定器**，使后续实验有一个 ground-truth 参照。

#### B.1 `construction_oracle.py`

输入：(topology, P, C, physical_params)。
输出：精确的 `M(C)`、peak memory、time-integrated occupancy、预期 fidelity（按 decay 模型）、对给定并发请求集 ℐ 的可行性向量。
实现：纯离散事件 + 解析公式，**无学习、无随机**，可作为 unit-testable ground truth。

依赖：复刻 SeQUeNCe 的物理模型接口（pg, ps, T₂, coherence time），但不调其事件引擎——oracle 是确定性的、可手算核对的。

#### B.2 `baseline_registry.py`

把每个 baseline 实现成 `b ∈ 𝔹_path` 的**纯函数决策器**（不是黑盒）：

- `qcast_decision(state) → (P, width)`
- `qddca_decision(state) → (P, width)`
- `realtime_ordering_decision(state, P) → C_within_path`（注意：此 baseline **能选 C**，但只在单请求内、不跨请求，用来精确划界"我们超越的是哪一类"）
- `fixed_C_template(i) → C_i`（sequential / balanced / hybrid，3 个手工模板）

这样所有 baseline 与待评估策略共享同一 `State`、同一 oracle、同一 metric——满足原提案"统一后端"诉求，且不依赖任何被威胁论文占用的 GNN+RL 机制。

#### B.3 单元测试即定理证据

为 A.2 的反例写 `test_alias_pair.py`：断言 `π_path(P,C_seq)=π_path(P,C_bal)=P` 且 oracle 计算的 R₂ 可行性不同。**测试通过 = A.3 的数值实例被机器验证**。这是 C10/C11 的可检查交付。

---

### 阶段 C — 实证评估：合成 + 真实 trace（3–4 周）

**目标**：回应 C02（不能只 synthetic）。

#### C.1 仿真后端

- **主**：SeQUeNCe（`qnet_core`），与原提案一致，保留所有算法共用同一后端的公平性控制。
- **校准锚点**：复现 `Realistic Simulation of Quantum Repeater...`（arxiv:2605.06928, QRE-CEC/SeQUeNCe 扩展）与 `Probabilistic Cutoffs...`（arxiv:2602.14738）的可复现参数集，使物理参数不是自选。

#### C.2 三层证据（满足 C02 的"ecological validity"）

1. **合成可控**（机制证据）：Waxman 200 节点，扫描 λ/memory/pg/ps。用于**证伪标准**：同 P 改 C 是否在所有配置下都不显著 → 若存在显著则主张成立。
2. **真实拓扑**（迁移证据）：至少 2 个公开拓扑——
   - SURFnet / ESnet science network graph（公开拓扑数据）
   - 拟议中量子测试床拓扑（Delft metropolitan / DOE AQT trace，用已发表论文的节点布局）
   只需节点图与边长，不需真实流量。
3. **deployment-trace 请求流**（外部有效性）：用已公开的量子实验 trace（或经典网络 trace 经量子化的到达模式，如 Poisson 从实测 inter-arrival 拟合）驱动请求到达。明确标注 trace 来源与转换。

#### C.3 待评估策略（注意：主结果**不是** GNN+RL）

按"是否避开威胁论文机制"排序的主结果：

1. **主结果：Oracle-Guided Construction Selection (OGCS)** —— 非学习、可解释。
   - 对每请求的 top-K 路径 × M 个 C 模板，用 B.1 oracle **精确**算 footprint；
   - 用一个**贪心/局搜**规则选择使"当前并发请求集合的预测接受率"最大的 (P,C)。
   - 这个策略**本身就是贡献的一部分**：它把 A.3 的存在性实例变成可大规模运行的算法，且机制上与 job-shop GNN+RL **完全不同**（确定性、oracle-driven、无 message passing）。
2. **次结果（消融用）：Joint (P,C) PPO** —— 仅用于回答"学习是否能进一步超过 OGCS"。
   - 此即原提案的 GNN+Masked PPO，但**论文中定位为 ablation**，novelty 不挂在它身上 → 规避 C13/C15。
   - 与 `arxiv:2302.02506v2` 的差异由 A.4 frontier 表 + 定理 1 保证，不由"GNN 用在量子域"保证。

#### C.4 对比矩阵

| 方法 | 决策 | 是 𝔹_path? | 编码 C? | 跨请求 footprint? | 角色 |
|---|---|---|---|---|---|
| Q-CAST | (P,width) | yes | no | no | 吞吐基线 |
| Q-DDCA | (P,width) | yes | no | no | 拥塞基线 |
| Real-time ordering | (P, C_within) | partial | yes(单请求) | no | 划界基线 |
| Fixed-C templates | (P, C_fixed) | no | yes(固定) | no | 手工下界 |
| **OGCS (主)** | (P, C_chosen) | no | yes | **yes** | 本工作主算法 |
| Joint PPO (消融) | (P, C_learned) | no | yes | yes | 学习是否更优 |

#### C.5 指标与证伪（沿用原提案，收紧）

- 主：pair throughput、acceptance rate。
- 次：timeout、peak/time-integrated memory、fidelity violation、Jain fairness。
- **证伪标准**（保留并强化）：
  1. (A.3 反例不成立) 在存储竞争下同 P 改 C 不造成网络级显著差异 → 主张被证伪。
  2. (OGCS 不超 baseline) 在相近 fidelity/latency 下 OGCS 不能在 throughput/acceptance 上严格超过 𝔹_path 最优 → 主张收缩为"存在性"而非"可工程化"。
  3. (PPO ≮ OGCS) 是可选结论，不影响主张。

统计：paired seeds (≥20)、bootstrap 95% CI、报告效应量。

---

### 阶段 D — 写作与可复现包（2 周）

**交付物**：
- `paper/main.pdf`：10 页正文 + 附录。附录含 A.2 数值表、A.4 frontier 表、定理证明、B 的 oracle 伪代码与测试输出。
- `artifact/`：可复现容器，含 oracle、baselines、OGCS、SeQUeNCe 配置、3 个真实拓扑数据、trace、随机种子、`make results` 一键跑出主表。
- `artifact/test_alias_pair.py` 通过 = A 节定理被机器验证。

---

## 3. 仓库结构（建议）

```
quantum_sim/
├─ con_design.md                  # 原构思
├─ idea-stage/                    # 原提案 + 评审产物（只读，存档）
├─ IMPLEMENTATION_PLAN.md         # 本文件
├─ src/
│  ├─ core/
│  │  ├─ plan.py                  # Construction Plan 数据结构 (B1,B2,Tswap)
│  │  ├─ footprint.py             # M(C) 计算
│  │  └─ state.py                 # native State, π_path projection
│  ├─ oracle/
│  │  └─ construction_oracle.py   # B.1 确定性 oracle
│  ├─ baselines/
│  │  ├─ qcast.py
│  │  ├─ qddca.py
│  │  ├─ realtime_ordering.py
│  │  └─ fixed_templates.py
│  ├─ methods/
│  │  ├─ ogcs.py                  # 主算法 (C.3.1)
│  │  └─ joint_ppo.py             # 消融 (C.3.2)
│  ├─ sim/
│  │  ├─ sequenc_backend.py
│  │  ├─ topologies/{surfnet.gml, esnet.gml, delft_metro.gml}
│  │  └─ traces/
│  └─ tests/
│     ├─ test_alias_pair.py       # A.2 反例机器验证 (C10 证据)
│     ├─ test_oracle.py           # oracle vs 手算
│     └─ test_frontier.py         # A.4 表自动核对
└─ paper/
   ├─ main.tex
   ├─ appendix_proof.tex
   └─ figures/
```

---

## 4. 里程碑与可检查门

| 周 | 里程碑 | 完成的判据（可被第三方核对） |
|---|---|---|
| 1 | A.1–A.2 形式化 + 反例数值表 | 5×4 占用矩阵表落地；π_path 定义文字化 |
| 2 | A.3 定理 + A.4 frontier 表 | 定理证明草稿；frontier 表 ≥6 篇含 2302.02506 |
| 3 | B.1 oracle + B.3 test_alias_pair | `pytest test_alias_pair.py` 绿 |
| 4 | B.2 baseline registry | 5 个 baseline 决策器 unit test 通过 |
| 5–6 | C.1 仿真后端 + C.2 合成层 | 在 Waxman 上复现 Q-CAST 数值（sanity） |
| 7 | C.2 真实拓扑接入 | SURFnet/ESnet 拓扑加载 + 物理参数取自 QRE-CEC 复现集 |
| 8 | C.3.1 OGCS 实现 | OGCS 在合成层超过 Fixed-C 与 Real-time ordering |
| 9 | C.4 主对比表 + 证伪检验 | 20 seeds CI；证伪标准 1&2 给出明确结论 |
| 10 | C.3.2 PPO 消融 + D 写作 | PPO 是否 >OGCS 给出结论；论文初稿 |

每个门的"判据"都是**机器或第三方可核对**的，而不是"感觉做完了"——这是为了避免原评审指出的"asserted rather than supplied"。

---

## 5. 风险与 reject-lesson 对应表

| 风险 | 对应 reject lesson | 缓解 |
|---|---|---|
| 又被读成"GNN+RL 拼装" | C13/C15 | 主算法 OGCS 无学习；PPO 明确降为消融；novelty 挂在定理 1 + oracle |
| "path-state 不可区分"仍是散文 | C10 | A.2 数值表 + `test_alias_pair.py` 机器验证 |
| "只是换了启发式" | C11 | A.3 严格超越定理 + baseline class 形式化定义 |
| 只 synthetic | C02 | C.2 三层证据，含真实拓扑 + deployment trace |
| 与 job-shop GNN+RL 重叠 | 2302.02506 | A.4 frontier 表 + 主算法不含 message passing；定理 1 的差异化锚点是"跨请求时序外部性"，job-shop 论文不编码 |
| 评审要求"独立硬件证据" | C02 末句 | 若 review 仍要求真实硬件：作为 future work 显式列出，并指出 oracle 的确定性使其可被真实实验直接核对（不需运行策略） |

---

## 6. 与原提案 (`construction_aware_routing_proposal.txt`) 的差异清单

1. **核心贡献迁移**：原"GNN+Masked PPO 联合 (P,C)" → 现"不可区分性定理 + oracle + OGCS"，PPO 降为消融。
2. **新增定理层**：A.1/A.3 的 π_path 等价类与严格超越定理，原提案无。
3. **新增 frontier 检查**：A.4，原提案无（这是 review 直接点名缺失项）。
4. **新增真实拓扑/trace**：C.2，原提案只有 Waxman 合成。
5. **主算法替换**：OGCS 取代"GNN+PPO"作为主结果。
6. **保留**：SeQUeNCe 后端、统一后端公平性、paired seeds、证伪标准、C 模板（sequential/balanced/hybrid）、10 周节奏。

---

## 7. 下一步立即可做（无需等待）

最小可执行第一步：在 `src/core/plan.py` 与 `src/core/footprint.py` 里实现 `M(C)` 计算，并为 `con_design.md` 第 3 节的 C_seq / C_bal 写出两个硬编码 footprint，然后 `test_alias_pair.py` 断言它们 π_path 相同、oracle 输出不同。这一步**纯 Python、无依赖、半天内可完成**，且其通过即构成主张 (C) 的第一个可机器核对的证据。
