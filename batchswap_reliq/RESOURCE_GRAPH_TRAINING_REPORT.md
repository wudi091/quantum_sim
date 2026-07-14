# Superseded resource-graph DFS pilot

This report records the intermediate direct-resource-graph path enumeration
design. The final design uses physical K-shortest paths followed by EPR-prefix
validation. See `K_SHORTEST_PREFIX_REPORT.md`.

## Decision pipeline

```text
RELiQ Edge.links token snapshot
  -> current EPR resource graph
  -> candidate resource paths per request
  -> token-level balanced swap plans
  -> masked PPO selects a conflict-free plan set
  -> STOP atomically executes the selected swap DAGs
```

The request state stores a physical frontier node. It is not restricted to its
canonical topology-shortest path. The canonical path is used only to construct
workloads and report initial hop distance.

## Training

- Four persistent requests, 2--5 topology hops.
- RELiQ topology degree 3, four EPR storage slots per physical edge.
- Maximum resource-path extension: eight EPR edges.
- Up to 128 simple resource paths explored per request and three diverse paths
  compiled into swap plans.
- Pure masked PPO, 100 updates, 256 decisions/update.
- Training topology seed 0.

Training-window means show a consistent learning signal:

| Updates | Loss | Value loss | Mean reward | Mean delay | Makespan | Failed plans |
|---|---:|---:|---:|---:|---:|---:|
| 1--20 | 10.584 | 21.186 | 0.790 | 6.618 | 22.063 | 1.165 |
| 21--40 | 3.697 | 7.410 | 0.978 | 6.070 | 18.949 | 0.800 |
| 41--60 | 1.571 | 3.159 | 1.088 | 4.829 | 16.534 | 0.561 |
| 61--80 | 1.319 | 2.655 | 1.272 | 4.301 | 12.613 | 0.423 |
| 81--100 | 0.783 | 1.589 | 1.564 | 3.939 | 12.521 | 0.213 |

## Evaluation on unseen topology seeds

Two unseen topologies (`20000`, `30000`) were used, with 20 episode seeds on
each topology.

### Learning effect

| Checkpoint | Completion | Mean completed delay | P95 | Makespan | Return |
|---|---:|---:|---:|---:|---:|
| PPO update 1 | 92.50% | 7.419 | 18.95 | 38.88 | 13.410 |
| PPO update 100 | **96.25%** | **4.857** | **18.00** | **25.73** | **17.071** |

Training raised completion by 3.75 percentage points, reduced completed-request
mean delay by 34.5%, and reduced makespan by 33.8%.

### Policy comparison

| Policy | Completion | Mean completed delay | P95 | Makespan | Return |
|---|---:|---:|---:|---:|---:|
| Dynamic-resource PPO | **96.25%** | 4.857 | 18.00 | **25.73** | **17.071** |
| Q-DDCA-style | 93.75% | **4.593** | **14.10** | 32.50 | 15.042 |
| Greedy | 93.75% | 6.040 | 22.65 | 33.60 | 14.228 |
| Random valid | 95.63% | 7.477 | 15.80 | 30.33 | 15.047 |

PPO has the best completion, makespan, and objective return. Q-DDCA-style still
has slightly lower delay among the requests it completes, so this pilot does not
establish across-the-board superiority.

## Artifacts

- `batchswap_rl/runs/reliq_resource_graph_convergence/history.json`
- `batchswap_rl/runs/reliq_resource_graph_convergence/checkpoint.pt`
- `eval_u100_t20000.json`
- `eval_u100_t30000.json`

## Remaining work

- Train the 5--15 and 20--50-hop curriculum stages.
- Increase path-enumeration scalability beyond bounded DFS.
- Evaluate more topology seeds and report confidence intervals.
- Add a stronger Q-DDCA local-congestion implementation while keeping the same
  RELiQ physical backend.
