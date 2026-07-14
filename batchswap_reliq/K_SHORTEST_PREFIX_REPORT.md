# K-shortest topology paths with resource-prefix validation

## Final decision pipeline

```text
current frontier + destination
  -> N shortest simple paths in physical topology
  -> scan each path against current concrete EPR inventory
  -> stop at first unavailable edge
  -> bind the farthest feasible prefix to EPR tokens
  -> compile its balanced swap DAG
  -> masked PPO selects one plan per request, then STOP executes the batch
```

This is deliberately simpler than searching directly in the resource graph.
The physical topology proposes routes; the quantum resource graph determines
how far each route can currently run.

## Training configuration

- RELiQ `QuantumNetwork`, `Edge.links`, fidelity/decay and `QuantumLink.swap`.
- Four persistent requests, 2--5 hops, topology degree 3.
- Eight shortest physical paths generated per request.
- Each path validated against the current EPR token snapshot.
- 100 masked-PPO updates, 256 decisions/update.
- No MILP/Oracle labels or supervised data.

| Updates | Loss | Value loss | Mean reward | Mean delay | Makespan | Failed plans |
|---|---:|---:|---:|---:|---:|---:|
| 1--20 | 6.025 | 12.064 | 0.561 | 6.723 | 31.347 | 1.696 |
| 21--40 | 3.784 | 7.580 | 0.835 | 5.448 | 20.478 | 0.984 |
| 41--60 | 2.843 | 5.696 | 1.064 | 4.690 | 19.603 | 1.153 |
| 61--80 | 1.675 | 3.365 | 1.112 | 4.687 | 16.862 | 0.911 |
| 81--100 | 0.856 | 1.724 | 1.339 | 4.446 | 13.774 | 0.279 |

The critic loss and failure rate decrease substantially while reward increases.
The delay curve is noisy but improves from 6.723 to 4.446 between the first and
last 20-update windows.

## Unseen-topology evaluation

Two topology seeds (`20000`, `30000`) and 20 episode seeds per topology were
used. Metrics pool all completed-request delays; completion and makespan include
truncated episodes.

| Policy | Completion | Mean completed delay | P95 | Makespan | Return |
|---|---:|---:|---:|---:|---:|
| PPO | **95.00%** | 5.533 | 20.45 | **30.53** | **16.331** |
| Q-DDCA-style | 94.38% | **5.212** | **16.00** | 32.55 | 14.919 |
| Greedy | 90.63% | 7.614 | 25.40 | 44.63 | 11.865 |
| Random valid | 89.38% | 6.322 | 21.40 | 49.45 | 11.520 |

PPO obtains the best completion, makespan, and reward. Q-DDCA-style remains
slightly better on the conditional delay/P95 of requests that complete. The
result is a feasibility and training validation, not yet a significance claim:
only two topology seeds and the short-hop curriculum were used.

Artifacts:

- `batchswap_rl/runs/reliq_kshortest_prefix_convergence/history.json`
- `batchswap_rl/runs/reliq_kshortest_prefix_convergence/checkpoint.pt`
- `eval_u100_t20000_final.json`
- `eval_u100_t30000_final.json`
