# Superseded fixed-path pilot

This report records the first fixed-candidate-path pilot. The environment now
uses dynamic resource-graph paths. Use `RESOURCE_GRAPH_TRAINING_REPORT.md` for
the current result.

## Setup

- Physical backend: unmodified RELiQ `QuantumNetwork`, `Edge.links`, and
  `QuantumLink.swap`.
- Four persistent requests, 2--5 hops.
- Concrete EPR tokens, fidelity decay/generation, and swap failures enabled.
- Balanced swap DAG with one RELiQ subslot per layer.
- Pure masked PPO: 100 updates, 256 decisions/update, no optimizer labels.
- Training topology seed: 0.
- Evaluation: two unseen topology seeds (`20000` and `30000`), 20 episode seeds
  per topology.

Artifacts are under `batchswap_rl/runs/reliq_short_convergence/`.

## Learning effect across the two unseen topologies

| Checkpoint | Completion | Mean completed delay | P95 completed delay | Makespan | Return |
|---|---:|---:|---:|---:|---:|
| PPO update 1 | 88.13% | 8.504 | 54.00 | 59.78 | 10.661 |
| PPO update 100 | **91.25%** | **6.514** | **17.25** | **40.38** | **13.460** |

Training improved completion by 3.12 percentage points, reduced the mean delay
of completed requests by 23.4%, reduced P95 by 68.1%, and reduced makespan by
32.5%.

## Trained PPO versus non-learning policies

| Policy | Completion | Mean completed delay | P95 completed delay | Makespan | Return |
|---|---:|---:|---:|---:|---:|
| PPO | **91.25%** | 6.514 | 17.25 | **40.38** | **13.460** |
| Q-DDCA-style | 89.38% | **5.720** | 14.90 | 45.28 | 12.499 |
| Greedy | 86.88% | 8.633 | 57.40 | 58.18 | 10.423 |
| Random valid | 90.63% | 7.531 | **11.00** | 42.50 | 13.003 |

PPO currently has the best completion rate, makespan, and training objective.
Q-DDCA-style has lower delay among the requests it completes, but completes
fewer requests and has a longer makespan. Random's low P95 must not be read as
overall superiority because incomplete requests are absent from its delay
sample.

This pilot proves that the centralized plan selector learns under RELiQ's
concrete-resource and fidelity dynamics. It does not yet prove statistically
significant superiority over Q-DDCA-style.

## Important limitations

- Only the short 2--5-hop curriculum was trained.
- Candidate routes are fixed shortest paths, not dynamically extracted from the
  full current resource graph.
- Each evaluation group uses one cached unseen topology with varied request and
  physical RNG seeds.
- RELiQ itself is a time-stepped physical abstraction, not hardware execution.
