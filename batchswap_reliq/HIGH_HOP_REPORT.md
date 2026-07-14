# 20--50-hop RELiQ physical-backend result

## Environment

- 110 RELiQ repeaters arranged as two 55-node rails.
- Sparse rungs provide route diversity while preserving a 55-hop diameter.
- Ten persistent requests with shortest-path lengths uniformly sampled in the
  20--50-hop range.
- Four concrete EPR storage slots per physical edge.
- Eight shortest simple topology paths generated per request; each is truncated
  to its farthest currently EPR-feasible prefix and compiled into a swap DAG.
- Empty `STOP` is masked while any request can advance, preventing deliberate
  request starvation.

## Curriculum effect

The short-hop checkpoint was evaluated zero-shot on five high-hop seeds, then
used to initialize 50 high-hop PPO updates.

| Model | Completion | Mean delay | P95 | Makespan |
|---|---:|---:|---:|---:|
| Short-hop checkpoint, zero-shot | 100% | 28.84 | 54.55 | 54.20 |
| High-hop fine-tuned checkpoint | 100% | 17.10 | 37.20 | 34.80 |

Fine-tuning reduced mean delay by 40.7% and makespan by 35.8% on these paired
five seeds. High-hop curriculum training is therefore necessary; short-hop
weights alone do not choose long swap plans effectively.

## Final 20-seed comparison

| Policy | Completion | Mean delay | P95 | Makespan | Return |
|---|---:|---:|---:|---:|---:|
| PPO | 100% | 18.705 | 39.00 | 37.40 | 48.186 |
| Q-DDCA-style | 100% | **18.325** | 39.00 | **36.50** | **48.358** |
| Greedy | 100% | 22.945 | 44.05 | 41.45 | 46.566 |
| Random valid | 100% | 22.915 | 46.00 | 43.70 | 46.466 |

PPO is substantially better than greedy and random, but remains slightly worse
than Q-DDCA-style: +2.1% mean delay and +2.5% makespan. This is not yet the
desired high-hop advantage.

## Training convergence

Across five ten-update windows, value loss fell from 11.182 to 1.255. However,
mean delay remained approximately 19 hops/subslots after the initial transfer,
so the high-hop policy has reached a plateau rather than continuing to improve.

## Interpretation

The method now executes and trains correctly at 20--50 hops. The remaining
research problem is no longer basic feasibility; it is finding a scenario and
policy signal where global batch plan selection beats the strong shortest-prefix
FIFO controller. More training with the current objective alone is unlikely to
create a large gap.

Artifacts:

- `batchswap_rl/runs/reliq_long_ladder_finetune/checkpoint.pt`
- `batchswap_rl/runs/reliq_long_ladder_finetune/history.json`
- `batchswap_rl/runs/reliq_long_ladder_finetune/eval_long_20seeds.json`

## Fixed request-lifetime stress test

The environment now supports a fixed per-request TTL measured in RELiQ
physical subslots.  It is the same for every request and does not depend on
path length.  A request completing exactly at `arrival + TTL` succeeds;
otherwise it becomes an exogenous timeout failure.  `max_subslots` remains
only an episode safety cap.

The existing checkpoint (trained without TTL) was evaluated on 20 paired
unseen seeds:

| Fixed TTL | Policy | Completion | Timeout | TTL-capped delay |
|---:|---|---:|---:|---:|
| 20 | PPO | 59.0% | 41.0% | 14.555 |
| 20 | Q-DDCA-style | 59.5% | 40.5% | **14.475** |
| 20 | Greedy | 44.5% | 55.5% | 16.095 |
| 20 | Random valid | 44.5% | 55.5% | 15.925 |
| 30 | PPO | 81.5% | 18.5% | 17.410 |
| 30 | Q-DDCA-style | **83.5%** | **16.5%** | **17.200** |
| 30 | Greedy | 71.0% | 29.0% | 20.425 |
| 30 | Random valid | 69.5% | 30.5% | 20.195 |

TTL therefore removes the uninformative 100% eventual-completion ceiling.
However, it does not yet establish a PPO advantage: the no-TTL checkpoint is
0.5 percentage points behind Q-DDCA-style at TTL 20 and 2.0 points behind at
TTL 30.  The next defensible experiment is TTL-aware PPO fine-tuning, not a
claim that the present policy already wins.

Artifacts:

- `batchswap_rl/runs/reliq_long_ladder_finetune/eval_ttl20_20seeds.json`
- `batchswap_rl/runs/reliq_long_ladder_finetune/eval_ttl30_20seeds.json`

## 100-request contention benchmark

The long stage now contains 100 simultaneous requests rather than 10.  Request
sampling is balanced per episode: 34 requests at 20--29 hops, 33 at 30--39,
and 33 at 40--50.  With three candidate plans per request, the dynamic action
interface exposes at most 300 plans plus STOP.

The old 10-request checkpoint was evaluated zero-shot with a fixed TTL of 100
on 10 unseen seeds (1,000 requests per policy):

| Policy | Completion | Timeout | TTL-capped delay |
|---|---:|---:|---:|
| PPO | 38.7% | 61.3% | 79.803 |
| Q-DDCA-style | **41.3%** | **58.7%** | **78.825** |
| Greedy | 31.5% | 68.5% | 84.394 |
| Random valid | 32.1% | 67.9% | 84.353 |

Completion by hop bucket exposes a stronger structural difference:

| Policy | 20--29 hops | 30--39 hops | 40--50 hops |
|---|---:|---:|---:|
| PPO | 74.12% | **37.58%** | **3.33%** |
| Q-DDCA-style | **100%** | 22.12% | 0.00% |
| Greedy | 12.65% | 18.48% | 63.94% |
| Random valid | 45.88% | 29.39% | 20.61% |

Q-DDCA-style maximizes service to the shortest bucket but completely starves
40--50-hop requests under contention.  The zero-shot PPO redistributes service
toward medium/high-hop requests, improving the 30--39 bucket by 15.46
percentage points and the 40--50 bucket by 3.33 points, but sacrifices too many
short requests and remains 2.6 points worse in aggregate completion.  This is
a promising fairness/high-hop signal, not yet an overall performance win.

Artifact:

- `batchswap_rl/runs/reliq_long_ladder_finetune/eval_100req_ttl100_10seeds.json`
