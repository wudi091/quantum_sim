# Quantum Resource-Graph Routing

This repository contains two shared-environment experiment tracks.  The
legacy routing track uses the SeQUeNCe-backed `SharedRoutingEnv`; the
formal swap-order track uses the event-driven `order_core` through
`OrderEpisodeEnv`.
Within either track, all compared planners share the same requests, resource
state, physical executor, settlement rules, and metrics.  Results from the two
tracks should not be presented as if they came from one identical backend.

Q-DDCA, Optimal-MILP, Greedy, and Random only select routing/exchange plan IDs from the
same immutable planning snapshot. They cannot create requests, mutate EPR
resources, advance time, reject requests, or perform settlement themselves.

Each `env.step(batch_action)` submits the complete exchange schedule for one
physical time slot and advances the environment by exactly one slot. A policy
may construct the batch with autoregressive candidate choices and an internal
STOP token, but those decoder operations never cross the environment boundary
or advance physical time. A multi-hop plan may therefore reach an intermediate
frontier or the destination in that slot.
## Packages

- `qnet_core`: the SeQUeNCe-backed routing environment plus the separate
  event-driven swap-order core, scenario generation, candidates, settlement,
  metrics, and planning-only baselines.
- `construction`: independent deterministic Phase 1 kernel for jointly selecting
  paths and construction plans, including sequential/balanced enumeration,
  peak-memory footprints, greedy admission, and CP-SAT selection that is exact
  only for that deterministic footprint abstraction.
- `QDDCA`: pristine upstream Q-DDCA source retained as the algorithm reference;
  its simulator is not used as an experiment backend.

## Environment

SeQUeNCe 1.0 requires Python 3.12 or newer. Install the project
dependencies:
```bash
pip install -r requirements.txt
```

Validate the physical kernel and shared environment:

```bash
python -m qnet_core.sequence_smoke
python -m unittest discover -v qnet_core/tests
```

Reproduce the Q-DDCA official throughput, timeout/drop, congestion-window,
rerouting, and fairness trends on the shared SeQUeNCe environment:

```bash
python -m qnet_core.qddca_trends \
  --seeds 3 --output results/qddca_sequence_trends_3seed.json
python -m qnet_core.plot_qddca_trends
```


## Fair evaluation

```bash
python -m qnet_core.evaluate --seeds 20 \
  --requests 100 --min-hops 2 --max-hops 50 --ttl 64
```

Q-DDCA, Optimal-MILP, Greedy, and Random receive the same episode seeds and therefore
the same topology, request set, and physical random process.

The legacy visible-catalogue MILP comparison is:

```bash
python -m qnet_core.evaluate --planners qddca optimal --seeds 30 \
  --requests 30 --min-hops 2 --max-hops 6 --ttl 10 \
  --arrival-rate 4 --p-gen 0.5 --p-swap 1.0 --memory 2 \
  --output results/qddca_vs_optimal_slot_30seed.json
```

Here `optimal` exactly solves the visible descriptor-level catalogue MILP.
With `p_gen=0.5`, it is not an exact optimizer of realized event completions
and is unrelated to the executor-verified swap-order MILP below.

## Multi-step swap-order training environment

`qnet_core.order_episode_env.OrderEpisodeEnv` is the authoritative training
and online-evaluation environment.  The formal profile runs 30 independent
episodes.  Each episode fixes one 20-node topology, contains exactly 30
control steps, and places exactly 100 requests in slots 0--29 using a
homogeneous Poisson process conditioned on that fixed total.  One
`env.step(batch_action)` advances exactly one control slot; topology, pending
requests, elementary-EPR inventory, TTLs, and metrics persist across steps.
The main memory-pressure profile uses node memory 2, request TTL 5 slots,
4,000 ps control slots, 1,000 ps automatic HEG intervals, and 1,000 ps swap
service.  Thus the controller still acts once per slot while the event-driven
physical layer may attempt generation up to four times.

At control boundary `t`, the formal request set `R_t` contains every request
that has arrived, remains unfinished, and has not expired, including backlog
from earlier slots.  Its size is variable, and by default every request in
`R_t` receives candidates.  The Gym observation/action tensors use the
episode request count only as a padding upper bound with masks; that fixed
shape is not a fixed decision batch.  `candidate_request_cap` (CLI:
`--candidate-request-cap`) is an optional EDF-prefix candidate-pruning
approximation and is disabled by default.  If used in an approximation study,
the same cap must be reported and applied consistently to compared methods.

The standard action is a `MultiBinary(max_candidates)` vector selecting one
atomic batch of complete `(request, path, swap_order)` candidates.  A policy
may decode that set autoregressively, but decoder tokens and STOP are internal
to the policy and are not environment steps.  Empty slots still require one
all-zero action and advance one slot.

```python
observation, info = env.reset(seed=episode_seed)
terminated = truncated = False
while not (terminated or truncated):
    batch_action = policy.complete_multi_hot_batch(observation)
    observation, reward, terminated, truncated, info = env.step(batch_action)
```

The formal action catalogue is deliberately finite: every request in the
variable-size `R_t` receives four preconfigured paths, and every path receives
four complete linear swap-order candidates (`candidate_paths=4`,
`order_variants_per_path=4`; if fewer than four legal paths or orders exist,
all of them are retained).  "Complete" means that each candidate specifies the
whole linear swap order; it does not mean that the catalogue contains every
permutation.  This `4 x 4` catalogue width is part of the formal action-space
definition, not a request-pruning approximation.  The CLI defaults
`--order-variants` to 4 even when the flag is omitted.  Enumerating every
permutation is available only to programmatic callers that explicitly set
`order_variants_per_path=None`, as an expanded-catalogue diagnostic rather
than the formal profile.  Q-DDCA and Q-CAST still apply their own path scores
and submit only the canonical fixed order.  An MILP/SAA method uses the same
visible `4 x 4` path/order catalogue as the method it bounds or ablates.

Q-DDCA, Q-CAST, optimization baselines, and future GNN+RL policies use the
same episode and physical API:

```bash
python -m qnet_core.order_waxman_benchmark \
  --episodes 30 --base-seed 0 \
  --nodes 20 --requests 100 --steps 30 \
  --candidate-paths 4 --order-variants 4 \
  --node-memory 2 --ttl 5 \
  --slot-duration-ps 4000 --generation-interval-ps 1000 \
  --planners qddca_fixed qcast_fixed \
             milp_nominal_path milp_nominal_path_order \
  --output results/order_episode_20n_100req_30step_30epi_stress_medium.json
```

For a fixed episode, every algorithm receives the same topology, request
arrivals, link parameters, and hidden exogenous random stream.  The physics
seed root is independent of both the public workload seed and the separate
planner seed.  Their later inventory and pending-request states may diverge
only because their routing actions differ.

`milp_reliable_memory_path` and `milp_reliable_memory_path_order` are the
deterministic online model baselines used by the formal CON generator
benchmark.  They convert link probabilities into per-link binomial reliable
EPR arrivals at an explicit confidence level, allocate current inventory at
most once, and enforce time-indexed node-memory, elementary-link-buffer, and
BSM constraints.  Complete swap groups determine when each internal node
releases its two memories.  The objective is lexicographic: completed request
count, nominal success probability, then memory-time/completion time.  The
reported optimum is exact for this deterministic abstraction; stochastic
executor completions are reported separately.

`milp_nominal_path` and `milp_nominal_path_order` solve the same online
one-slot objective: maximize the number of requests that complete in a fixed,
planner-owned deterministic physical scenario.  The original link-generation
and swap probabilities remain active in that scenario, but the environment's
realized physics seed is never exposed.  The path-only variant keeps only the
canonical order; the path+order variant searches the given formal `4 x 4`
candidate catalogue.  A compact 0-1 master proposes a complete assignment,
the shared executor validates it on the fixed planning scenario(s), and an infeasible
assignment adds a safe no-good cut excluding only that exact action.  Because
the candidate action space is finite, this lazy procedure terminates with the
exact optimum of that finite planning model; in the worst case it may require
exponentially many assignments/cuts.  Every MILP master requests zero relative
MIP gap and rejects a result whose reported gap or primal/dual bound is not
closed, rather than labeling a tolerance-optimal result exact.  The nominal
objective and the shared executor's realized completion count are reported
separately.  It is not a clairvoyant optimum for hidden physics or a global
optimum for the episode.
The executor fixes cross-request arbitration to EDF-derived request priority
and fixes the within-timestamp event phase (generation before swap), so the
certificate optimizes admission, candidate path, and one of the four complete
linear swap-order candidates per path under that executor.  Its exactness is
therefore relative to the given `4 x 4` catalogue.  It does not optimize an
arbitrary global inter-request event schedule.

The legacy `OrderGymEnv` remains a one-slot observation encoder and
mechanism-test helper.

The current 30-episode result is in
`results/order_episode_20n_100req_30step_30epi_stress_medium_with_milp.json`.
Its independently rolling MILP-Path and MILP-Path+Order trajectories have
hidden-physics completion rates 0.5283 and 0.5267; those trajectories cannot
isolate swap-order value because their inventories and pending queues diverge.

The paired ablation in
`results/order_paired_same_snapshot_20n_30epi_fixed_driver.json` evaluates
both MILPs on each of the same 900 snapshots before only FixedOrder advances
the environment.  The mean nominal objective rises from 2.950 to 3.110
requests per decision slot (`+0.160`, or `+5.42%`), with a strict gain on
140/900 snapshots.  This archived experiment used
`candidate_request_cap=4`, so `+5.42%` is a result for a request-capped
same-snapshot finite planning subproblem rather than the full variable-size
`R_t`.  Its `candidate_paths=4` and `order_variants_per_path=4` catalogue,
however, already matches the formal action-space definition and is not a
second erroneous cap.  The result is neither unique episode throughput nor
evidence that swap-order optimization generally improves realized throughput
under random physics and the full variable-size `R_t`.
See its matching `_summary.md` for strata, invariants, and planning cost.

## Swap-order memory-release mechanism benchmark

The order-aware shared core uses physical-time events rather than controller
subrounds. A planner acts once; automatic HEG attempts, BSM service, memory
reset, and settlement are driven by `slot_duration_ps`, protocol intervals,
and operation latencies.

```bash
python -m qnet_core.order_benchmark --seeds 30 \
  --hotspot-capacities 2 4 \
  --output results/order_core_milp_30seed.json
```

With deterministic physics this compares `Q-DDCA+Fixed`, `MILP-Path` with the
same fixed swap order, and `MILP-Path+Order` over the complete small-instance
order catalogue.  The profile MILP supplies an upper bound; only plan IDs are
sent to the shared event executor.  A result is certified exact only when the
executor attains that upper bound, checked as
`MILP objective == environment completed count`. Capacity 4 is the control
that removes the throughput benefit of early hotspot release.

The certificate is currently limited to a validated deterministic single-
hotspot motif: one required preloaded multi-swap main request, lower-priority
two-hop waiting requests that intersect only at one even-capacity hotspot, no
initial inventory, unit edge/BSM capacities, `p_gen=p_swap=1`, swap service
equal to the generation interval, reset no longer than one generation
interval, and no elementary-edge sharing. Unsupported problems are rejected.
