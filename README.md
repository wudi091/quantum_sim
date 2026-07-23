# Quantum Resource-Graph Routing

This repository uses one fixed SeQUeNCe physical environment for every
algorithm. Request generation, EPR generation, physical step advancement,
memory lifetime, exchange execution, conflict validation, TTL settlement,
rewards, and metrics are implemented once in `qnet_core`.

PPO, Q-DDCA, Greedy, and Random only select routing/exchange plan IDs from the
same immutable planning snapshot. They cannot create requests, mutate EPR
resources, advance time, reject requests, or perform settlement themselves.

Each committed planning batch is the complete exchange schedule for one
physical time slot. Candidate-selection microsteps do not advance time; STOP
atomically executes all compatible plans selected for the requests in that
batch and advances the environment by exactly one slot. A multi-hop plan may
therefore reach an intermediate frontier or the destination in that slot.
Checkpoints and benchmark results produced with the former swap-depth timing
belong to a different MDP and must be retrained before comparison.

## Packages

- `qnet_core`: shared SeQUeNCe kernel, scenario generation, candidates,
  settlement, metrics, and planning-only baselines.
- `routing_rl`: masked PPO model, rollout, training, and paired evaluation.
- `QDDCA`: pristine upstream Q-DDCA source retained as the algorithm reference;
  its simulator is not used as an experiment backend.

## Environment

SeQUeNCe 1.0 requires Python 3.12 or newer. Install CUDA-enabled PyTorch for
your server separately, then install the project dependencies:

```bash
pip install -r requirements.txt
```

Validate the physical kernel and shared environment:

```bash
python -m qnet_core.sequence_smoke
python -m unittest discover -v qnet_core/tests
python -m unittest discover -v routing_rl/tests
```

Reproduce the Q-DDCA official throughput, timeout/drop, congestion-window,
rerouting, and fairness trends on the shared SeQUeNCe environment:

```bash
python -m qnet_core.qddca_trends \
  --seeds 3 --output results/qddca_sequence_trends_3seed.json
python -m qnet_core.plot_qddca_trends
```

## Train

There is no backend option. Training always uses SeQUeNCe.

```bash
python -m routing_rl.train \
  --device cuda \
  --min-hops 2 \
  --max-hops 50 \
  --updates 1000 \
  --requests 100 \
  --request-ttl 64 \
  --potential-coef 0.1 \
  --completion-bonus 1.0 \
  --generation-probability 0.5 \
  --swap-probability 0.95 \
  --memory-capacity 2 \
  --topology-nodes 200 \
  --waxman-alpha 0.05 \
  --waxman-beta 0.02 \
  --output routing_rl/runs/sequence_default
```

By default, training uses one full-range stage (2--50 hops); it does not
switch through short/medium/long curricula. Add `--curriculum` to restore the
legacy three-stage schedule.

Each episode uses a seeded sparse Waxman topology plus its Euclidean minimum
spanning tree for guaranteed connectivity. Topologies whose true graph
diameter is below `max_hops` are rejected. Requested hop lengths are spread
across the selected range, and source/destination endpoints are sampled from
the corresponding true shortest-path-distance buckets. Requests are not all
anchored at node 0.

The full-range default uses a shared swap success probability of 0.95. With
the old 0.5 setting, a failed extension destroys the carried EPR and a 20--50
hop request requires an exponentially unlikely run of consecutive successful
swaps, so neither PPO nor Q-DDCA receives a meaningful completion signal.

## Multi-width allocation and recovery

Set `--max-width` above one to enable the common QCAST-style two-phase
resource protocol.  Every algorithm, including PPO, receives the same public
catalogue and follows the same state machine:

```text
ALLOCATE(path, width) -> realized link EPRs -> RECOVER/EXECUTE -> settle
```

Allocations claim exact `(edge, lane)` units and consume real node memory
slots.  Realized surplus EPRs form one shared recovery graph for the slot;
recovery candidates are exposed as plan IDs rather than being chosen inside
the backend.  `demand_pairs` controls how many end-to-end EPRs complete one
request, so width greater than one contributes to pair throughput instead of
silently discarding extra successful lanes.

Example:

```bash
python -m routing_rl.train \
  --max-width 2 --demand-pairs 2 \
  --memory-capacity 2 --node-memory-capacity 12 \
  --candidates-per-request 6 \
  --output routing_rl/runs/qcast_common
```

## Fair evaluation

```bash
python -m routing_rl.evaluate \
  --checkpoint routing_rl/runs/sequence_default/checkpoint.pt \
  --episodes 20 \
  --requests 100 --min-hops 2 --max-hops 50 \
  --request-ttl 64 \
  --output routing_rl/runs/sequence_default/evaluation.json
```

PPO, Q-DDCA, Greedy, and Random receive the same episode seeds and therefore
the same topology, request set, and physical random process.
