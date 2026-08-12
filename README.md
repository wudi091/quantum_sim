# Quantum Resource-Graph Routing

This repository provides a shared SeQUeNCe-backed quantum-network routing
environment and two planning-only baselines: Q-DDCA and Q-CAST.

The only active research plan is documented in
[`TELGEN_CONSTRUCTION_AWARE_ROUTING_PLAN.md`](TELGEN_CONSTRUCTION_AWARE_ROUTING_PLAN.md):
a generalizable construction-aware planner that jointly selects a path and an
entanglement-construction plan. The planning-only LP teacher, hard decoder,
rolling SeQUeNCe physical validation, and paired Q-CAST comparison are
implemented; the learned GNN planner is not implemented yet.

## Repository scope

- `qnet_core`: simulator-neutral planning contracts, scenario generation,
  construction-plan execution, metrics, and the SeQUeNCe physical adapter.
- `algorithms/qddca`: Q-DDCA planning adapter and reproduction utilities.
- `algorithms/qcast`: Q-CAST planning adapter.
- `QDDCA`: upstream Q-DDCA reference source.
- `QCAST`: upstream Q-CAST reference source.

Q-DDCA and Q-CAST are retained as comparison baselines. They are not part of
the proposed TELGEN method.

## Layer boundary

The planning layer receives immutable, simulator-neutral snapshots and returns
plan identifiers or construction plans. It cannot mutate the backend or
advance physical time.

SeQUeNCe exclusively owns:

- elementary entanglement generation;
- quantum memories and expiration;
- entanglement swapping;
- fidelity and decoherence;
- stochastic physical outcomes;
- physical event time.

```text
requests + topology
        |
        v
planning layer  --->  neutral plan interface  --->  SeQUeNCe physical layer
        ^                                                |
        +---------------- neutral results ---------------+
```

Optimization iterations and SeQUeNCe physical events are separate timelines.

## Environment

SeQUeNCe 1.0 requires Python 3.12 or newer.

```bash
pip install -r requirements.txt
python -m qnet_core.sequence_smoke
python -m pytest -q
```

Generate a small set of simultaneous-request LP teacher records:

```bash
python -m algorithms.telgen.generate_teacher_data \
  --output results/telgen_teacher \
  --samples 10 \
  --requests 8 \
  --min-hops 2 \
  --max-hops 5 \
  --ttl 12 \
  --horizon 12 \
  --paths 3
```

Each NPZ record contains both LP stages, their complete primal interior-point
trajectories, constraint violations, topology/request provenance, and the
same opaque resource-capacity catalogue used by the SeQUeNCe construction
executor.

Calibrate low, medium, and high static loads before large-scale generation:

```bash
python -m algorithms.telgen.calibrate_static_load \
  --output results/telgen_load_calibration \
  --samples 1 \
  --seed-start 100
```

The default profiles use `8 requests / 12 slots`, `24 / 6`, and `40 / 5`.
For each seed they share one topology and a nested request pool. The command
writes self-contained NPZ records, `calibration.json`, and `calibration.csv`
with completion, latency, fractionality, utilization, violation, and solver
statistics.

Audit the continuous relaxation against an exact small-instance binary MILP:

```bash
python -m algorithms.telgen.validate_discrete_gap \
  --seed 100 \
  --requests 8 \
  --horizon 6 \
  --paths 1
```

The MILP reuses exactly the same variables, lexicographic objectives, and
resource--time constraints as the LP teacher. It is a validation oracle only,
not a second teacher or a production planner.

Decode continuous LP scores into one executable, capacity-feasible plan:

```bash
python -m algorithms.telgen.evaluate_hard_decoder \
  --seed 100 \
  --requests 8 \
  --horizon 6 \
  --paths 1
```

The decoder enforces one candidate per request and all resource--time
capacities. It combines bounded beam search, deterministic multi-start greedy
rounding, one-request augmentation, and pair exchange, then reports its gap to
the small-instance MILP optimum.

Run TELGEN on one periodic micro-batch episode:

```bash
python -m algorithms.telgen.run_online \
  --output results/telgen_periodic
```

The default episode contains 100 requests. Ten requests arrive every four
slots at `0, 4, ..., 36`; every request has TTL 16, so the episode drains
through slot 52. At each boundary the planner sees all currently pending
requests, may start new plans only in the next four slots, and may let an
accepted construction complete after the next boundary. Already running plans
remain fixed and expose their future resource reservations. Results are written
as versioned JSON/CSV files plus fixed-name latest copies.

Compare TELGEN against the Q-CAST expected-throughput path baseline on the
exact same generated episodes and the same persistent SeQUeNCe execution
contract:

```bash
python -m algorithms.telgen.compare_online \
  --seeds 100
```

The primary benchmark uses a 64-node Waxman graph and the same periodic
100-request protocol: ten new requests every four slots, TTL 16, and automatic
drain to slot 52. Every request endpoint pair has shortest-path distance
exactly four. The
generator retries Waxman topology generation when that endpoint contract is
unavailable and fails explicitly after the configured attempt limit; it does
not silently relax the hop constraint. Yen supplies up to four real paths, and
TELGEN builds up to five valid order-preserving swap trees per path. Q-CAST
sees the same paths but always uses left-deep construction.

Pass `--uniform-random-endpoints` to disable the fixed shortest-hop endpoint
contract.

Both methods use the identical `EpisodeSpec`, arrival schedule, pending-request
queue, decision interval, topology, physical parameters, path limit, TTL, and
metrics. Q-CAST is an adaptation to this common rolling environment: it does
not use the TELGEN LP teacher or its hard decoder; only neutral construction
plans are submitted to an independent persistent SeQUeNCe scheduler.

Analyze multiple paired comparison reports with balanced per-scenario
bootstrap confidence intervals and paired randomization tests:

```bash
python -m algorithms.telgen.analyze_online_benchmark \
  results/telgen_qcast_waxman_fixed4_periodic/online_comparison.json \
  --output results/telgen_qcast_waxman_fixed4_periodic
```

The current sanity result and the frozen 100-episode protocol are recorded in
`refine-logs/EXPERIMENT_RESULTS.md` and `refine-logs/EXPERIMENT_PLAN.md`.

Run the Q-DDCA/Q-CAST comparison on identical seeded episodes:

```bash
python -m qnet_core.evaluate \
  --seeds 20 \
  --requests 100 \
  --min-hops 2 \
  --max-hops 50 \
  --ttl 64
```

Run the Q-DDCA reproduction utility:

```bash
python -m algorithms.qddca.reproduce --experiment exp1 --quick
```

## Current research boundary

The repository already contains a simulator-neutral construction DAG, the LP
teacher data pipeline, hard decoding, and rolling physical validation.
These are foundations for the TELGEN plan, not an
implemented learning method. No alternative learned planning scheme is
retained in the current tree.
