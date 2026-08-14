# Routing algorithms

The `algorithms` package contains planning-only comparison baselines for the
shared environment in `qnet_core`.

Current implementations:

- `qddca/legacy_planner.py`: Q-DDCA planning adapter;
- `qddca/reproduce.py`: Q-DDCA parameter sweeps on SeQUeNCe;
- `qddca/plot.py`: optional trend rendering;
- `qcast/legacy_planner.py`: Q-CAST expected-throughput ranking;
- `qcast/online_planner.py`: width-one Q-CAST EXT path ranking and
  resource--time greedy packing for fixed construction plans;
- `qcast/online.py`: rolling Q-CAST adaptation on the same queue and persistent
  neutral scheduler contract used by TELGEN;
- `telgen/time_expansion.py`: dependency-respecting construction schedules and
  resource--time candidate expansion;
- `telgen/teacher.py`: two-stage continuous LP teacher and complete primal IPM
  trajectory recording.
- `telgen/dataset.py`: simultaneous multi-request batch generation, candidate
  expansion, LP solving, and self-contained NPZ dataset records.
- `telgen/generate_teacher_data.py`: command-line teacher dataset entry point.
- `telgen/calibration.py`: nested low/medium/high static workloads and LP
  completion, latency, fractionality, utilization, and solver statistics.
- `telgen/calibrate_static_load.py`: command-line static-load calibration.
- `telgen/milp_oracle.py`: exact two-stage binary oracle and LP relaxation-gap
  report for small-instance auditing.
- `telgen/validate_discrete_gap.py`: command-line LP-versus-MILP validation.
- `telgen/milp_imitation.py`: candidate--constraint graph encoder,
  residual-capacity state, unordered MILP-set imitation loss, and a dynamically
  feasibility-masked autoregressive candidate/STOP policy.
- `telgen/gnn_policy.py`: checkpoint loading and direct online autoregressive
  GNN inference without thresholding, post-hoc plan repair, or local search.
- `telgen/train_online_milp_gnn.py`: episode-disjoint training and evaluation
  from persisted exact-MILP online graphs, with pooled validation throughput
  used for checkpoint selection.
- `telgen/generate_online_milp_data.py`: resumable exact-MILP rollout
  collection on parameterized Waxman, Barabasi--Albert, or corridor graphs.
- `telgen/combine_online_milp_datasets.py`: provenance-checked assembly of
  disjoint train/validation/test topology collections.
- `telgen/compare_online_gnn.py`: paired online GNN/Q-CAST execution with the
  exact episode, checkpoint hash, metrics, and violation records persisted.
- `telgen/analyze_online_gnn.py`: paired bootstrap, randomization tests, and
  hard-gate auditing for held-out online GNN/Q-CAST scenarios.
- `telgen/hard_decoder.py`: continuous-score to feasible 0/1 plan decoding,
  feasibility validation, and decoder-versus-MILP quality reporting for the
  legacy LP baseline only.
- `telgen/evaluate_hard_decoder.py`: command-line hard-decoder evaluation.
- `telgen/physical_validation.py`: hard-decoded schedule compilation,
  SeQUeNCe execution, repeated physical trials, and nominal-to-physical
  consistency metrics.
- `telgen/online.py`: rolling control with selectable LP-decoder, exact-MILP,
  or autoregressive-GNN planning backends on one persistent SeQUeNCe episode.
- `telgen/run_online.py`: command-line periodic micro-batch execution with versioned
  JSON/CSV results.
- `telgen/compare_online.py`: paired rolling TELGEN/Q-CAST comparison
  with versioned JSON/CSV results.
- `telgen/analyze_online_benchmark.py`: balanced paired bootstrap,
  randomization tests, hard-gate checks, and versioned JSON/CSV/Markdown
  summaries across multiple online comparison scenarios.

The Q-DDCA and Q-CAST planners consume an immutable `PlanningSnapshot` and
return plan IDs.
They do not create requests, mutate physical resources, advance time, or
perform settlement.

```python
from algorithms import QCASTPlanner, QDDCAPlanner
```

Run a quick Q-DDCA reproduction:

```bash
python -m algorithms.qddca.reproduce --experiment exp1 --quick
```

The active TELGEN-based construction-aware routing direction is recorded in
[`../TELGEN_CONSTRUCTION_AWARE_ROUTING_PLAN.md`](../TELGEN_CONSTRUCTION_AWARE_ROUTING_PLAN.md).
Its first implemented component is the planning-only teacher model. It uses
opaque resource capacities supplied by the environment, never imports
SeQUeNCe objects, and produces continuous LP trajectories rather than an
executable discrete plan.

```text
route/construction DAG
    -> relative nominal slots
    -> feasible start-slot variables
    -> maximize completed-request mass
    -> fix that mass and minimize total completion latency
    -> save both interior-point trajectories
```

The teacher now receives a conservative candidate-fidelity estimate by
default. The estimate is computed from the declared physical configuration,
construction tree, and relative storage slots through the neutral core
boundary. It neither executes the current test request nor reads future random
outcomes. Candidates below the request threshold are removed before LP
assembly, which is equivalent to fixing their decision variables to zero and
prevents a continuous LP from averaging feasible and infeasible fidelities.

Externally calibrated estimates may still be supplied explicitly. Final
throughput and fidelity claims always come from independent SeQUeNCe seeds;
the estimator is not used as the reported physical result.

Generate teacher samples with:

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

Waxman batches use independently distributed source--destination pairs whose
arrivals are all slot zero. Parallel-corridor batches intentionally retain one
shared `0 -> 1` pair to create controlled resource contention.

Before generating a large training corpus, verify that the chosen workload
contains meaningful decisions:

```bash
python -m algorithms.telgen.calibrate_static_load \
  --output results/telgen_load_calibration \
  --samples 1 \
  --seed-start 100
```

The default profiles are light `8 requests / 12 slots`, medium `24 / 6`, and
heavy `40 / 5`. Profiles for the same seed reuse exactly the same topology;
lighter request sets are prefixes of heavier sets. The resulting JSON and CSV
files expose completion ratio, average completion latency, fractional request
ratio, resource utilization, numerical violation, and solve time.

Validate the continuous teacher against a true 0/1 optimum on a small batch:

```bash
python -m algorithms.telgen.validate_discrete_gap \
  --seed 100 \
  --requests 8 \
  --horizon 6 \
  --paths 1
```

The report separates two cases. If LP and MILP complete the same number of
requests, their latency values are comparable and the LP value is a lower
bound. If LP throughput exceeds the integer optimum, latency is deliberately
marked incomparable. This MILP remains a small-instance validation oracle and
does not replace the single IPM trajectory teacher.

Decode the LP scores and compare the executable result with MILP:

```bash
python -m algorithms.telgen.evaluate_hard_decoder \
  --seed 100 \
  --requests 8 \
  --horizon 6 \
  --paths 1
```

The decoder first preserves multiple feasible partial schedules with a bounded
beam, then runs deterministic multi-start rounding and small one-request/pair
repairs. Its output always satisfies request uniqueness and resource--time
capacity constraints. It is heuristic rather than an exact solver, so the
MILP comparison remains part of the evaluation protocol.

The decoded variables can now be compiled without exposing simulator objects:

```text
hard-decoded variables
    -> admitted/rejected requests
    -> absolute operation slots
    -> neutral batch schedule
    -> SeQUeNCe physical execution
    -> completion, fidelity, memory, and schedule-adherence metrics
```

The adapter retains coarse planning slots. SeQUeNCe still chooses the real
event times and may safely serialize incompatible protocols inside one slot;
an operation that crosses its planned boundary is recorded as a violation
rather than hidden by the evaluator.

Run TELGEN on one periodic micro-batch episode:

```bash
python -m algorithms.telgen.run_online \
  --output results/telgen_periodic
```

By default ten requests arrive every four slots until 100 requests have been
generated. At each boundary the controller replans all pending requests. New
plans may start only before the next boundary, but their full construction may
finish later, subject to the request TTL and episode end. Deferred and failed
unexpired requests return to later decisions; accepted running plans are never
rearranged and reserve their future resource--time usage.
