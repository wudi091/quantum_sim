# Shared SeQUeNCe routing core

`qnet_core` is the algorithm-independent boundary between planning methods and
the SeQUeNCe physical simulator.

The planning side sees only immutable specifications, snapshots, resource
views, construction DAGs, and execution results. It does not receive SeQUeNCe
entities and cannot directly edit quantum memories or Bell states.

## Main components

- `planning_spec.py` and `planner_api.py`: planner-visible topology, requests,
  candidates, and snapshots;
- `physical_api.py`: simulator-neutral physical calls and resource views;
- `fidelity_estimation.py`: adapter-owned conservative BDS/Werner fidelity
  estimates derived only from declared physical parameters and construction
  timing, without probing future test outcomes;
- `env.py`: shared routing execution, settlement, and metrics;
- `sequence_backend.py`: SeQUeNCe routers, channels, memories, generation,
  swapping, fidelity, and physical time;
- `runtime.py`: composition root connecting the neutral environment to
  SeQUeNCe;
- `construction_api.py`: neutral construction operation, DAG, snapshot, and
  event contracts;
- `construction_catalog.py`: bounded path and construction-plan candidates;
- `construction_executor.py`: deterministic contract executor;
- `sequence_construction_executor.py`: event-driven SeQUeNCe construction
  executor;
- `scheduled_execution.py`: neutral coarse-slot schedule DTOs and the adapter
  that preserves planned start/operation slots while SeQUeNCe advances real
  protocol time; it also contains the persistent scheduler used to execute a
  rolling episode;
- `construction_evaluate.py`: fixed-plan physical evaluator;
- `evaluate.py`: seeded Q-DDCA/Q-CAST comparison entry point.

## Dependency direction

```text
planner -> neutral snapshot / construction plan
                         |
                         v
              qnet_core execution boundary
                         |
                         v
                 SeQUeNCe physical layer
```

Planning code calls the physical layer only through neutral contracts.
SeQUeNCe remains responsible for generation, swapping, memory state,
decoherence, fidelity, failures, expiration, and physical event time.
The fidelity estimator is only a planning-side feasibility abstraction. Final
claims always use independent SeQUeNCe executions.

## Construction-plan validation

The construction path can execute either a fixed route/construction mapping
through `run_joint_plan_baseline()` or a fully time-expanded batch schedule
through `run_scheduled_construction_plan()`. The latter preserves admission,
start slots, operation slots, and construction dependencies. Compatible
operations are submitted atomically; protocols that SeQUeNCe must serialize
may run at different physical instants inside the same coarse planning slot.
Crossing the slot boundary is reported explicitly as a schedule violation.

Both results contain request settlements, physical metrics, memory telemetry,
and neutral event traces. No SeQUeNCe entity crosses into planning code.

`PersistentConstructionScheduler` keeps one SeQUeNCe executor alive throughout
the rolling episode. It accepts immutable plans at decision boundaries,
executes their absolute operation slots, keeps running plans fixed across later
decisions, and exposes only neutral events, outcomes, reservations, launches,
and violations.

This execution foundation is retained for the active TELGEN plan. The future
TELGEN planner will output discrete feasible plans to this boundary; the core
does not contain a learned planning method.

## Checks

```bash
python -m qnet_core.sequence_smoke
python -m unittest discover -s qnet_core/tests -q
```
