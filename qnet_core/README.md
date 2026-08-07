# Shared SeQUeNCe routing core

`qnet_core` is the algorithm-independent simulation boundary for this
repository. The common side owns request generation, resource locking, TTL
settlement, rewards, and metrics. `SequenceBackend` delegates the physical
layer to SeQUeNCe: routers and memory arrays, quantum/classical channels, BSM
devices, single-heralded generation, Bell-diagonal swapping, detector/channel
loss, physical time, and memory expiration.

The physical implementation is `SequenceBackend`, a small adapter around
SeQUeNCe. Routing code only sees pair IDs and immutable planning contracts; it
does not receive SeQUeNCe objects.

## Main components

- `scenario.py`: deterministic Waxman-style topology and request generation;
- `planning_spec.py`: topology, requests, and horizon visible to routing;
- `spec.py`: composition-level episode and physical configuration types;
- `command_api.py`: simulator-neutral allocation and swap command DTOs;
- `planner_api.py`: `PlanningSnapshot`, `PlanDescriptor`, `SwapAction`, and
  related planner contracts;
- `physical_api.py`: simulator-neutral calls and immutable resource views used
  by the routing environment;
- `env.py`: `SharedRoutingEnv`, including allocation, execution, settlement,
  progress potential, and metrics;
- `sequence_backend.py`: the routing index around SeQUeNCe's physical
  entities and protocols;
- `sequence_scheduler.py`: conservative resource, segment, and physical-node
  launch validation for the SeQUeNCe adapter;
- `runtime.py`: the only module that wires `SharedRoutingEnv` to
  `SequenceBackend`;
- `gym_env.py`: masked fixed-size observation/action wrapper;
- `construction_api.py`: neutral operation/DAG/snapshot/event contracts;
- `construction_decoder.py`: exact canonical resource-feasibility mask;
- `construction_executor.py`: deterministic contract/reference executor;
- `sequence_construction_executor.py`: event-driven SeQUeNCe construction
  adapter with physical timestamps and event feedback;
- `construction_repair.py`: neutral bounded retry, failed-SWAP prefix
  reconstruction, and catalogue-DAG rebasing for reroute choices;
- `construction_catalog.py` / `construction_evaluate.py`: bounded joint
  `(path, construction)` catalogue and baseline evaluator;
- `evaluate.py`: seeded Q-DDCA/Q-CAST comparison entry point.

## Planner contract

Each planning step follows this boundary:

```python
snapshot = env.snapshot()
plan_ids = planner.select(snapshot)
env.commit(plan_ids)
```

`commit` is the only path that can execute exchanges or advance physical time.
An empty commit is a one-slot wait. Multi-hop plans are atomic within that
slot; swap count is recorded as physical work and is not treated as a separate
time increment.

The Q-DDCA and Q-CAST implementations live in the top-level `algorithms`
package. They only score and pack candidates from the snapshot.

Q-DDCA-specific paper reproduction also lives under `algorithms/qddca`; the
core contains no policy or experiment implementation:

```bash
python -m algorithms.qddca.reproduce --experiment exp1 --quick
```

The module dependency is deliberately one-way:

```text
planner -> PlanningSnapshot <- SharedRoutingEnv -> PhysicalBackend
                                                   |
                                                   v
runtime.py composition root -> SequenceBackend -> SeQUeNCe
```

The event-driven construction path is separate from the legacy atomic-slot
environment.  A minimal fixed-plan run is:

```python
from algorithms.caappo import ShortestPathLeftDeepPolicy
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.construction_evaluate import run_joint_plan_baseline

catalogue = build_route_construction_catalogue(spec.planning, candidate_count=3)
selection = ShortestPathLeftDeepPolicy().select(catalogue)
result = run_joint_plan_baseline(spec, selection)
```

`SequenceConstructionExecutor.snapshot()` is read-only. Generation and swap
protocols are started through the executor, SeQUeNCe is advanced to the next
physical event, and outcomes are returned as neutral `ExecutionEvent` values.
The snapshot also carries the complete current operation universe, so a
planner-side dependency encoder can see completed, ready, and blocked DAG
relations without receiving simulator objects.
The adapter currently advertises conservative protocol concurrency. SeQUeNCe
1.0.0 can safely overlap independent GEN operations across epochs, but a
GEN/SWAP overlap or concurrent SWAP protocols can race in the shared
Bell-diagonal state manager. `SequenceConcurrencyScheduler` therefore checks
resource capacity, input-segment exclusivity, post-completion holds, and
physical-node conflicts, and rejects those unsupported combinations before
starting a protocol. The deterministic executor remains the contract oracle
for DAG and mask tests.

`run_joint_plan_baseline()` is a fixed-plan SeQUeNCe evaluator with explicit
arrival, deadline, expiration, failure, and horizon settlement boundaries. It
does not perform repair; repair-aware training uses `JointConstructionBatchEnv`.
The joint environment exposes structured `RETRY` and `REROUTE` choices in the
REPAIR phase. A reroute is bounded by `max_route_repairs`, releases surviving
segments through explicit operations, supersedes only the old uncommitted DAG
suffix, and executes a freshly versioned catalogue candidate. DROP remains a
request-level settlement action.

When `dynamic_repair_paths > 0`, admission still uses a fixed catalogue, but
the `REPAIR` phase may enumerate up to that many previously unseen shortest
simple routes from the neutral topology DTO. Each route is compiled into the
same `(path, construction)` candidate format and passes the same intrinsic
capacity/schedule check before exposure to the policy. Per-request route-plan
lineage prevents a later reroute from returning to an already attempted
`(route, construction)` pair while still allowing another construction plan on
the same route.
This is bounded topology-generated repair; arbitrary unbounded route synthesis
and a backend protocol arbiter for unsupported SeQUeNCe overlap remain future
work.

`SharedRoutingEnv` accepts only `PlanningSpec` plus an injected
`PhysicalBackend`. It never imports `SequenceBackend`, reads `PhysicalConfig`,
touches SeQUeNCe entities, or mutates the backend's pair inventory. Capacity
checks, physical estimates, generation, swapping, resource ownership, and
time advancement are backend calls.
