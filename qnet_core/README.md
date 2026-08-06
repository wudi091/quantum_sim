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
- `spec.py`: immutable episode, request, and physical configuration types;
- `planner_api.py`: `PlanningSnapshot`, `PlanDescriptor`, `SwapAction`, and
  related planner contracts;
- `env.py`: `SharedRoutingEnv`, including allocation, execution, settlement,
  progress potential, and metrics;
- `sequence_backend.py`: the routing index around SeQUeNCe's physical
  entities and protocols;
- `gym_env.py`: masked fixed-size observation/action wrapper;
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
