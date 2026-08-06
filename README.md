# Quantum Resource-Graph Routing

This repository contains a shared SeQUeNCe-backed quantum-network routing
environment and two planning-only baselines: Q-DDCA and Q-CAST.

The simulator owns request generation, resource locking, TTL settlement,
rewards, and metrics. SeQUeNCe owns the physical layer: `QuantumRouter` and
`MemoryArray` entities, per-link `QuantumChannel` and `BSMNode` hardware,
single-heralded entanglement generation, BDS entanglement swapping, detector
and channel losses, physical time, and memory expiration events. A planner
receives an immutable `PlanningSnapshot` and returns plan IDs. It cannot
mutate the backend or advance physical time.

## Packages

- `qnet_core`: scenario generation, shared routing environment, SeQUeNCe
  backend, Gym-style wrapper, planner contracts, rewards, metrics, and
  reproduction utilities.
- `algorithms/qddca`: Q-DDCA planning adapter.
- `algorithms/qcast`: Q-CAST planning adapter.
- `QDDCA`: upstream Q-DDCA reference code based on SimQN, isolated from the
  project test suite by a package-level discovery shim.
- `QCAST`: upstream Kotlin/Maven Q-CAST reference simulator.

The upstream `QDDCA` and `QCAST` directories are references. The runnable
comparison environment is assembled by `qnet_core.runtime.make_sequence_env`.

## Environment

SeQUeNCe 1.0 requires Python 3.12 or newer. Install dependencies with:

```bash
pip install -r requirements.txt
```

Run the physical three-node smoke test:

```bash
python -m qnet_core.sequence_smoke
```

Run the shared-environment test suite:

```bash
python -m pytest -q
# or only the core tests:
python -m unittest discover -v qnet_core/tests
```

Run a fair Q-DDCA/Q-CAST comparison on identical seeded episodes:

```bash
python -m qnet_core.evaluate \
  --seeds 20 \
  --requests 100 \
  --min-hops 2 \
  --max-hops 50 \
  --ttl 64
```

The two planners receive the same episode specification and physical random
process. They differ only in the plans selected from the shared snapshot.

The legacy probability fields remain in `PhysicalConfig` as compact scenario
controls. `generation_probability` is translated into endpoint memory
efficiency, while `swap_probability`, channel attenuation, detector
efficiency, BSM success, and memory coherence are passed to SeQUeNCe
components and protocols. The backend never samples an EPR or edits a Bell
state directly.

## Runtime flow

```text
ScenarioConfig + seed
        |
        v
make_episode() -> EpisodeSpec
        |
        v
make_sequence_env() [composition root]
        |
        +--> EpisodeSpec.planning -> SharedRoutingEnv -> PhysicalBackend
        |                                                    |
        |                                                    v
        +------------------------------------------> SequenceBackend -> SeQUeNCe
        |
        +--> PlanningSnapshot
                 |
                 +--> QDDCAPlanner.select()
                 +--> QCASTPlanner.select()
        |
        v
env.commit(plan_ids) -> metrics
```

`qnet_core.runtime.make_sequence_gym_env` exposes the same environment through
a masked, fixed-size observation/action interface for PPO-style controllers.

The dependency is one-way. Planning and Gym code call the simulator-neutral
`PhysicalBackend` protocol and consume immutable `PhysicalResource` views.
`SharedRoutingEnv` receives a physical-config-free `PlanningSpec`; only the
composition root sees both the episode specification and concrete backend.
The composition root carries `PhysicalConfig` into `SequenceBackend`; only the
backend imports SeQUeNCe and interprets those physical parameters. No
SeQUeNCe entity is exposed to candidate construction or routing algorithms.

## Algorithm boundary

Algorithms are planning-only. Shared execution and settlement remain in
`qnet_core`; algorithm-specific code stays under its own package.

- Q-DDCA keeps bounded retry history and optional rerouting state.
- Q-CAST ranks candidates by expected throughput, memory cost, and remaining
  hops.

The stable public imports are:

```python
from algorithms import QCASTPlanner, QDDCAPlanner
from qnet_core.planner_api import PlanningSnapshot
from qnet_core.runtime import make_sequence_env
```
