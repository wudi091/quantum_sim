# Quantum Resource-Graph Routing

This repository contains a shared SeQUeNCe-backed quantum-network routing
environment and two planning-only baselines: Q-DDCA and Q-CAST.

The simulator owns request generation, EPR generation, memory accounting,
entanglement swapping, resource locking, time advancement, TTL settlement,
rewards, and metrics. A planner receives an immutable `PlanningSnapshot` and
returns plan IDs. It cannot mutate the backend or advance physical time.

## Packages

- `qnet_core`: scenario generation, shared routing environment, SeQUeNCe
  backend, Gym-style wrapper, planner contracts, rewards, metrics, and
  reproduction utilities.
- `algorithms/qddca`: Q-DDCA planning adapter.
- `algorithms/qcast`: Q-CAST planning adapter.
- `QDDCA`: pristine upstream Q-DDCA reference code based on SimQN.
- `QCAST`: upstream Kotlin/Maven Q-CAST reference simulator.

The upstream `QDDCA` and `QCAST` directories are references. The runnable
comparison backend in this repository is `qnet_core.SharedRoutingEnv` backed
by `qnet_core.SequenceBackend`.

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

## Runtime flow

```text
ScenarioConfig + seed
        |
        v
make_episode() -> EpisodeSpec
        |
        v
SharedRoutingEnv
        |
        +--> SequenceBackend (SeQUeNCe memories, EPRs, swaps)
        |
        +--> PlanningSnapshot
                 |
                 +--> QDDCAPlanner.select()
                 +--> QCASTPlanner.select()
        |
        v
env.commit(plan_ids) -> metrics
```

`qnet_core.gym_env.SequenceGymEnv` exposes the same environment through a
masked, fixed-size observation/action interface for PPO-style controllers.

## Algorithm boundary

Algorithms are planning-only. Shared execution and settlement remain in
`qnet_core`; algorithm-specific code stays under its own package.

- Q-DDCA keeps bounded retry history and optional rerouting state.
- Q-CAST ranks candidates by expected throughput, memory cost, and remaining
  hops.

The stable public imports are:

```python
from algorithms import QCASTPlanner, QDDCAPlanner
from qnet_core.env import SharedRoutingEnv
from qnet_core.planner_api import PlanningSnapshot
```
