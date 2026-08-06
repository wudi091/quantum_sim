# Routing algorithms

Each algorithm is a planning-only adapter for the routing environment in
`qnet_core`.

Current implementations:

- `qddca/legacy_planner.py`: Q-DDCA local scoring, retry history, and optional
  rerouting;
- `qcast/legacy_planner.py`: Q-CAST expected-throughput ranking.

Both planners receive an immutable `PlanningSnapshot` and return plan IDs.
They do not create requests, mutate EPR resources, advance time, reject
requests, or perform settlement. Request lifecycle remains in
`SharedRoutingEnv`; physical actions are delegated through the injected
`qnet_core.physical_api.PhysicalBackend`.

```python
from algorithms import QCASTPlanner, QDDCAPlanner
```
