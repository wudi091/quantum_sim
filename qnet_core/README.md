# Shared quantum-routing core

This package defines the boundary between the common simulation and routing
algorithms. The common side owns request generation, EPR generation, time
advancement, resource locking, exchange execution, TTL settlement, rewards,
and metrics. A planner receives an immutable `PlanningSnapshot` and returns
plan IDs (or `COMMIT`). It must not mutate the backend or call SeQUeNCe.

The SeQUeNCe implementation will live behind this contract. Q-DDCA, PPO,
Greedy, and Random will use the same snapshot and commit path.
