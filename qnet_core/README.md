# Shared quantum-routing core

This package defines the boundary between the common simulation and routing
algorithms. The common side owns request generation, EPR generation, time
advancement, resource locking, exchange execution, TTL settlement, rewards,
and metrics. A planner receives an immutable `PlanningSnapshot` and returns
plan IDs (or `COMMIT`). It must not mutate the backend or call SeQUeNCe.

Episode metrics distinguish final request completion from intermediate
resource progress. Evaluation reports successful and partial plan counts,
cumulative `progress_hops`, and remaining active shortest-path distance.
Plan-level rates are diagnostic rather than a standalone ranking because
planners may choose different plan granularities.

The PPO reward uses a graph-derived frontier-progress potential. It does not
encode an expert route, fixed next-hop preference, or planner-specific action
rule.

The SeQUeNCe implementation will live behind this contract. Q-DDCA, PPO,
Greedy, and Random will use the same snapshot and commit path.
