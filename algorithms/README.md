# Routing algorithms

Each algorithm is a planning-only adapter for the routing environment in
`qnet_core`.

Current implementations:

- `qddca/legacy_planner.py`: Q-DDCA Algorithm 2/3 planning adapter;
- `qddca/reproduce.py`: paper exp1/exp2 parameter sweeps on SeQUeNCe;
- `qddca/plot.py`: optional rendering of the saved trend comparison;
- `qcast/legacy_planner.py`: Q-CAST expected-throughput ranking.
- `caappo/policy.py`: dependency-light CAAPPO action semantics, relation-aware
  DAG encoder, canonical operation masking, and CMDP dual update;
- `caappo/trainer.py`: event-epoch rollout loop with Monte Carlo value targets,
  transition risk cost-to-go, and episode-level CMDP accounting;
- `caappo/torch_policy.py` and `caappo/torch_trainer.py`: trainable PPO heads,
  event-duration GAE, masked route/operation/repair actions, and the same
  neutral SeQUeNCe-backed rollout contract;
- `caappo/oracle.py` and `caappo/experiment.py`: bounded nominal oracle plus
  seeded baseline/CI/ablation harness for small-instance validation;
- `caappo/baselines.py`: shortest-path/left-deep, balanced, and memory-aware
  joint route/construction baselines.

Both planners receive an immutable `PlanningSnapshot` and return plan IDs.
They do not create requests, mutate EPR resources, advance time, reject
requests, or perform settlement. Request lifecycle remains in
`SharedRoutingEnv`; physical actions are delegated through the injected
`qnet_core.physical_api.PhysicalBackend`.

```python
from algorithms import QCASTPlanner, QDDCAPlanner
```

CAAPPO consumes the newer `ConstructionSnapshot` contract. Its NumPy policy is
a reference implementation for action semantics and reproducibility; it does
not claim a converged RL result by itself. The PyTorch module trains actor /
critic heads over the same masked actions and categorically scores DROP plus
all generated retry options. Its trainable relation-aware message-passing
encoder and bounded failed-SWAP prefix repair are implemented; broader repair
candidate families remain explicit current limitations.
Physical execution remains in the SeQUeNCe-backed executor under `qnet_core`.

Run a small construction-aware sanity experiment with:

```bash
python -m algorithms.caappo.experiment --quick
```

The harness keeps QDDCA and QCAST reproductions separate because their legacy
action spaces are not the construction-SMDP action space used by this table.
Confidence intervals use evaluation seeds as the independent units; repeated
training replicas are averaged within each evaluation seed before aggregation.
The `no_capacity_context` ablation removes only the observation feature; the
hard capacity mask stays enabled, while true mask-removal is a future
action-rejection experiment.

Run a quick parameter sweep with:

```bash
python -m algorithms.qddca.reproduce --experiment exp1 --quick
```

Run the original qualitative trend check through the same entry point:

```bash
python -m algorithms.qddca.reproduce --mode trends --seeds 3
```
