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
  joint route/construction baselines, plus split-path contention baselines for
  parallel-corridor workloads.

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
candidate families remain explicit current limitations. The default experiment
variant enables a bounded topology-generated repair catalogue
(`dynamic_repair_paths=4`) after admission; repair routes are neutral DTOs and
are still executed only by the SeQUeNCe-backed environment. This does not claim
arbitrary route synthesis or support for GEN/SWAP overlap and concurrent SWAP
protocols.
Physical execution remains in the SeQUeNCe-backed executor under `qnet_core`.

Run a small construction-aware sanity experiment with:

```bash
python -m algorithms.caappo.experiment run --quick
```

The experiment CLI separates training from frozen evaluation. Training writes
an atomic, versioned checkpoint containing the policy, optimizer, CMDP dual,
completed episode counter, CPU/CUDA PyTorch RNG state, training history,
held-out validation record, and a matched best policy/optimizer/dual snapshot.
Resume always continues from the final training state; evaluation uses the
best validation state by default and never updates the policy. Strict loading
also checks the checkpoint hash, schema, runtime, and complete seed protocol.

```bash
# Train one replica.
python -m algorithms.caappo.experiment train \
  --quick --episodes 2 --training-seed 1 \
  --checkpoint results/checkpoints/caappo.seed-1.pt

# Continue the same replica to a larger total episode count.
python -m algorithms.caappo.experiment train \
  --quick --episodes 4 --training-seed 1 --resume \
  --checkpoint results/checkpoints/caappo.seed-1.pt

# Evaluate a frozen checkpoint on held-out seeds.
python -m algorithms.caappo.experiment evaluate \
  --checkpoint results/checkpoints/caappo.seed-1.pt \
  --evaluation-seeds 101 102 \
  --output results/caappo-evaluation.json

# Evaluate fixed baselines without retraining an RL policy.
python -m algorithms.caappo.experiment baselines \
  --config algorithms/caappo/configs/parallel_corridor_batch.json \
  --evaluation-seeds 1201 1202 1203 \
  --output results/parallel-corridor-baselines.json

# Compare multiple frozen RL variants with the same baselines and seeds.
python -m algorithms.caappo.experiment compare \
  --config algorithms/caappo/configs/parallel_corridor_batch.json \
  --checkpoint caappo=results/checkpoints/parallel-corridor.seed-11.pt \
  --checkpoint caappo=results/checkpoints/parallel-corridor.seed-12.pt \
  --checkpoint no_route_overlap=results/checkpoints/parallel-corridor-no-route-overlap.seed-11.pt \
  --checkpoint no_route_overlap=results/checkpoints/parallel-corridor-no-route-overlap.seed-12.pt \
  --evaluation-seeds 1201 1202 1203 \
  --output results/parallel-corridor-comparison.json
```

Training, validation, and evaluation seeds are disjoint. Each training replica
derives its own episode stream from its replica seed, validation seeds select
the checkpoint without touching the evaluation split, and confidence
intervals use evaluation seeds as the independent statistical units. A full
run writes persistent checkpoint hashes and run metadata into the result
manifest. All experiment parameters can also be supplied as JSON through
`--config`; the `manifest.config` object in a previous run result is accepted
directly for reproduction.

The primary CI measures performance over held-out evaluation seeds conditional
on the fixed averaged ensemble of supplied training replicas. Replicas are
averaged within each evaluation seed. A separate
`training_replica_aggregate` reports descriptive dispersion across the
supplied independently trained replicas; it is not combined into the primary
conditional CI.

Rows also preserve selected candidates and neutral physical event traces.
Physical failure, backend rejection, fidelity rejection, expiration,
post-completion validation failure, and executor launch rejection remain
separate counts. Derived rates use ratio-of-sums over evaluation-seed clusters;
zero-denominator clusters are omitted for that rate, and cluster influence/
delta intervals are reported. The expiration metric is an event density per
physical memory-unit-slot, not an intrinsic hazard. Admission and execution
mask-pruning fractions describe different policy stages and are absent for
fixed baselines where no corresponding learned-action mask exists.

The harness keeps QDDCA and QCAST reproductions separate because their legacy
action spaces are not the construction-event action space used by this table.
The fully observed SMDP interpretation is conditional on the documented
Snapshot Sufficiency Assumption; the current Torch encoder uses a lossy
information-state projection of the neutral environment snapshot.
Confidence intervals use evaluation seeds as the independent units; repeated
training replicas are averaged within each evaluation seed before aggregation.
The `no_capacity_context` ablation removes only the observation feature; the
hard capacity mask stays enabled, while true mask-removal is a future
action-rejection experiment.

The first three-replica medium training configuration is versioned at
`caappo/configs/medium_train3.json`. Current frozen results and their claim
boundary are recorded in `../RL_EXPERIMENT_STATUS.md`; the result files and
checkpoints remain under the ignored `results/` directory.

Run a quick parameter sweep with:

```bash
python -m algorithms.qddca.reproduce --experiment exp1 --quick
```

Run the original qualitative trend check through the same entry point:

```bash
python -m algorithms.qddca.reproduce --mode trends --seeds 3
```
