# Construction-Aware Routing Implementation Status

Date: 2026-08-07

The repository now contains a runnable construction-event foundation on top of
SeQUeNCe. The planning side exchanges neutral DTOs; the SeQUeNCe adapter owns
memory, link, BSM, protocol, fidelity, decoherence, and physical time.

Implemented and tested:

- immutable construction contracts, DAG versioning, atomic launch and repair;
- canonical cross-request ready-set decoding with additive resource demands;
- persistent link/node-memory holds, transient generation lanes and BSM demands;
- deterministic event executor and SeQUeNCe event executor with physical timestamps;
- same-path left-deep versus balanced construction traces;
- fidelity-gated terminal delivery and horizon-censored flow-time accounting;
- explicit admission -> execution -> repair/drop -> terminal wrapper;
- NumPy reference CAAPPO route actor, masked operation actor, value/constraint
  critics, transition-level risk cost-to-go targets, and an episode-level risk
  dual update.
- PyTorch CAAPPO policy heads with masked route/operation/repair sampling,
  clipped PPO losses, event-duration GAE, constraint critic, and CMDP dual
  updates; the PyTorch relation-aware DAG message-passing encoder and actor /
  critic heads are trainable.
- the potential-shaping term is the normalized remaining-DAG critical-path
  length and is zero on terminal states;
- Complete operation-universe snapshots keep dependency relations visible to
  the planner without exposing SeQUeNCe objects.
- canonical operation authenticity and global output-segment collision checks;
- physical-versus-logical completion boundaries, pending fidelity refresh, and
  request-level pending/drop guards.
- request settlement/drop releases resident SeQUeNCe pairs; output holds are
  checked against post-completion capacity and the exact physical pair schema;
  SWAP outer endpoints and request ownership are validated; operation/output
  IDs are revalidated globally even after direct DAG mutation or repair.
- sequential multi-pair delivery is compiled into one request DAG with fresh
  operation/segment IDs and explicit intermediate RELEASE operations; the
  construction evaluator reports delivered pairs and supports demand_pairs>1.
- request-level fidelity is enforced at the SeQUeNCe terminal delivery gate,
  again at neutral settlement, and cannot be weakened by a repair DAG.
- candidate admission checks each selected DAG for at least one sequential
  topological schedule that fits its own resource holds; admission preview
  usage is context only because execution resources can be time-shared by
  later requests.
- retry lineage and `retry_limit` are enforced by both executors. Failed SWAP
  branches can rebuild missing GEN/SWAP ancestors from the surviving prefix
  before a bounded retry. `ConstructionRepairChoice` now exposes structured
  RETRY and bounded REROUTE alternatives, including topology-generated
  out-of-catalogue routes; reroute atomically keeps
  the completed prefix, retires stale uncommitted operations, releases old
  physical segments through explicit RELEASE operations, rebases the selected
  candidate into a fresh DAG version, and replaces terminal segment IDs.
  Per-request route-plan lineage prevents a later reroute from returning to an
  already attempted `(route, construction)` pair while retaining same-route
  construction alternatives. The Torch repair head scores DROP and every
  structured choice categorically.
- the `no_capacity_context` ablation removes only the capacity feature from
  the policy observation. The hard capacity-feasibility mask remains enabled
  to keep every rollout physically valid; a mask-removal ablation requires an
  explicit action-rejection transition and is not claimed as implemented.
- the event boundary is the minimum of operation completion, SeQUeNCe memory
  expiration, request arrival, deadline, and horizon; expiration is stamped at
  its physical time and deadlines appear as neutral boundary events. Arrival
  is intentionally a boundary-only transition because it changes eligibility
  but has no physical success/failure outcome.
- admission is autoregressive in canonical request order with a preview state
  and legal candidate mask; the reference CAAPPO also has a retry/drop repair
  head whose retry options are generated from the surviving DAG prefix.
- reproducible CAAPPO training checkpoints now persist the policy and optimizer
  state, CMDP dual variable, completed episode count, PyTorch RNG state,
  experiment/variant contract, exact runtime package manifest, training
  history, and held-out validation-best policy. Writes are atomic and loads
  reject incompatible schemas, runtimes, scenarios, variants, or seed
  protocols unless runtime compatibility is explicitly relaxed.
- the experiment CLI now has independent `train`, `evaluate`, and `run`
  workflows. Training replicas derive distinct episode streams from their own
  seeds; validation and evaluation seeds are disjoint; resume continues the
  final optimizer state exactly; frozen evaluation defaults to the best
  validation state and verifies that no model or CMDP dual state is mutated.
- the experiment CLI also has a `baselines` mode for fixed-plan evaluation and
  a `compare` mode that evaluates multiple frozen checkpoint variants on the
  same held-out seeds, merges them with the fixed baselines, and reports the
  same aggregate and paired confidence intervals in JSON/CSV.
- JSON/CSV results contain checkpoint SHA-256 hashes and a machine-readable run
  manifest. Each checkpoint also has a JSON history sidecar, and a prior
  result's `manifest.config` can be passed back through `--config`.
- frozen evaluation rows record the actual route nodes and construction kind
  selected for every request; paired confidence intervals are reported against
  every fixed baseline, not only shortest-left-deep.
- SeQUeNCe's process-global protocol RNG is reset for every physical episode
  using a stable SHA-256 mapping from the 64-bit episode seed to the 32-bit
  timeline seed. Policy and baseline outcomes are therefore independent of
  method execution order.
- checkpoint validation rejects duplicate seed lists, locks every seed list
  used by future episode derivation during resume, preserves caller RNG during
  read-only loads, records validation eligibility at non-interval stopping
  points, and selects risk-feasible validation checkpoints when possible. The
  primary evaluation-seed CI estimand and separate training-replica dispersion
  are both reported explicitly.
- `ConstructionDAG.clone()` reconstructs a pristine operation graph with fresh
  execution markers. The SeQUeNCe executor applies this boundary on every
  construction run, so fixed-plan evaluation and `ConstructionBatchEnv.reset()`
  are idempotent even when the same catalogue objects are reused. Regression
  tests cover repeated evaluator runs and reset restoration.
- Fixed-plan evaluation exposes the next physical memory-expiration boundary
  through the neutral executor contract and waits for it even when no
  operation is in flight. This keeps expiration events, settlement timestamps,
  and makespan consistent with the SeQUeNCe backend.

The current SeQUeNCe capability declaration is intentionally conservative:
inter-epoch launch is enabled for protocol-compatible operations, while
same-epoch GEN/SWAP mixing, GEN/SWAP overlap across epochs, and concurrent
SWAPs are rejected because SeQUeNCe 1.0.0 shares Bell-diagonal protocol state
across those transitions. `SequenceConcurrencyScheduler` validates resource
capacity and logical holds, and the backend-owned `SequenceProtocolArbiter`
validates protocol-family coexistence, input-pair leases, and physical-node
scope before the adapter starts a protocol. Multiple operations that pass
those checks are packed deterministically into one launch epoch.

The SeQUeNCe executor now waits for SWAP protocol completion rather than
treating every SWAP as physically complete when a shorter logical boundary
event fires. This prevents delayed SWAP messages from accessing Bell states
after a request release has already freed the underlying memories.

The current parallel-corridor run trains three independent CAAPPO replicas for
400 episodes each on the SeQUeNCe backend, with route-overlap, no-route-
overlap, no-construction-choice, and no-route-choice variants. The frozen
30-seed completion rates are 0.5389, 0.1722, 0.5722, and 0.1444 respectively;
the split-balanced heuristic reaches 0.5500. This is evidence that the RL
policy uses the batch context, but it is not evidence of optimality or
convergence; the strong split heuristic remains statistically indistinguishable
from CAAPPO. The corrected result and claim boundary are recorded in
`RL_EXPERIMENT_STATUS.md`.

The SeQUeNCe executor also rejects logical-only `initial_segments`; importing
an existing physical state requires a future explicit logical-to-physical pair
binding contract. The deterministic reference executor may still use logical
initial segments for contract tests.

Remaining paper-complete gates:

- physical validation before enabling mixed GEN/SWAP or concurrent SWAP
  execution; the backend-level arbiter is implemented and conservatively keeps
  those capabilities disabled. Bounded topology-generated out-of-catalogue
  repair is implemented, while arbitrary unbounded repair synthesis remains a
  future extension;
- converged training and scaling evidence; the trainable PyTorch graph encoder
  and bounded failed-SWAP prefix repair are runnable, but convergence is not
  claimed;
- broader stochastic multi-seed evidence and catalogue/oracle coverage for a
  paper submission. The bounded nominal oracle and seeded CI/ablation harness
  are implemented for sanity and small-instance validation.

This is a runnable construction-aware event foundation with a SeQUeNCe-backed
Torch CAAPPO head, not yet a complete CCFA paper system. The multi-pair
evaluator, request-level fidelity gate, deadline/expiration settlement, bounded
nominal oracle, and reproducible sanity harness are implemented; mixed-protocol
physical scheduling, converged training, and statistically supported paper
claims remain outside the current claim.

Current regression commands, using the Conda environment
`D:\\software\\miniconda3\\envs\\qnet312` (Python 3.12.13, SeQUeNCe 1.0.0):

```text
conda run -p D:\\software\\miniconda3\\envs\\qnet312 python -m pytest -q
conda run -p D:\\software\\miniconda3\\envs\\qnet312 python -m unittest discover -s qnet_core/tests -q
conda run -p D:\\software\\miniconda3\\envs\\qnet312 python -m unittest discover -s algorithms/caappo/tests -q
conda run -p D:\\software\\miniconda3\\envs\\qnet312 python -m unittest discover -s algorithms/qddca/tests -q
conda run -p D:\\software\\miniconda3\\envs\\qnet312 python -m unittest discover -s qnet_core/qcast_paper/tests -q
conda run -p D:\\software\\miniconda3\\envs\\qnet312 python -m compileall -q algorithms qnet_core QCAST
conda run -p D:\\software\\miniconda3\\envs\\qnet312 python -m qnet_core.sequence_smoke

# Recreate from the repository declaration:
conda env create -f environment.yml
conda activate quantum-sim
```
