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
- the event boundary is the minimum of operation completion, SeQUeNCe memory
  expiration, request arrival, deadline, and horizon; expiration is stamped at
  its physical time and deadlines appear as neutral boundary events. Arrival
  is intentionally a boundary-only transition because it changes eligibility
  but has no physical success/failure outcome.
- admission is autoregressive in canonical request order with a preview state
  and legal candidate mask; the reference CAAPPO also has a retry/drop repair
  head whose retry options are generated from the surviving DAG prefix.

The current SeQUeNCe capability declaration is intentionally conservative:
inter-epoch launch and mixed physical operation concurrency are rejected until
a separately validated scheduler exists. Multiple operations known to be
physically compatible are packed into one launch epoch.

The SeQUeNCe executor also rejects logical-only `initial_segments`; importing
an existing physical state requires a future explicit logical-to-physical pair
binding contract. The deterministic reference executor may still use logical
initial segments for contract tests.

Remaining paper-complete gates:

- richer repair generation for failed SWAP branches and an independently
  audited physical scheduler for inter-epoch/mixed operation concurrency;
- a high-performance autodiff PPO implementation with GAE and a constraint
  critic; the current trainer is a NumPy reference implementation;
- deterministic small-instance oracle, multi-seed confidence intervals,
  ablations, and the baseline experiment harness.

This is a runnable construction-aware event foundation and a NumPy reference
CAAPPO, not yet a complete CCFA paper system. The multi-pair evaluator,
request-level fidelity gate, and event-boundary semantics are implemented;
production-grade PPO, oracle-backed experiments, and statistically supported
paper claims remain outside the current claim.

Current regression commands, using the Conda environment
`D:\\software\\miniconda3\\envs\\qnet312` (Python 3.12.13, SeQUeNCe 1.0.0):

```text
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
