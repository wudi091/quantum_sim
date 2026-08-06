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

The current SeQUeNCe capability declaration is intentionally conservative:
inter-epoch launch and mixed physical operation concurrency are rejected until
a separately validated scheduler exists. Multiple operations known to be
physically compatible are packed into one launch epoch.

The SeQUeNCe executor also rejects logical-only `initial_segments`; importing
an existing physical state requires a future explicit logical-to-physical pair
binding contract. The deterministic reference executor may still use logical
initial segments for contract tests.

Remaining paper-complete gates:

- demand_pairs/multi-lane end-to-end delivery in the construction evaluator;
- one unified event queue for arrivals, deadlines, expiration, and online repair;
- a high-performance autodiff PPO implementation with GAE and a constraint
  critic; the current trainer is a NumPy reference implementation;
- deterministic small-instance oracle, multi-seed confidence intervals,
  ablations, and the baseline experiment harness.

This is a runnable construction-aware event foundation and a NumPy reference
CAAPPO, not yet a complete CCFA paper system. In particular, multi-pair
delivery, unified online arrivals/expiration/repair, and production-grade PPO
remain outside the current claim.

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
