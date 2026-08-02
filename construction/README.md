# Construction-Aware Routing Kernel

This package is the deterministic Phase 1 implementation of `con_design.md`.
It is intentionally independent from `qnet_core`: existing SeQUeNCe
baselines and slot execution are unchanged.

## Model

A routing candidate is `(P, C)`, where `P` is a physical path and `C`
contains:

- ordered elementary-pair generation layers (operations in one layer run
  in parallel);
- a swap dependency tree stored in topological execution order;
- a construction family: sequential (`seq`), balanced (`bal`), or an
  intermediate variant (`mid`).

Within one centralized slot, a construction's node cost is the peak
number of generation operations incident on that node in any layer.
Selected requests reserve these peak footprints additively. Nodes omitted from
the capacity map have zero capacity. This gives the `con_design.md` counterexample
directly: on A-B-C-D-E, `C_seq` generates BC and CD in different layers and
costs 1 memory at C, while `C_bal` generates them together and costs 2.

Elementary generation is deterministic in Phase 1. Probabilistic EPR
generation, fidelity, decoherence, and cross-slot accumulation remain
Phase 2 work.

## Run

```bash
python -m construction.reproduce_con_md
python -m construction.reproduce_intraslot_generation
python -m construction.reproduce_intraslot_milp
python -m construction.reproduce_batch_plan_milp
python -m unittest discover -s construction/tests -v
```

## Automatic intra-slot EPR generation experiment

`intraslot_simulator.py` is a separate event kernel for the swap-order memory
experiment.  The controller selects each request's path and complete linear
swap order once.  During the slot, the environment automatically performs a fixed
number of Bernoulli EPR-generation rounds whenever both endpoint nodes have a
free memory slot.  Ready swaps then execute in the selected order; released
memory becomes available to the next generation round.

The deterministic demo (`p_gen = p_swap = 1`) reproduces:

- `B -> C -> D`, `M_C = 2`: 2/3 requests complete;
- `C -> B -> D`, `M_C = 2`: 3/3 requests complete;
- `M_C = 4`: both orders complete 3/3.

Generation timing and outcomes are environment dynamics, not planner actions.

The module intraslot_order_milp.py is a time-indexed 0-1 MILP oracle for the
deterministic three-request counterexample. It selects only R1's complete swap
order; fixed simulator-derived hotspot occupancy and BSM profiles determine
when the automatic priority executor can finish R2 and R3. It proves that
C -> B -> D and C -> D -> B are the two global optima for M_C = 2, each
completing 3/3 requests. This small model is deliberately not claimed as an
exact formulation of the later stochastic SeQUeNCe environment.

The module batch_plan_milp.py generalizes the same idea to joint request
admission, candidate-path selection, and complete linear swap-order selection.
Variables select a complete candidate and a feasible physical start window;
time-shifted memory, BSM, and exclusive-current-EPR constraints enforce the
deterministic resource model. Optimized start windows make the generic model
a strong scheduling oracle unless fixed-executor precedence constraints or a
post-solve simulator validation are also applied.

Both SciPy/HiGHS MILP oracles explicitly require zero relative MIP gap and
reject a successful return unless every available gap and primal/dual-bound
certificate is closed. They never label a positive-gap incumbent as exact.

The exact planner requires OR-Tools, included in the root
`requirements.txt`. It returns only after CP-SAT proves optimality; if the
configured time limit expires first, it raises `TimeoutError` rather than
returning a feasible incumbent as an exact result.

## Components

- `plan.py`: immutable construction-plan and swap-dependency-tree structures.
- `enumerator.py`: sequential, balanced, and intermediate candidates.
- `simulator.py`: swap-dependency-tree validation and peak-memory footprints.
- `planners.py`: greedy same-slot admission.
- `cpsat.py`: exact maximum-throughput batch selection.
- `reproduce_con_md.py`: executable `con_design.md` §3-4 counterexample.
