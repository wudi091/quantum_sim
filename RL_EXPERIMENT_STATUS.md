# CAAPPO RL Experiment Status

Date: 2026-08-07

The CAAPPO implementation is being trained as an RL policy over joint
`(path, construction)` candidates and event-epoch operation choices. Physical
GEN, SWAP, memory, fidelity, decoherence, and time are executed by SeQUeNCe.

## Reproducibility fix

SeQUeNCe 1.0.0 draws protocol outcomes from a process-global legacy NumPy RNG.
Each `SequenceBackend` now maps the 64-bit episode seed through SHA-256 to a
32-bit timeline seed before constructing the physical world. This makes
policy and baseline results independent of method execution order and supports
the namespaced 64-bit training episode seeds.

## Current training run

- config: `algorithms/caappo/configs/medium_train3.json`;
- training replicas: seeds 1, 2, and 3;
- episodes per replica: 300;
- validation interval: 25 episodes;
- config-reserved diagnostic evaluation seeds: 231 through 235;
- fresh extended frozen evaluation seeds: 301 through 330, supplied through
  the evaluation CLI override;
- checkpoint selection: best risk-feasible validation state;
- backend: SeQUeNCe 1.0.0, Bell-diagonal formalism, CPU PyTorch.

Best validation checkpoints occurred at episodes 250, 25, and 300. All three
reported completion rate 1.0 and risk count 0 on the three validation seeds.
The different best epochs show substantial optimization-speed variance, so
validation performance alone is not treated as convergence evidence.

## Historical medium-workload run

The original 30-seed medium-workload comparison was produced before the
executor-boundary DAG isolation fix. The trained-policy rows are retained as
diagnostic history, but its fixed-baseline rows are not authoritative because
reusing a candidate catalogue could contaminate later baseline executions.
Those numbers must not be used as the paper result.

## Parallel-corridor RL run

To test the joint path-selection part of the problem, the current workload has
two equal-length, node-disjoint corridors, two simultaneous requests, and
unit link memory capacity. Three independent CAAPPO replicas were trained for
400 episodes (training seeds 11, 12, 13); validation uses seeds 1101--1103 and
frozen evaluation uses seeds 1201--1230. The main variant includes the
candidate-specific route-overlap context. The `no_route_overlap` variant
removes only that context feature.

The fixed baselines below were regenerated after the DAG isolation audit fix;
each run constructs a pristine execution DAG and repeated evaluation is
idempotent.

Evaluation seeds are the independent units. The three trained replicas are
averaged within each evaluation seed before confidence intervals are computed.

| Method | Completion rate | Censored flow-time (ps) | Risk count |
|---|---:|---:|---:|
| CAAPPO, trained | 0.5389 [0.4241, 0.6536] | 14,491,677 [11,014,801, 17,968,554] | 0.9222 [0.6927, 1.1517] |
| CAAPPO, no route-overlap context | 0.1722 [0.1240, 0.2205] | 29,005,170 [28,725,454, 29,284,886] | 1.6556 [1.5591, 1.7520] |
| CAAPPO, no construction choice | 0.5722 [0.4396, 0.7048] | 14,307,678 [10,355,299, 18,260,057] | 0.8556 [0.5903, 1.1208] |
| CAAPPO, no route choice | 0.1444 [0.0982, 0.1907] | 28,892,614 [28,394,136, 29,391,092] | 1.7111 [1.6185, 1.8037] |
| CAAPPO, untrained | 0.3722 [0.2869, 0.4576] | 20,123,007 [17,710,423, 22,535,591] | 1.2556 [1.0849, 1.4262] |
| Shortest path + left-deep | 0.1500 [0.0666, 0.2334] | 28,317,170 [26,952,331, 29,682,008] | 1.7000 [1.5332, 1.8668] |
| Split paths + left-deep | 0.5000 [0.3590, 0.6410] | 15,055,510 [10,843,561, 19,267,459] | 1.0000 [0.7181, 1.2819] |
| Split paths + balanced | 0.5500 [0.4142, 0.6858] | 13,547,011 [9,486,059, 17,607,963] | 0.9000 [0.6284, 1.1716] |

Relative to shortest-left-deep, trained CAAPPO has completion-rate delta
`+0.3889` (95% CI `[+0.2368, +0.5410]`). Relative to the stronger
split-balanced heuristic, the delta is `-0.0111` (95% CI
`[-0.0996, +0.0774]`), so this experiment does not establish superiority over
that heuristic. Removing route-overlap context reduces completion by `0.3778`
relative to split-balanced (95% CI `[-0.5088, -0.2468]`).

Results are stored in `results/parallel-corridor-comparison30.json` and its
CSV companion. The twelve frozen checkpoint evaluations are stored beside it.
The trained-variant and fixed-baseline rows are independently reproducible via
the formal `compare` CLI output
`results/parallel-corridor-comparison30-formal.json`; the canonical table also
includes the separately generated untrained control rows.

## Claim boundary

The current parallel-corridor run supports a narrower claim: CAAPPO learns to
use candidate-specific batch route context, and the context ablation fails on
the same contention workload. The no-route-choice ablation also collapses to
the shortest corridor, while the no-construction-choice ablation remains
competitive on this particular workload. The learned policy is not better
than the hand-designed split-balanced policy within the current confidence
interval, and these results do not establish convergence or generalization
beyond this topology family. Larger topologies and additional workload
families remain required for a paper claim.

This is positive initial RL evidence, not a convergence claim and not yet a
paper-complete CCFA result.
