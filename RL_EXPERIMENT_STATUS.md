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

## Frozen 30-seed results

Evaluation seeds are the independent units. The three trained replicas are
averaged within each evaluation seed before confidence intervals are computed.

| Method | Completion rate | Censored flow-time (ps) | Risk count |
|---|---:|---:|---:|
| CAAPPO, trained | 0.8296 [0.7202, 0.9390] | 55,278,525 [22,391,560, 88,165,489] | 0.5111 [0.1830, 0.8393] |
| CAAPPO, untrained | 0.6037 [0.4984, 0.7090] | 112,257,185 [82,789,473, 141,724,896] | 1.1889 [0.8730, 1.5048] |
| Shortest path + left-deep | 0.6444 [0.5234, 0.7655] | 95,704,186 [63,964,830, 127,443,542] | 1.0667 [0.7035, 1.4298] |
| Shortest path + balanced | 0.6556 [0.5405, 0.7706] | 92,895,520 [62,650,167, 123,140,873] | 1.0333 [0.6883, 1.3784] |

Paired against shortest-left-deep, trained CAAPPO improves completion rate by
0.1852 with 95% CI [0.0775, 0.2929], reduces censored flow-time by 40,425,661 ps
with CI [12,410,173, 68,441,150] in absolute reduction, and reduces risk count
by 0.5556 with CI [0.2325, 0.8786]. Against balanced construction, the
completion-rate improvement is 0.1741 with CI [0.0662, 0.2819].

Results are stored in `results/caappo-medium-seeded-comparison30.json` and its
CSV companion. Checkpoints and generated results are intentionally ignored by
Git.

## Claim boundary

The policy clearly learned construction selection: across 90 frozen replica x
evaluation-seed episodes, 162 of 270 admissions selected balanced construction
and 108 selected left-deep; 64 episodes mixed construction types across the
batch. The corresponding untrained policies selected balanced only 34 times.

However, trained policies selected a non-shortest catalogue path only 3 times
out of 270 admissions. The current workload therefore supports a
construction-aware RL claim, but does not yet provide strong evidence for the
joint path-selection part of the paper problem. The next experiment must use
contention-aware topology/request distributions where an alternate route is
sometimes optimal, followed by route-choice and construction-choice
ablations.

This is positive initial RL evidence, not a convergence claim and not yet a
paper-complete CCFA result.
