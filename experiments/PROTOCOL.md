# ARC-Q experiment protocol

The formal protocol contains five long-running online sweeps: offered load,
network scale, node memory, elementary-link generation probability, and
unseen-topology generalization. Every point uses paired episodes across three
topology seeds and ten disjoint request/physical seeds.

The primary metric is mean censored completion latency. Completion rate,
completion-delay Gini coefficient, and end-to-end planning time are supporting
metrics. ARC-Q planning time includes candidate generation, resource--time
expansion, graph construction, and actor inference, but excludes its
training-only critic. Schedule violations and physical-backend rejections are
validity checks and must remain zero.

The comparison set is ARC-Q, Greedy, Path-only, Construction-only, Q-LEAP,
and Q-CAST. LP and MILP are not online baselines. The trained checkpoint is
selected only on the held-out validation traces configured for training; the
formal experiment seeds must not be used for model selection.

Formal evaluation accepts only `arcq_best.pt` after the complete training run
has finalized model selection. It rejects a checkpoint when the action-space
configuration differs from training, when a formal request seed overlaps a
training or validation seed, or when a formal topology seed equals the fixed
training topology seed.

Experiment execution records raw JSON and CSV data and supports resuming at
an episode boundary. Plotting is a separate command that only reads those
files. Development smoke tests and pilot checkpoints are not paper evidence.

The plotting command rejects incomplete paired suites and any result that
fails the validity checks. It exports per-suite PDF/PNG line figures plus CSV
and JSON source data with two-sided 95% Student-t confidence intervals. The
paired-difference table uses a positive sign when ARC-Q is better. Plotting
never invokes the simulator or experiment runner.

Every raw result file is bound to the protocol fingerprint, checkpoint hash,
finalized checkpoint provenance, Git revision, tracked-worktree state, Python
version, and core dependency versions. Formal execution rejects uncommitted
tracked code changes by default.
