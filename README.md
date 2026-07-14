# Quantum Resource-Graph Batch Routing

Centralized batch entanglement routing for high-hop quantum networks.  The
controller builds candidate routes, validates their farthest executable
prefixes against concrete EPR resources, compiles swap DAGs, and uses masked
PPO to select a conflict-free batch.

The current high-contention benchmark uses 100 simultaneous requests balanced
across 20--29, 30--39, and 40--50 hops, with a fixed request TTL measured in
physical RELiQ subslots.

The PPO reward contains no expert action, routing priority, or hop-specific
bonus. It uses request-count/TTL-normalized completion, timeout, and actual
flow time, plus duration-aware potential shaping for credit assignment.

## Packages

- `batchswap_reliq`: RELiQ-backed physical environment and deterministic
  baselines.
- `batchswap_rl`: dynamic-set actor-critic, masked PPO, training, evaluation,
  and tests.
- `RELiQ`: vendored pristine physical simulator and original RELiQ code.
- `QDDCA`: vendored pristine official Q-DDCA prototype for algorithm
  comparison.

The repository is self-contained: RELiQ and Q-DDCA are included at their
recorded upstream commits. See `THIRD_PARTY.md` for provenance and licenses.

## Test

Install CUDA-enabled PyTorch separately, then install the simulator dependencies:

```bash
pip install -r requirements.txt
```

Run the tests:

```bash
python -m unittest discover -v batchswap_reliq/tests
python -m unittest discover -v batchswap_rl/tests
```

## High-contention training

```bash
python -m batchswap_rl.train \
  --backend reliq \
  --device cuda \
  --request-ttl 100 \
  --short-updates 200 \
  --medium-updates 400 \
  --long-updates 800 \
  --rollout-steps 1024 \
  --minibatch-size 32 \
  --learning-rate 1e-4 \
  --output batchswap_rl/runs/reliq_100req_ttl100
```

To fine-tune an existing compatible model, add
`--init-checkpoint /path/to/checkpoint.pt --reset-critic` and set the
short/medium update counts to zero. Resetting the critic is required when the
checkpoint was trained with the legacy unnormalized reward.

See `batchswap_reliq/HIGH_HOP_REPORT.md` for the latest paired evaluation.
