# Quantum Resource-Graph Batch Routing

Centralized batch entanglement routing for high-hop quantum networks.  The
controller builds candidate routes, validates their farthest executable
prefixes against concrete EPR resources, compiles swap DAGs, and uses masked
PPO to select a conflict-free batch.

The current high-contention benchmark uses 100 simultaneous requests balanced
across 20--29, 30--39, and 40--50 hops, with a fixed request TTL measured in
physical RELiQ subslots.

## Packages

- `batchswap_reliq`: RELiQ-backed physical environment and deterministic
  baselines.
- `batchswap_rl`: dynamic-set actor-critic, masked PPO, training, evaluation,
  and tests.

RELiQ is kept as an independent upstream checkout and is intentionally not
vendored into this repository. Place it at `RELiQ/` beside these packages
before running the physical backend.

## Test

```powershell
uv run --with numpy,networkx python -m unittest discover -v batchswap_reliq/tests
uv run --with torch,numpy python -m unittest discover -v batchswap_rl/tests
```

## High-contention fine-tuning

```powershell
python -m batchswap_rl.train `
  --backend reliq `
  --device cuda `
  --request-ttl 100 `
  --init-checkpoint batchswap_rl/runs/reliq_long_ladder_finetune/checkpoint.pt `
  --short-updates 0 `
  --medium-updates 0 `
  --long-updates 800 `
  --rollout-steps 1024 `
  --minibatch-size 32 `
  --output batchswap_rl/runs/reliq_100req_ttl100
```

See `batchswap_reliq/HIGH_HOP_REPORT.md` for the latest paired evaluation.
