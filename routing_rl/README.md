# Routing RL

Masked PPO for the fixed shared SeQUeNCe environment in `qnet_core`.

```bash
python -m routing_rl.train --device cuda --output routing_rl/runs/default
python -m routing_rl.evaluate \
  --checkpoint routing_rl/runs/default/checkpoint.pt --episodes 20
```

No simulator/backend switch is exposed. Baselines and PPO share the same
request generation, physical steps, exchange execution, settlement, and
metrics.

## Formal large-scale training

The reviewed 2--50 hop configuration is stored in `routing_rl/large_scale.py`.
It continues from the small-scale GNN checkpoint and selects CUDA automatically
when a CUDA-enabled PyTorch installation is available:

```bash
python -m routing_rl.large_scale
```

The preset trains from scratch for at most 1,000 PPO updates with 512 rollout
steps (roughly 15,000 training episodes), evaluates the full range and 41--50
hop bucket every 10 updates, and writes checkpoints under
`results/gnn_large_scale_scratch_seed67001_u1000`.

The formal preset uses no initialization checkpoint. It stops after 20
consecutive evaluations (200 updates) without improvement, giving a scratch
policy enough time to cross a temporary plateau. Validate the resolved device
and configuration without training with:

```bash
python -m routing_rl.large_scale --check
```
