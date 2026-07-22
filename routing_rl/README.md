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
It trains from scratch and selects CUDA automatically when a CUDA-enabled
PyTorch installation is available.  Before it can start, run the fixed
full-range direction pilot:

```bash
python -m routing_rl.small_scale
```

The pilot compares the optimizer-step-zero model with the learned checkpoint
on fixed seeds.  Only a passing `direction_report.json` unlocks formal
training:

```bash
python -m routing_rl.large_scale
```

The preset trains from scratch on the full 2--50 hop distribution for at most
1,000 updates. It uses 512 rollout steps (roughly 15,000 training episodes at
the full budget), evaluates every 10 updates, and writes checkpoints under
`results/gnn_large_scale_scratch_seed67001_u1000`.

The formal preset uses no initialization checkpoint and no curriculum. Early
stopping is disabled for the first 300 updates; after that, 10 consecutive
evaluations without improvement stop the run. This prevents an accidental
early validation peak from terminating scratch learning. Validate the resolved
device and configuration without training with:

```bash
python -m routing_rl.large_scale --check
```
