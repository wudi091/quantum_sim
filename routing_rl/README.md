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
