# Quantum Network Simulation

This branch is the clean starting point for a construction-aware online
reinforcement-learning router.

## Layout

- `qnet_core/`: simulator-facing specifications, resource accounting,
  construction execution, and SeQUeNCe integration.
- `algorithms/routing_core/`: reusable path/construction candidate expansion,
  time-resource footprints, packing helpers, physical validation, and the
  shared persistent execution lifecycle.
- `algorithms/qcast/`: Q-CAST baseline and its online planner.
- `algorithms/baselines/`: non-learning routing baselines.
- `algorithms/rl_routing/`: the new reinforcement-learning method.
- `configs/`: fixed ARC-Q training configurations.
- `data/reliq_topologies/`: topology inputs used by simulator experiments.

## Environment

```bash
conda env create -f environment.yml
conda activate quantum-sim
python -m qnet_core.sequence_smoke
python -m pytest -q
```

Experiment runners and generated results from the previous method were
removed. ARC-Q uses no LP, MILP, supervised teacher, or post-hoc decoder.
Its method contract is documented in
`algorithms/rl_routing/METHOD.md`.

Run the deterministic end-to-end training smoke test with:

~~~bash
python -m algorithms.rl_routing.train --config configs/arcq_smoke.yaml
~~~

The fixed single-topology training configuration is
`configs/arcq_train.yaml`. Training records measurements only; future
plotting commands must read those recorded files rather than rerun training.
