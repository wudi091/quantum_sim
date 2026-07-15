# Quantum Resource-Graph Routing

This repository uses one fixed SeQUeNCe physical environment for every
algorithm. Request generation, EPR generation, physical step advancement,
memory lifetime, exchange execution, conflict validation, TTL settlement,
rewards, and metrics are implemented once in `qnet_core`.

PPO, Q-DDCA, Greedy, and Random only select routing/exchange plan IDs from the
same immutable planning snapshot. They cannot create requests, mutate EPR
resources, advance time, reject requests, or perform settlement themselves.

## Packages

- `qnet_core`: shared SeQUeNCe kernel, scenario generation, candidates,
  settlement, metrics, and planning-only baselines.
- `routing_rl`: masked PPO model, rollout, training, and paired evaluation.
- `QDDCA`: pristine upstream Q-DDCA source retained as the algorithm reference;
  its simulator is not used as an experiment backend.

## Environment

SeQUeNCe 1.0 requires Python 3.12 or newer. Install CUDA-enabled PyTorch for
your server separately, then install the project dependencies:

```bash
pip install -r requirements.txt
```

Validate the physical kernel and shared environment:

```bash
python -m qnet_core.sequence_smoke
python -m unittest discover -v qnet_core/tests
python -m unittest discover -v routing_rl/tests
```

Reproduce the Q-DDCA official throughput, timeout/drop, congestion-window,
rerouting, and fairness trends on the shared SeQUeNCe environment:

```bash
python -m qnet_core.qddca_trends \
  --seeds 3 --output results/qddca_sequence_trends_3seed.json
python -m qnet_core.plot_qddca_trends
```

## Train

There is no backend option. Training always uses SeQUeNCe.

```bash
python -m routing_rl.train \
  --device cuda \
  --request-ttl 32 \
  --long-requests 20 \
  --generation-probability 0.5 \
  --swap-probability 0.5 \
  --memory-capacity 2 \
  --output routing_rl/runs/sequence_default
```

## Fair evaluation

```bash
python -m routing_rl.evaluate \
  --checkpoint routing_rl/runs/sequence_default/checkpoint.pt \
  --episodes 20 \
  --requests 20 --min-hops 20 --max-hops 50 \
  --request-ttl 32 \
  --output routing_rl/runs/sequence_default/evaluation.json
```

PPO, Q-DDCA, Greedy, and Random receive the same episode seeds and therefore
the same topology, request set, and physical random process.
