# P1 generalization suite: 64-node-trained GNN on unseen scales/topologies

| Case | Topology | Nodes | Hops | Violations | GNN | Q-PASS | Greedy | GNN-Q-PASS (d, p) | GNN-Greedy (d, p) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| waxman128 | waxman | 128 | 3-5 | 0 | 111.95 | 111.8 | 111.95 | 0.15, 0.2478 | 0.0, 1.0 |
| waxman192 | waxman | 192 | 3-5 | 0 | 111.7 | 111.3 | 111.8 | 0.4, 0.0991 | -0.1, 0.6286 |
| waxman256 | waxman | 256 | 3-4 | 0 | 149.9 | 149.25 | 149.85 | 0.65, 0.075 | 0.05, 1.0 |
| ba128 | barabasi_albert | 128 | 3-5 | 0 | 111.4 | 103.55 | 108.4 | 7.85, 0.0001 | 3.0, 0.0035 |
| ba192 | barabasi_albert | 192 | 3-5 | 0 | 111.2 | 105.8 | 109.95 | 5.4, 0.0001 | 1.25, 0.0297 |
| ba256 | barabasi_albert | 256 | 3-5 | 0 | 111.75 | 106.45 | 110.35 | 5.3, 0.0001 | 1.4, 0.0107 |

Positive d always means the GNN is better. All cases are 20 paired seeds, 150 requests, no physical noise, memory capacity 1.