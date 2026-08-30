# Scale-efficiency curve: GNN vs exact MILP decision time

| Case | Nodes | Violations | GNN decision (s) | MILP decision (s) | MILP/GNN speedup | GNN completed | MILP completed |
|---|---:|---:|---:|---:|---:|---:|---:|
| n48 | 48 | 0 | 0.0886 | 0.1017 | 1.15 | 19.9 | 19.9 |
| n64 | 64 | 0 | 0.0777 | 0.113 | 1.45 | 20.0 | 20.0 |
| n96 | 96 | 0 | 0.0819 | 0.1187 | 1.45 | 20.0 | 20.0 |
| n128 | 128 | 0 | 0.095 | 0.1434 | 1.51 | 20.0 | 20.0 |
| n192 | 192 | 0 | 0.097 | 0.143 | 1.47 | 19.9 | 19.9 |
| n256 | 256 | 0 | 0.0855 | 0.1325 | 1.55 | 20.0 | 20.0 |

All cases: 10 paired seeds, 20 requests, 4-hop, 4 paths, 5 construction plans, TTL 16, no physical noise, memory capacity 1, Waxman topology. MILP gap=0 with 300 s limit; no timeout was observed at this request load.