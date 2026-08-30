# P0 evaluation suite: construction-aware routing vs. baselines

Paired-trial comparison of the frozen online GNN against Q-PASS and Greedy.
Positive advantage always means the GNN is better; for latency and decision
time the signed difference is negated before testing.

## Load sweep (64-node Waxman, no physical noise)

| Case | Trials | GNN completed | Q-PASS completed | Greedy completed | GNN vs Q-PASS (95% CI, p) | GNN vs Greedy (95% CI, p) |
|---|---:|---:|---:|---:|---:|---:|
| L050 | 20 | 50.00 | 49.85 | 50.00 | +0.15 [+0.00, +0.30], p=0.2519 | +0.00 [+0.00, +0.00], p=1.0000 |
| L100 | 20 | 97.70 | 94.05 | 94.30 | +3.65 [+2.85, +4.45], p=0.0001 | +3.40 [+2.15, +4.75], p=0.0001 |
| L150 | 20 | 132.00 | 120.90 | 118.05 | +11.10 [+9.50, +12.60], p=0.0001 | +13.95 [+12.35, +15.45], p=0.0001 |
| L200 | 20 | 154.20 | 139.75 | 129.75 | +14.45 [+12.50, +16.30], p=0.0001 | +24.45 [+22.80, +26.05], p=0.0001 |
| L300 | 20 | 178.75 | 158.05 | 142.95 | +20.70 [+19.15, +22.30], p=0.0001 | +35.80 [+33.55, +38.10], p=0.0001 |

## Noise sweep (150 requests, 64-node Waxman)

| Case | Generation / swap | Trials | GNN completed | Q-PASS completed | Greedy completed | GNN vs Q-PASS (95% CI, p) | GNN vs Greedy (95% CI, p) |
|---|---|---:|---:|---:|---:|---:|---:|
| N_low | 0.9 / 0.9 | 20 | 81.45 | 70.00 | 68.90 | +11.45 [+10.00, +12.85], p=0.0001 | +12.55 [+10.10, +14.95], p=0.0001 |
| N_mid | 0.7 / 0.8 | 20 | 24.85 | 21.05 | 20.80 | +3.80 [+1.65, +5.90], p=0.0040 | +4.05 [+2.35, +5.85], p=0.0002 |
| N_high | 0.5 / 0.7 | 20 | 4.20 | 4.20 | 3.40 | +0.00 [-0.85, +0.85], p=1.0000 | +0.80 [+0.10, +1.60], p=0.0728 |
