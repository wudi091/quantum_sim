# TELGEN experimental figure preview

All plots are generated from raw paired-trial JSON reports. PDF and SVG are vector outputs; PNG files are inspection previews.

## Figure 1: fig01_construction_value

Construction-aware selection improves both nominal planning and physical completion over the strongest per-instance fixed-tree oracle.

- Data: `data\fig01_construction_value.csv`
- Sources: `formal_experiments\B1_construction_physical_100\construction_physical_validation.json`

## Figure 2: fig02_small_scale_throughput

On exactly solvable online instances, TELGEN closes most of the MILP completion gap and exceeds the routing baselines.

- Data: `data\fig02_small_scale_throughput.csv`
- Sources: `formal_experiments\B2_small_oracle\online_gnn_comparison.json`

## Figure 3: fig03_small_scale_latency

TELGEN lowers mean censored and tail completion latency relative to the routing baselines on small online instances.

- Data: `data\fig03_small_scale_latency.csv`
- Sources: `formal_experiments\B2_small_oracle\online_gnn_comparison.json`

## Figure 4: fig04_gnn_milp_quality

TELGEN tracks the per-instance MILP solution while retaining a high fraction of the exact solver's completed requests.

- Data: `data\fig04_gnn_milp_quality.csv`
- Sources: `formal_experiments\B2_small_oracle\online_gnn_comparison.json`

## Figure 5: fig05_decision_time

TELGEN is substantially faster than exact MILP, although lightweight routing heuristics remain faster.

- Data: `data\fig05_decision_time.csv`
- Sources: `formal_experiments\B2_small_oracle\online_gnn_comparison.json`

## Figure 6: fig06_topology_generalization_throughput

The frozen TELGEN model preserves its completion advantage on unseen Waxman and Barabási–Albert topologies.

- Data: `data\fig06_topology_generalization_throughput.csv`
- Sources: `formal_experiments\B3_waxman192\online_gnn_comparison.json`, `formal_experiments\B3_barabasi128\online_gnn_comparison.json`

## Figure 7: fig07_topology_generalization_latency

The frozen TELGEN model consistently reduces mean censored latency and remains comparable in P95 completion latency across unseen topologies.

- Data: `data\fig07_topology_generalization_latency.csv`
- Sources: `formal_experiments\B3_waxman192\online_gnn_comparison.json`, `formal_experiments\B3_barabasi128\online_gnn_comparison.json`

## Figure 8: fig08_load_throughput

Construction-aware routing maintains higher completion rate and throughput as offered load increases.

- Data: `data\fig08_load_throughput.csv`
- Sources: `formal_experiments\B4_load_low_50\online_gnn_comparison.json`, `formal_experiments\B4_load_medium_100\online_gnn_comparison.json`, `formal_experiments\B4_load_high_150\online_gnn_comparison.json`

## Figure 9: fig09_load_latency

TELGEN reduces mean censored latency as contention increases, while P95 completion latency remains comparable rather than uniformly dominant.

- Data: `data\fig09_load_latency.csv`
- Sources: `formal_experiments\B4_load_low_50\online_gnn_comparison.json`, `formal_experiments\B4_load_medium_100\online_gnn_comparison.json`, `formal_experiments\B4_load_high_150\online_gnn_comparison.json`

## Figure 10: fig10_construction_ablation

Adaptive construction outperforms every fixed swap tree, while gains increase with the number of available construction candidates.

- Data: `data\fig10_construction_ablation.csv`
- Sources: `formal_experiments\B5_adaptive_5\online_gnn_comparison.json`, `formal_experiments\B5_fixed_tree_0\online_gnn_comparison.json`, `formal_experiments\B5_fixed_tree_1\online_gnn_comparison.json`, `formal_experiments\B5_fixed_tree_2\online_gnn_comparison.json`, `formal_experiments\B5_fixed_tree_3\online_gnn_comparison.json`, `formal_experiments\B5_fixed_tree_4\online_gnn_comparison.json`, `formal_experiments\B5_candidates_1\online_gnn_comparison.json`, `formal_experiments\B5_candidates_3\online_gnn_comparison.json`
