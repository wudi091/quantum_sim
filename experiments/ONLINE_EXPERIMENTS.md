# Long-Run Online Experiments

The protocol is defined in `online_experiments.yaml`. It contains five
experiments:

- `standard_stability`: one continuous run, reported in five time segments.
- `request_load`: requests per online run: 20, 50, 100, 200, 300.
- `network_scale`: node counts: 32, 64, 128, 192, 256.
- `topology_generalization`: unseen Waxman, BA, Cost266, Germany, Bellcanada.
- `physical_conditions`: independent sweeps of generation and swap success.

TopoHub points use the downloaded graph structure and the shared physical
configuration. Edge distances are retained in the source files but are not
silently converted into per-edge physical parameters by this protocol.

Every point is a long-running online episode. The summary records the primary
metric `mean_completion_delay_slots` (censored at the episode horizon), the
auxiliary metrics `max_completion_delay_slots`,
`mean_final_fidelity_loss`, `completion_delay_gini`, and
`planning_time_seconds`, plus `completed_requests` as a throughput supplement.
Raw comparison JSON files remain in each point directory for auditability.

Before launching jobs, inspect the expanded commands:

```bash
python -m experiments.run_online_experiments --dry-run
```

Run all configured points:

```bash
python -m experiments.run_online_experiments
```

Run one top-level experiment:

```bash
python -m experiments.run_online_experiments --experiment request_load
```

Generate figures from the newest completed run (the plotting command never
reruns experiments):

```bash
python -m experiments.plot_online_experiments
```

The default output is a `figures/` directory next to the selected
`experiment_summary.csv`.  To select a run explicitly and choose formats:

```bash
python -m experiments.plot_online_experiments \
  --input results/online_long_run/run_YYYYMMDD_HHMMSS/experiment_summary.csv \
  --output results/online_long_run/run_YYYYMMDD_HHMMSS/figures \
  --formats png,pdf,svg
```

One compact qnet_sim-style line-plot figure is written for each configured
experiment as `<experiment>/plots/sweep_comparison_ieee.*`. It uses equally
spaced sweep points, open markers, alternating line styles, inward ticks,
dotted grids, and a shared legend. Panels are emitted for mean completion
delay, maximum delay, fidelity loss, delay fairness, planning time, and the
throughput supplement when those columns are present. A
`figure_manifest.json` records the source summary and generated files.

The online comparison set is fixed to GNN, Q-CAST, Q-PASS, and Greedy. MILP is
kept as an offline teacher and is not executed by this long-run protocol.
