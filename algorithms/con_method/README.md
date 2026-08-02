# CON

CON is one algorithm with two stages:

- `offline_library/`: enumerate topology-wide paths and complete schedules,
  select a request-independent structural portfolio, and cache fixed 4x4 grids;
- `online_selector/`: read the immutable library and jointly select
  request-path-schedule candidates with GNN/RL.

Tests for both stages live under `tests/`. The shared environment never imports
this package; CON adapts its artifacts into `qnet_core` through public
contracts.

The topology-wide offline interface is:

```python
from algorithms.con_method.offline_library import (
    compile_waxman_topology_library,
    instantiate_con_library_for_episode,
)

compiled = compile_waxman_topology_library(
    topology_episode,
    generator="pareto",
    path_pool_per_pair=8,
    max_hops=6,
    output_path="artifacts/con/library.json",
)

candidates, valid_mask = compiled.library.lookup(source, destination)
online_episode = instantiate_con_library_for_episode(
    evaluation_episode,
    compiled.library,
)
```

The implemented topology-only generator presets are `canonical`, `quality`,
`banded`, `pareto`, and `exact_kcenter`.  They never read request IDs, arrival
slots, or request paths.  The old scenario-MILP constructor remains available
only as an experimental baseline; it is not the default compiler.

Every unordered node pair has one canonical cache entry.  Online lookup always
returns a fixed 16-slot (4 paths x 4 schedules) view; unavailable short-path
slots are `None` and are never filled by duplicating a real schedule.  Reverse
queries reuse the same IDs and mask while orienting paths toward the request.

Generator quality is evaluated by
`algorithms.con_method.benchmarks.run_generator_oracle_benchmark(...)`.  Every
generator receives the same topology.  The formal 100-request benchmark now
uses the same online `MilpReliableMemoryPathOrderPlanner` on every frozen
library.  It proves the optimum of a deterministic time-indexed abstraction
that accounts for current EPR inventory, per-link binomial reliable supply,
link buffers, BSM capacity, and schedule-dependent memory release.  Internal
start times and EPR assignments are feasibility certificates only; the
controller still emits one set of complete candidate IDs per slot.  This is a
model optimum, not a hidden-physics completion certificate.  The shared event
executor records stochastic physical completions separately under the same
hidden random stream.  The old `MilpStaticPathOrderPlanner` remains available
as an explicitly labeled resource relaxation.  The executor-validated
`MilpNominalPathOrderPlanner` remains available as a much slower exact
finite-scenario oracle for small snapshots only.  Neither MILP is part of
online deployment.
