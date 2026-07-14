# BatchSwap on the RELiQ physical backend

This package is an independent adapter. It does not modify `RELiQ/` or
`QDDCA/`.

## Reused RELiQ components

- `QuantumNetwork`: topology, physical time, EPR refresh and decay;
- `Edge.links`: concrete elementary-EPR token inventory;
- `QuantumLink`: fidelity and creation state;
- `QuantumLink.swap`: RELiQ's fidelity/swap kernel.

The original `EntanglementEnv` is not used because its action is a per-request
next hop, it drops/resets requests, and its default mode performs the complete
swap chain at the destination in one environment step.

## BatchSwap execution

At every planning epoch the environment generates the first `N` shortest simple
paths from the current frontier to the destination in the physical topology. It
then validates each path against the current quantum resource graph, stopping at
the first edge without a usable EPR. The farthest feasible prefix of each path is
compiled into a concrete swap plan with fixed input tokens, swap nodes, expected
fidelity features, and DAG depth.

The centralized agent sequentially selects these `(request, swap plan)` actions.
`STOP` validates the whole selected plan set and atomically consumes its concrete
EPR inputs.

Swaps execute as a balanced DAG. Each DAG layer advances one RELiQ physical
subslot, during which unused elementary EPRs and request-owned EPRs decay and
the RELiQ network may generate new EPRs. A failed swap consumes its inputs, but
the request remains pending and restarts from its source.

## Dependencies and tests

```powershell
uv run --with torch,numpy,networkx,scipy,matplotlib,geopy,topohub `
  python -m unittest discover -v batchswap_reliq/tests
```

The deterministic P0 tests cover token double spending, ownership, atomic
validation, node capacity, swap depth, time advancement, generated-EPR
retention, and persistent requests.

## Train

```powershell
uv run --with torch,numpy,networkx,scipy,matplotlib,geopy,topohub `
  python -m batchswap_rl.train `
  --backend reliq `
  --output batchswap_rl/runs/reliq_short_convergence `
  --rollout-steps 256 --minibatch-size 64 --ppo-epochs 4 `
  --short-updates 100 --medium-updates 0 --long-updates 0
```

## Evaluate

```powershell
uv run --with torch,numpy,networkx,scipy,matplotlib,geopy,topohub `
  python -m batchswap_rl.evaluate `
  --backend reliq `
  --checkpoint batchswap_rl/runs/reliq_short_convergence/checkpoint.pt `
  --stage 0 --episodes 20 --seed 20000
```

## Current boundary

This is more physical than the aggregate prototype, but it is still a
simulator. Candidate generation currently keeps eight topology-shortest paths
per request; resource availability only determines each path's executable
prefix. Each evaluation group uses a cached topology so paired policies see
identical topology state. Medium/long-hop curriculum training remains future
work.
