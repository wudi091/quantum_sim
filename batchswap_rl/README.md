# BatchSwap-RL

Pure reinforcement-learning implementation of centralized batch entanglement
swapping. It does not use MILP labels, Oracle actions, behavior cloning, or
supervised pretraining.

## Decision process

At each planning epoch, the environment exposes up to three feasible frontier
extension plans for every active request: maximum reach, half reach, and one-hop
reach. A centralized masked-PPO policy repeatedly selects a `(request, plan)`
action or `STOP`.

After every plan selection, the action mask removes:

- all other plans of that request;
- plans that would consume the same elementary EPR capacity;
- plans that would exceed node swap capacity.

`STOP` commits the selected set atomically. Plan-selection microsteps have zero
physical duration. Batch execution advances time by the maximum balanced
swap-tree depth in the selected batch.

Requests are persistent by default: unavailable or conflicting requests wait
in place and are never rejected by the policy.  The RELiQ backend can
optionally impose an exogenous fixed lifetime with `--request-ttl`; an
unresolved request then becomes a timeout failure after that many physical
steps.  The TTL is identical for all requests and is independent of hop count.

## State and objective

The actor receives fixed-size global, request, and candidate-plan features. The
critic pools the complete request/candidate sets, so the controller is
centralized even though the action set is dynamic.

The base reward minimizes normalized flow time and makespan, with optional EPR
and swap costs. State-only potential shaping uses aggregate remaining hops and
does not require an expert policy. PPO uses duration-aware semi-Markov GAE.

## Train

```powershell
uv run --with torch,numpy python -m batchswap_rl.train `
  --output batchswap_rl/runs/default `
  --device cpu
```

The optional `batchswap_reliq` backend uses the same observation/action
contract through a thin adapter. It is selected without changing PPO code:

```powershell
uv run --with torch --with numpy python -m batchswap_rl.train `
  --backend reliq --request-ttl 30 --output batchswap_rl/runs/reliq_ttl30
```

The backend must provide `batchswap_reliq.env.make_env(stage, seed)`. Its
candidate mask uses the same convention as this package: legal candidates are
`True` and STOP is the final action. Plan-selection steps report
`duration=0`; STOP reports the physical swap-DAG duration.

Quick integration test:

```powershell
uv run --with torch,numpy python -m batchswap_rl.train `
  --smoke --output batchswap_rl/runs/smoke
```

## Evaluate

Evaluation uses identical keyed EPR traces for learned PPO, greedy,
Q-DDCA-style one-hop, and random-valid controllers.

```powershell
uv run --with torch,numpy python -m batchswap_rl.evaluate `
  --checkpoint batchswap_rl/runs/default/checkpoint.pt `
  --stage 2 --episodes 20 `
  --output batchswap_rl/runs/default/evaluation_long.json
```

Use `--backend reliq` to evaluate the same checkpoint against the RELIQ-backed
environment. Learned, greedy, Q-DDCA, and random-valid policies are evaluated
on the same seed list. With TTL enabled, the primary metrics are completion,
timeout rate, TTL-capped delay, successful-request delay, and episode end time:

```powershell
uv run --with torch,numpy python -m batchswap_rl.evaluate `
  --backend reliq --request-ttl 30 `
  --checkpoint batchswap_rl/runs/reliq_ttl30/checkpoint.pt `
  --stage 2 --episodes 20
```

## Test

```powershell
uv run --with torch,numpy python -m unittest discover -v batchswap_rl/tests
```

## Modeling boundary

This first implementation uses aggregate per-edge elementary-EPR inventory and
an implicit, request-owned source-to-frontier virtual segment. It prevents
cross-request reuse and same-batch double spending, but it is not a token-level
physical simulator. Results must therefore be described as performance under
the aggregate resource-graph abstraction. A strict token-level environment is
required before making hardware-level fidelity or wall-clock claims.
