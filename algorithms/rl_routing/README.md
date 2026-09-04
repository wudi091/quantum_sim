# ARC-Q

ARC-Q is the new construction-aware online routing method on the
`rl-routing` branch.  It treats one routing decision as an autoregressive
sequence of joint `(request, path, construction, start slot)` choices followed
by `STOP`.

The method has three deliberate properties:

1. A sparse graph represents topology, requests, candidate construction plans,
   and their resource--time incidence.
2. At every autoregressive step the environment exposes exactly the actions
   that remain physically feasible.  The policy itself chooses among them;
   there is no post-hoc greedy decoder.
3. The transition reward is the negative area under the unfinished-request
   curve.  With the terminal censoring charge, the episode return is exactly
   the negative mean censored completion time measured by the evaluator.

Planning code sees only immutable specifications, opaque resource keys, and
neutral physical snapshots.  Selected plans are executed by the existing
SeQUeNCe-backed lifecycle in `algorithms/routing_core/`.
