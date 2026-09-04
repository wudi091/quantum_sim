# ARC-Q

ARC-Q (Autoregressive Routing and Construction for Quantum Networks) is the
new construction-aware online routing method on the
`rl-routing` branch.  It treats one routing decision as an autoregressive
sequence of joint `(request, path, construction, start slot)` choices followed
by `STOP`.

The method has three deliberate properties:

1. A sparse graph represents topology, requests, candidate construction plans,
   and their resource--time incidence.
2. At every autoregressive step, one learned categorical decision chooses
   STOP or one currently feasible request. Conditional heads then choose that
   request's path, construction, and start slot. A request therefore receives
   one top-level alternative regardless of how many joint candidates it owns.
   There is no post-hoc greedy decoder.
3. The transition reward is the negative area under the unfinished-request
   curve.  With the terminal censoring charge, the episode return is exactly
   the negative mean censored completion time measured by the evaluator.

Candidate choices and STOP are PPO transitions in an augmented decision
process. Candidate transitions consume no physical time and receive zero
reward; STOP advances SeQUeNCe and receives the interval reward. This gives
the critic a value for every partial feasible plan without changing the
physical objective. Actor and critic graph encoders are independent and their
gradients are clipped separately, so value regression cannot scale away the
policy update.

ARC-Q uses a duration-aware GAE trace: candidate choices consume zero physical
time and therefore use trace factor one, whereas STOP transitions use the
configured physical trace factor. Credit is not weakened merely because a
feasible plan contains more selected requests.

With equal top-level logits and no newly induced conflicts, the initial
cardinality of a plan is uniform from zero through the number of feasible
requests. This avoids the short-plan exploration collapse produced by an
independent 0.5 STOP hazard while adding no hand-written selection rule.

Planning code sees only immutable specifications, opaque resource keys, and
neutral physical snapshots.  Selected plans are executed by the existing
SeQUeNCe-backed lifecycle in `algorithms/routing_core/`.

The exact guarantees and their limitations are recorded in `THEORY.md`.
