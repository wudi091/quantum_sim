# ARC-Q theoretical contract

ARC-Q means **Autoregressive Routing and Construction for Quantum Networks**.
The statements below describe properties of the implemented decision process;
they do not claim that PPO finds a globally optimal policy.

## Model and assumptions

At one decision boundary there is a finite set of joint candidates. A
candidate specifies one arrived request, one path, one construction DAG, and
one start slot. Its resource demand is a finite collection of non-negative
integer amounts indexed by opaque resource--slot keys. Existing reservations
are fixed during the autoregressive decision. STOP commits the selected set
and advances the physical simulator.

The planning feasibility contract has two constraints: at most one candidate
per request, and reserved plus selected demand no greater than capacity for
every resource--slot key. Physical generation and swapping success are not
planning constraints; SeQUeNCe resolves them after commitment.

The policy uses one permutation-equivariant graph encoder for the current
state, followed by separate actor and critic heads. The encoder is trained by
both PPO losses; the actor head selects actions and the critic head estimates
continuation value. This parameter sharing changes optimization efficiency but
does not change the feasible action set or any of the propositions below.

## Proposition 1: finite feasible output

Every ARC-Q decision terminates after at most one plus the number of eligible
requests, and the committed plan satisfies the planning feasibility contract.

Reason. STOP is legal at every prefix. Every non-STOP action consumes a
previously unselected request, so only finitely many non-STOP actions are
possible. The empty prefix is feasible. Before adding a candidate, the mask
checks request uniqueness and every resource--slot residual capacity. Because
all demands are non-negative, this preserves feasibility inductively. STOP
does not change usage, so the final selected set remains feasible.

This guarantee is about the submitted schedule. It does not guarantee that a
stochastic physical attempt succeeds. Schedule violations and backend
rejections are separately checked during evaluation.

## Proposition 2: no feasible candidate set is removed

For every set that satisfies the planning feasibility contract, ARC-Q has an
action sequence that emits exactly that set followed by STOP.

Reason. Take any ordering of the set. Each prefix contains no repeated request.
Its resource demand is component-wise no larger than the complete feasible
set because demands are non-negative. Therefore every next member remains
legal, and STOP emits the desired set. The autoregressive factorization changes
the probability model over feasible sets but not the represented feasible
set space.

## Proposition 3: exact delay-aligned episode return

For a request arriving at time `a`, define its censored settlement time as its
successful completion time or the episode horizon `H` if it does not complete.
Its censored latency is settlement time minus `a`. The area under that
request's unfinished indicator from `a` to settlement is exactly this latency.

ARC-Q charges, at every physical transition, the unfinished-request area in
that interval. If a request expires before `H`, a terminal charge covers the
remaining interval from expiry to `H`. Summing the interval areas and terminal
charges therefore gives the sum of censored request latencies. Dividing by the
fixed episode request count and negating proves:

    episode return = -mean censored completion latency in slots

The evaluator recomputes the right-hand side from physical settlements after
every rollout. A non-zero identity error invalidates the run. Discount factor
one is required; any smaller value would optimize a different objective.

## Proposition 4: duration-aware credit is invariant to intra-plan position

Ordinary token-level GAE would multiply feedback by its trace factor once for
every autoregressive token. It would therefore weaken the same physical
feedback merely because a plan contains more selected requests.

ARC-Q instead assigns trace factor one after every zero-time candidate action
and uses the configured physical trace factor only after STOP advances the
simulator. Across consecutive zero-time actions, the GAE recursion therefore
contains no multiplicative attenuation. The weight of later physical feedback
depends on the number of physical decision intervals, not on how many
candidates precede STOP or on a candidate's position inside the same plan.

This property removes a factorization-induced plan-length bias. A physical
trace factor below one can still introduce the usual GAE bias--variance tradeoff
across real decision intervals; it does not change the environment reward.

## Proposition 5: node-order equivariance

The graph encoder applies shared node transforms, relation-specific shared
message transforms, permutation-symmetric aggregation, and mean pooling. If
the tensor order of graph nodes is permuted while edges and candidate indices
are permuted consistently, node embeddings and candidate logits undergo the
same permutation, while the global context and STOP logit remain unchanged.

This removes dependence on arbitrary tensor or node-label order. It is an
inductive bias for transfer, not a proof that performance will generalize to
unseen topology sizes or families; that claim requires the held-out topology
experiments.

## Proposition 6: neutral plan-cardinality exploration

Suppose a prefix has `m` mutually compatible requests, every request still has
at least one legal candidate, and the top-level STOP and request logits are
equal. ARC-Q assigns probability `1 / (m + 1)` to every possible final plan
cardinality from zero through `m`.

Reason. The probability of selecting exactly `k < m` requests is the product
of continuing while `m, m-1, ..., m-k+1` requests remain and then selecting
STOP. The factors telescope to `1 / (m + 1)`. Selecting all `m` requests has
the same telescoping probability, after which STOP is forced. Low-level path,
construction, and start-slot counts do not enter this calculation because
each request owns exactly one top-level branch.

Resource conflicts can remove several request branches after one selection,
so the uniform-cardinality statement does not apply to every constrained
state. The structural guarantee that candidate multiplicity does not inflate
a request's top-level probability still applies.

## Explicit limits

- Candidate generation bounds the routes and construction DAGs considered;
  ARC-Q is complete only relative to that generated candidate set.
- Feasibility masking guarantees declared resource--time capacity, not
  stochastic physical success or optimality.
- Analytical success and fidelity values are observable estimates. Final
  outcomes and reported performance come from SeQUeNCe execution.
- PPO is an approximate policy optimizer. Learning value and generalization
  must be established empirically with frozen checkpoints and disjoint seeds.
