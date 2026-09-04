# ARC-Q method contract

## Research problem

At each online decision boundary, the controller sees arrived but unfinished
entanglement requests and the neutral physical state. A decision jointly
chooses a request, a path, an ordered swap construction, and a start slot.
The selected requests then execute in SeQUeNCe; generation, swapping,
purification, decoherence, and failure are physical events rather than planner
assumptions.

The primary objective is mean censored completion time. Completion rate,
completion-delay fairness, fidelity, and planning time are reported separately
and are not mixed into a hand-tuned reward.

## State representation

One sparse heterogeneous graph contains four node types:

- physical network nodes;
- arrived unfinished requests;
- joint path--construction--start candidates;
- only the resource--slot pairs touched by a candidate or reservation.

Edges encode topology, request endpoints, candidate ownership, route
membership, candidate resource usage, and the physical owner of each opaque
resource. Live logical segments, fidelity, age, in-flight work, residual
capacity, request age, deadline, construction depth, and success estimates
enter as neutral features. No SeQUeNCe object crosses this boundary.

The graph size follows the current workload rather than a fixed topology.
Shared message-passing parameters and symmetric aggregation make the network
compatible with unseen node labels, request counts, and topology sizes. This
is an architectural inductive bias; cross-topology generalization remains an
experimental claim that must be tested.

## Action representation

The policy creates one plan autoregressively:

1. score every currently legal joint candidate and STOP;
2. sample or select one action;
3. if a candidate is chosen, subtract its resource--time footprint and remove
   every other candidate for the same request;
4. recompute graph features and scores;
5. terminate only when the policy chooses STOP.

The environment mask contains only hard conditions already present in the
execution contract: request uniqueness and residual resource--slot capacity.
It neither ranks candidates nor adds, repairs, or removes a selected plan
after inference. Thus the sampled action itself is the executable discrete
decision; there is no post-hoc decoder.

Every emitted plan is feasible because the empty prefix is feasible and each
accepted autoregressive extension is checked against the residual capacity.
Conversely, every feasible candidate set has at least one ordering whose
successive prefixes are feasible, so the factorization does not remove a
feasible plan from the action space.

## Delay-aligned reward

For each physical interval, let the unfinished-request area be the sum of time
for which arrived requests remain unsettled. The transition reward is the
negative of this area, divided by the episode request count. If a request is
censored before the horizon, a one-time terminal charge covers the remaining
time to the horizon.

Across a complete episode:

    negative cumulative reward = mean censored completion time in slots

This follows by exchanging the order of summation: each request contributes
one unit to the unfinished count from arrival until successful completion, or
until censoring plus its terminal charge. The implementation checks this
identity after every rollout. PPO therefore uses discount factor 1; allowing
a smaller factor would silently change the paper objective and is rejected by
configuration validation.

## Learning algorithm

The actor and critic use independent relation-aware graph encoders. This keeps
the much larger value-regression gradient from suppressing the combinatorial
policy gradient; their gradients are clipped separately. Candidate heads
produce categorical logits, while a global head scores STOP. Planning is
treated as an augmented Markov process: selecting a candidate changes only
the residual-capacity graph and yields zero immediate reward; selecting STOP
advances the physical environment and receives the exact interval reward.
PPO therefore optimizes each masked categorical choice rather than multiplying
an entire variable-length plan into one probability ratio. The critic values
every partial feasible plan, generalized advantage estimation propagates the
physical reward through the zero-time planning steps, and gamma remains one,
so autoregressive factorization does not introduce an artificial time cost or
change the episode objective.

The critic is used only while collecting training rollouts and updating PPO.
Frozen online evaluation executes the actor alone, so reported planning time
does not include a training-only value network.

Training varies request traces and physical randomness while holding one graph
topology fixed. Evaluation must use disjoint request seeds and include both
unseen instances of the training topology family and unseen topology families.
LP, MILP, and supervised labels are not used in training or online inference.

## Paper claims and gates

The implementation can support the following claims only after their gates
pass:

1. Objective correctness: reward identity error is numerically zero on
   deterministic, stochastic, failure, expiry, and rolling-arrival tests.
2. Execution reliability: no resource oversubscription, backend rejection, or
   schedule-contract violation is produced by ARC-Q.
3. Learning value: a frozen trained policy improves mean censored completion
   time over the same untrained architecture on held-out traces.
4. Construction value: ARC-Q beats a path-only method with fixed construction
   and a construction-only method with fixed path under
   construction-sensitive contention.
5. Generalization: a model trained on one topology retains its advantage on
   unseen topology sizes and families without fine-tuning.
6. Online viability: decision latency remains bounded as topology, requests,
   candidate paths, and construction choices scale.

Novelty status is currently unsearched. No paper should claim that the graph
factorization, masked PPO, or reward identity is individually new until a
current related-work search is completed. The intended contribution is their
problem-specific interaction: exact delay alignment plus a
feasibility-preserving joint path--construction action space under real
physical feedback.
