# Algorithms

`algorithms/routing_core/` contains shared planning data structures and the
SeQUeNCe-backed execution lifecycle. It does not contain a learning method.

`algorithms/qcast/` and `algorithms/baselines/` are independent online
baselines that use the shared lifecycle.

`algorithms/rl_routing/` contains ARC-Q. Its sparse graph policy directly
emits a feasibility-preserving autoregressive sequence of joint
request--path--construction actions and learns from SeQUeNCe feedback. It
does not call an optimization teacher or a post-hoc decoder.
