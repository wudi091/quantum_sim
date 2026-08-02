# Algorithm layout

Each routing or control algorithm owns one directory. Algorithm-specific
candidate generation, optimization, training, checkpoints, and tests must stay
inside that directory. Shared physical state, execution, and immutable
contracts stay in `qnet_core`.

Current layout:

- `con_method/`: the CON method described by `con_design.md`;
- `conflict_aware_greedy/`: greedy schedule-portfolio baseline.

Future sequential, balanced, random, Q-DDCA, Q-CAST, M-PSES-like, and other
baselines should each receive their own sibling directory instead of being
added to a common algorithm file.
