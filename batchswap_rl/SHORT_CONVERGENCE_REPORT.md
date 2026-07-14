# Short-stage PPO convergence check

## Setup

- Pure masked PPO; no supervised data or optimizer labels.
- Four persistent requests per episode.
- Paths span 2--5 hops.
- EPR generation probability 0.8; node swap capacity 3.
- 50 PPO updates, 256 decisions/update, four PPO epochs.
- Training instances use a deterministic but changing episode-seed stream.
- Evaluation uses 100 unseen paired seeds (`20000..20099`).

Artifacts:

- `runs/short_convergence_v2/history.json`
- `runs/short_convergence_v2/checkpoint_000001.pt`
- `runs/short_convergence_v2/checkpoint.pt`
- `runs/short_convergence_v2/eval_update001.json`
- `runs/short_convergence_v2/eval_update050.json`

## Training signal

| Metric | Updates 1--10 | Updates 41--50 | Change |
|---|---:|---:|---:|
| Mean completed-request delay | 4.891 | 4.415 | -9.7% |
| Episode makespan | 7.157 | 6.619 | -7.5% |
| Decisions per episode | 13.346 | 10.861 | -18.6% |
| Rollout mean reward | 1.421 | 1.755 | +23.5% |
| Policy entropy | 1.082 | 0.829 | -23.4% |

The critic value loss fell from 51.85 to 1.46 on average. No NaNs, invalid
masked actions, or incomplete episodes were observed.

## Unseen-seed evaluation

| Policy | Completion | Mean delay | P95 delay | Makespan |
|---|---:|---:|---:|---:|
| PPO at update 1 | 100% | 4.3175 | 8 | 6.83 |
| PPO at update 50 | 100% | 3.9675 | 7 | 5.77 |
| Greedy | 100% | 3.9825 | 7 | 5.73 |
| Q-DDCA-style | 100% | 4.0100 | 6 | 5.79 |
| Random valid | 100% | 5.6075 | 10 | 8.21 |

Relative to its update-1 checkpoint, trained PPO reduced mean delay by 8.1%
and makespan by 15.5%. It clearly beat random selection and reached the same
performance band as the hand-designed greedy and Q-DDCA-style controllers.

PPO's advantage over greedy/Q-DDCA-style is small in this short, lightly
contended setting: 42 wins, 29 ties, and 29 losses against Q-DDCA-style by
per-seed mean delay; 15 wins, 76 ties, and 9 losses against greedy. This run
demonstrates learnability and generalization, not a statistically established
superiority claim.

## Long-hop sanity check

Without any medium/long curriculum training, the short-stage checkpoint was
tested on three 20--50-hop seeds. It completed every request and obtained mean
delay 55.41, versus 55.03 for greedy, 65.96 for Q-DDCA-style, and 66.62 for
random. Three seeds are only a pipeline sanity check and are not an experimental
result.

## Conclusion

The masked sequential plan-selection formulation is trainable with pure RL.
The next required experiment is full curriculum training followed by a larger
paired 20--50-hop evaluation. The aggregate-resource abstraction must remain an
explicit limitation until a token-level physical environment is added.
