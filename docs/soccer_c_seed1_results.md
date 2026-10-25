# Method C seed 1: verified results and PR #1 comparison

Method C completed 2,000 effective PPO iterations and scored **282/300 valid goals (94.0%)** on the original RTX 4060/WSL machine. Successful goals took **3.321 s mean**, **3.200 s median**, and **4.300 s p90**. This is one trained seed, evaluated on 100 fixed layouts three times.

## Artifacts

The [complete comparison ZIP](../results/soccer/c_seed1_20261025/microduck_C_seed1_comparison_20261025.zip) contains the final checkpoint, run configs, fixed evaluation set, per-episode outputs, accepted learning curve, recovery history and offline videos. Its SHA-256 is `b7b3ba1b1ae252c3b71a4cec1e91daffc76c1d715c219a0c64f55dd4e6ad287a`.

Readable files are also checked in under [results/soccer/c_seed1_20261025](../results/soccer/c_seed1_20261025). The result is based on actual policy updates, finite model tensors, optimizer step 40,000, a verified final checkpoint and completed episodes; it is not merely a model-load check.

| Original C seed 1 outcome | Episodes | Rate over all 300 |
|---|---:|---:|
| Valid goal | 282 | 94.0% |
| Ball out of bounds | 9 | 3.0% |
| Robot out of bounds | 4 | 1.33% |
| Timeout | 3 | 1.0% |
| Fall | 2 | 0.67% |

These failures do not overlap in this evaluation. Repeats scored 93/100, 95/100 and 94/100. There were 100 distinct layouts, not 300 independent layouts and not three training seeds.

## What PR #1 contains

[PR #1](https://github.com/8787greenhat/microduck_rl/pull/1), reviewed through `b375fc1d1eb2e60ce1f80e412e0c0b03a4eb0076`, contains:

1. B seed 1 results and behavior analysis in `docs/soccer.md`.
2. Saving/restoring the success-gated reset-curriculum state in `SoccerOnPolicyRunner`, respecting `MICRODUCK_WARM_START`.
3. A HW3 progress report and build files.
4. A documentation correction: B's early timeout/no-kick phase is approximately iterations 5–140.

The four soccer test modules passed **30 tests, one existing actuator-regex warning**, using a CPU-only isolated checkout at `25bb4d6`. The subsequent `b375fc1` changes only documentation; functional code is identical. No new training was run and the original C training checkout was not upgraded.

## Source-labelled B/C comparison

| Policy | Evaluation machine | Valid goal rate | Mean successful goal time | Evidence available here |
|---|---|---:|---:|---|
| B seed 1 | Teammate machine | 95.3% | 3.2 s | Reported in PR #1; raw files not yet provided |
| C seed 1, re-evaluated | Same teammate machine as B, according to PR #1 | 92.3% | 3.3 s | Reported in PR #1; raw files not yet provided |
| C seed 1, original evaluation | RTX 4060 / WSL | 94.0% (282/300) | 3.321 s | Verified summary, every episode and checkpoint hash |

The reported same-machine point estimates favor B by **3.0 percentage points**. The teammate's C evaluation is **1.7 points lower** than the original C evaluation. Keep these evaluation runs separate: do not silently replace 94.0% with 92.3% or combine B's result with a different-machine C result as a controlled comparison.

PR #1 states the layout hashes and low-level skills match. However, its body names `microduck_B_seed1_comparison_20261025.zip` without a download link. At this review, the fetched Git branches, PR body/comments/reviews and repository Releases did not contain the B bundle, raw B run configs, checkpoint, or per-episode evaluation files. Those identity claims and the same-machine C rerun therefore remain reported rather than independently audited. The original C package is provided here so the missing checks can be completed.

These one-seed results do **not** establish that curriculum improves final success, nor establish that it is worse across training seeds. PR #1's statement that the gap is “not significant” is not an independently reproduced test: the repeated layouts are clustered and the policies are paired by layout. Raw outcomes and a stated paired/cluster-aware procedure would be needed for an evaluation-level test; multiple independent training seeds are needed for algorithm-level variability. No p-value is asserted here.

Similarly, positive training reward earlier under easier C resets is an early-exploration observation, not a controlled measure of held-out sample efficiency. Compare both methods at the same interaction budgets on the same fixed test set before concluding C learns faster. Stage 2 begins at iteration 174 in C's accepted curve; being in stage 2 for about 91% of iterations is not the same as using B's reset distribution for 91% of samples, because C retains 20% easier resets.

## Reproducibility and recovery

- Training commit: `7fa62987d76dd2ba00ec15ffdd1de5bd93480845`.
- Training seed 1; 2,048 environments; 24 rollout steps/environment; 4 minibatches; 5 epochs; 2,000 effective iterations = 98,304,000 high-level transitions.
- Evaluation generator seed 20261023; evaluation RNG seed 0; 100 layouts × 3 repeats.
- Evaluation JSON SHA-256: `3a59dee2c22b5129fcb10685192630443a29550790ee5ccbffc04e414d053464`.
- Final checkpoint SHA-256: `8bda158d82f3e3834ebdc76028cadcc4584294d5e7cc751c84adc7d9d06b4de4`.
- Interrupted after a driver update; restored `model_200.pt` with stage 2 from its log, reinitialized physics episodes/counters, discarded the original iterations 201–267, then completed 201–1999. This is not a bit-exact uninterrupted run.
- `resume/agent.yaml` has `max_iterations: 1799` because that is the remaining count; the total budget is still 2,000. Save interval changed from 100 to 25 for recovery protection.
- The accepted learning curve includes original 0–200 and resumed 201–1999, excluding the discarded 67 updates. The discarded segment and diagnostic work still count toward compute spent.

To finish the comparison, obtain B's actual configs/commit/checkpoint, its fixed-set JSON/hash, B and teammate-C `summary.json` plus per-episode files, and B's training curve. No seeds 2/3, new long training, reward changes, or merges are initiated by this results PR.
