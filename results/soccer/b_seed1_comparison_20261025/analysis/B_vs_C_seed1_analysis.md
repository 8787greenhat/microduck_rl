# Method B vs C, seed 1: verification and analysis (2026-10-25)

All numbers below come from files in this package; the source of each is named.

## 1. Are B and C comparable?

| check | B (this package) | C (teammate package) | result |
| --- | --- | --- | --- |
| commit | 7fa6298, working tree clean | 7fa6298 | same |
| frozen walking skill SHA256 | e36332d3…daa6c | e36332d3…daa6c | same |
| frozen kick skill SHA256 | 147a32c3…edb5af | 147a32c3…edb5af | same |
| eval set SHA256 | 3a59dee2…53464 | 3a59dee2…53464 | same bytes; the in-memory layouts used to evaluate B differ by 0.0 |
| envs × steps × iterations | 2048 × 24 × 2000 | 2048 × 24 × 2000 (effective) | same budget, 98,304,000 transitions |
| training seed | 1 | 1 | same |
| agent.yaml | — | — | only experiment/run name and logger differ (`training_configs/diff_B_vs_C_agent.yaml.txt`) |
| env.yaml | — | — | only C's `reset_layout` curriculum term, 55 lines (`training_configs/diff_B_vs_C_env.yaml.txt`) |
| interruptions | none | resumed from model_200; 201–267 discarded and redone | disclose |

Conclusion: the comparison is controlled except for the curriculum term and C's
interrupted run.

## 2. Final performance on the 300 held-out episodes

| | valid goal | foot contact | ball out | robot out | timeout | fell | time to goal | placement error at kick | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: geometric, 5 cm | 40.3 % | 65.7 % | 31.7 % | 1.0 % | 26.7 % | 0.3 % | 4.9 s | 3.0 cm | `results/A_baseline_tol5cm/` |
| B seed 1 (this laptop) | **95.3 %** | 98.7 % | 2.3 % | 1.3 % | 0.3 % | 0.7 % | 3.2 s | 1.4 cm | `results/B_seed1/` |
| C seed 1 (this laptop) | **92.3 %** | 97.0 % | 5.3 % | 1.0 % | 0.3 % | 1.0 % | 3.3 s | 2.3 cm | `results/C_seed1_rerun_on_dennis_laptop/` |
| C seed 1 (teammate's RTX 4060) | 94.0 % | 97.3 % | 3.0 % | 1.3 % | 1.0 % | 0.7 % | 3.3 s | 2.3 cm | teammate `results/summary.json` |

- **Cross-machine reproduction of C**: 92.3 % here vs 94.0 % on the teammate's
  machine, 5 episodes apart; the binomial standard error at n = 300 is about
  1.4 points. Same checkpoint, same layouts, same eval seed; GPUs differ.
- **B vs C on the same machine**: 95.3 % vs 92.3 %. Standard error of the
  difference ≈ 2.0 points, so 3 points is about 1.5 SE: **not significant**.
  The 3 repeats are re-evaluations of one trained policy, not 3 training seeds,
  so they say nothing about seed-to-seed variance.

## 3. Different strategies, similar outcome

Measured on the same 300 held-out episodes (diagnostic run, same layouts):

| | B | C |
| --- | --- | --- |
| velocity command near the ball (< 5 cm from prep pose), median | 0.31 m/s | 0.50 m/s |
| share of those commands above the walking dead zone (0.15 m/s) | 96 % | 99 % |
| robot speed when the kick is triggered, median | 0.20 m/s | 0.16 m/s |
| robot heading relative to the shot line at the kick | 3.1° left ± 3.7° | 7.8° left ± 5.7° |

The official kick sends the ball 8.2° right on average (blind-kick tolerance
measurement). C's 7.8° left aim nearly cancels it; B's 3.1° offsets part of it.
Both kick on the move instead of settling, which method A cannot do by design.
With one seed each we cannot attribute this difference to the curriculum.

## 4. Training dynamics

From `learning_curves/B_seed1.csv` and the teammate's `learning_curves/accepted_seed1.csv`:

| iteration | B mean episode length (10 Hz steps; 120 = timeout) | C mean episode length |
| --- | --- | --- |
| 25 | 119 | 81 |
| 50 | 120 | 64 |
| 100 | 120 | 35 |
| 150 | 87 | 27 |
| 200 | 43 | 37 |

- **B collapsed to never kicking** for iterations ~5–140 (every episode timed
  out), then found kicking; its mean reward turned positive around iteration 150.
- **C never fully collapsed**: on the near-ready stage early kicks score, and its
  mean reward was positive by iteration 75. This is the effect the plan
  predicted for a curriculum (better early exploration), seen once.
- **C's curriculum left the easy stages early**: stage 1 at iteration 15 and
  stage 2 (= B's distribution) at iteration 174, both promoted by the 50 % goal
  rate, not the 300-iteration cap. So ~91 % of C's training used B's
  distribution, which is consistent with the near-identical final rates.
- C's training `goal_rate` and mean reward before iteration 174 are on easier
  layouts and must not be compared with B's directly; sample efficiency should
  be measured on the fixed eval set at matching checkpoints (not done yet).

## 5. Limitations to state in the report

1. One training seed per method; seeds 2 and 3 are needed before any claim.
2. C's run was interrupted (driver update) and resumed from model_200; 67
   iterations were discarded and recomputed, and episodes were re-initialized.
3. At commit 7fa6298 the curriculum stage was not saved in checkpoints; the
   teammate restored stage 2 from the iteration-200 log by hand, and the curve
   stays on stage 2 from iteration 174 to 1999, so seed 1 is unaffected.
   Fixed in commit 64c57e5 (branch `dennis`): checkpoints now carry the stage.
4. Kicking on the move works in simulation, although the kick skill was trained
   from a standing start; no hardware claim.

## 6. Next steps

- Seeds 2 and 3 for B and C, same settings.
- Fixed-eval learning curves: evaluate every saved checkpoint of B and C on the
  300 held-out episodes to compare iterations-to-X% on the same test.
- Merge commit 64c57e5 (curriculum stage in checkpoints) before C seeds 2 and 3.
