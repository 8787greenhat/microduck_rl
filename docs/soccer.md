# Shaolin Duck Soccer

## Project overview

Shaolin Duck Soccer is a simulation-only autonomous shooting project for
Microduck. The robot must approach a stationary ball, move to a foot-specific
preparation pose, align the shot with an undefended goal, and trigger an
existing kick skill without falling.

The project deliberately reuses frozen walking and kicking policies. It trains
only a new high-level controller, so the experiment measures skill composition
and decision making instead of relearning low-level joint control.

The central questions are:

- Can a learned high-level controller coordinate the existing skills better
  than a geometric state machine?
- Does a reset-distribution curriculum improve learning speed or final goal
  rate compared with sampling the full task from the first iteration?

The controlled comparison is:

1. geometric state machine;
2. high-level PPO trained on the full reset distribution;
3. the same high-level PPO and reward with a staged reset curriculum.

All three conditions must use the same frozen low-level policy checkpoints and
the same held-out evaluation configurations.

Camera perception, hardware deployment, opponents, moving-ball starts, and
two-robot passing are outside the initial scope.

## Architecture

```text
Simulator state
  - robot proprioception
  - robot-relative ball position and velocity
  - robot-relative goal position and heading
  - active skill and skill progress
            |
            v
High-level controller at 10 Hz
  [vx_cmd, vy_cmd, yaw_rate_cmd, kick_trigger]
            |
            v
Frozen-skill action adapter
  +----------------------+----------------------+
  | WALK                                        | KICK
  | frozen walking policy                       | frozen kick policy
  | consumes velocity command                   | fixed lockout duration
  +----------------------+----------------------+
            |
            v
Low-level joint targets at 50 Hz
            |
            v
Microduck + ball + goal in MuJoCo/mjlab
```

### Components

| Component | Responsibility |
| --- | --- |
| Soccer environment | Scene entities, randomized resets, observations, rewards, terminations, and episode metrics |
| Geometry helpers | Shared goal direction, preparation pose, and goal-plane crossing definitions |
| Geometry baseline | Approach, align, slow down, kick, and recover state machine |
| Frozen-skill adapter | Convert one high-level action into five 50 Hz low-level policy steps |
| High-level PPO | Select walking commands and kick timing from simulator state |
| Curriculum | Expand reset distance, heading offsets, and goal configurations without changing rewards |
| Evaluator | Run fixed held-out episodes and produce metrics, plots, trajectories, and videos |

## Control contract

- Low-level control remains at 50 Hz and keeps the existing 61D observation
  contract.
- The high-level controller updates at 10 Hz and outputs
  `[vx_cmd, vy_cmd, yaw_rate_cmd, kick_trigger]`.
- The first implementation uses the right-foot kick. Adding foot selection is
  a later action-space change and must not be mixed into the first comparison.
- Once triggered, the kick policy owns joint targets for a calibrated lockout
  duration; rapid policy switching is forbidden.
- A PyTorch checkpoint is preferred for batched training inference. ONNX is a
  deployment and single-environment validation path.

## Geometry contract

The goal normal points from the field into the goal. A valid plane crossing is
detected only when the ball moves from the negative side to the non-negative
side and its interpolated crossing point fits between the posts and under the
crossbar.

The right-foot preparation pose uses BallKick's nominal ball offset
`(0.09, -0.042)` in the robot yaw frame:

```text
goal_direction = normalize(goal_xy - ball_xy)
target_yaw = atan2(goal_direction.y, goal_direction.x)
target_robot_xy = ball_xy - rotate(target_yaw, kick_offset_xy)
```

The geometric baseline and PPO environment must call the same geometry helpers
so their success conditions cannot drift apart.

## Valid shot state

Goal-plane crossing alone is not enough. The environment will maintain these
per-episode latches before awarding a valid goal:

- kick was triggered;
- the selected kicking foot contacted the ball;
- the ball separated from the foot after contact;
- the ball subsequently crossed the goal plane through the opening.

Falls, timeouts, robot out-of-bounds, and ball out-of-bounds terminate as
failures.

## Experiment and acceptance criteria

The primary metric is valid goal rate over all episodes, including falls,
timeouts, invalid contacts, and missed shots. Secondary metrics are completion
time for successful episodes, fall rate, timeout rate, post-contact shot-angle
error, held-out-pose goal rate, and environment interactions required to reach
a selected success threshold.

Each learned condition will use at least three seeds. Each final policy will be
evaluated on at least 100 pre-generated held-out episodes. PPO without a
curriculum and PPO with a curriculum must share network size, rewards, frozen
skills, and total environment-interaction budget.

The engineering acceptance target is at least 70 valid goals in 100 standard
test episodes. This is a target, not an assumed result.

## Implementation stages

1. Pure geometry helpers and CPU tests.
2. Goal entity, reset distribution, goal/field terminations, and episode logs.
3. Geometry state machine using the frozen walking and kick skills.
4. Batched frozen-skill action adapter: high-level 10 Hz to low-level 50 Hz.
5. PPO without curriculum.
6. Identical PPO with reset-distribution curriculum.
7. Three seeds and fixed held-out evaluation episodes for each learned method.

Do not register a public `Mjlab-Soccer-*` task until stages 2-4 can build and
step without a checkpoint-dependent import failure.

## Development schedule

| Week | Deliverable |
| --- | --- |
| 1 | Reproduce walking and BallKick inference; record compatible checkpoints and observation contracts |
| 2 | Add goal geometry, field boundaries, reset distributions, goal detection, and episode logging |
| 3 | Implement and tune the geometric state-machine baseline |
| 4 | Implement the batched frozen-skill adapter and verify the 10 Hz / 50 Hz timing contract |
| 5 | Train high-level PPO on the full reset distribution; identify reward exploits |
| 6 | Add the staged reset curriculum while keeping the reward identical |
| 7 | Run three seeds per learned condition and the fixed held-out evaluation set |
| 8 | Produce learning curves, success maps, failure analysis, videos, and the final report |

## Current progress

Status as of 2026-10-23:

- [x] Existing official walking and BallKick task structure reviewed.
- [x] Soccer scope, architecture, control contract, and experiment constraints documented.
- [x] Shared vectorized goal-direction helper implemented.
- [x] Right-foot preparation-pose helper implemented using BallKick's nominal offset.
- [x] Directional goal-plane crossing with post, crossbar, radius, and high-speed interpolation checks implemented.
- [x] Eight CPU unit tests covering nominal and failure cases pass.
- [x] Goal entity and soccer field added to an mjlab scene.
- [x] Randomized episode reset and held-out evaluation-set generator implemented.
- [x] Contact/separation latches and valid-shot termination implemented
      (`soccer_valid_goal`: kick triggered → right foot touches ball while the
      kick skill runs → ball leaves the foot → crossing).
- [x] Geometric state-machine baseline implemented (`geometric_baseline_actions`).
      It does not score yet; see findings below.
- [x] Frozen walking/kick action adapter implemented (`soccer_adapter.SoccerSkillAdapter`,
      ONNX skills batched on the GPU through `onnx_mlp.OnnxMlp`).
- [x] `Mjlab-Soccer-Flat-MicroDuck` registered and smoke-tested (16 envs, 2 iterations, CPU).
- [x] Held-out evaluation (`scripts/eval_soccer.py`) and viewer (`scripts/view_soccer_policy.py`).
- [ ] Baseline evaluated on the 100 held-out episodes (needs the GPU).
- [ ] PPO and curriculum experiments completed.
- [ ] Evaluation plots, videos, and report completed.

## Findings so far (2026-10-24)

- **The blind kick only scores from a ~±2 cm placement.** `scripts/eval_kick_tolerance.py`
  (2016 envs, ball offset grid, 1 m goal): `ball_kick_right.onnx` scores 53 %
  with the ball exactly at its trained spot, ~0 % beyond ±3 cm laterally or
  +4.5 cm forward, and kicks 8° right on average. BallKick's reward pays only
  the forward ball-speed component capped at 1 m/s, so direction is free.
- **The walking skill cannot creep.** `alpha_walking.onnx` tracks a 0.1 m/s
  forward command at 0.00 m/s, 0.2 → ~0.05, 0.3 → ~0.15 m/s, identically in
  its own velocity env and in the soccer env (so the soccer env is not the cause).
- Together: the baseline walks to ~2.5 cm of the preparation pose within ~3 s,
  then its proportional command falls into the walking dead zone. It now kicks
  after settling 0.5 s inside a tolerance. Kick-tolerance sweep on the 100
  held-out layouts × 3 runs (300 episodes per point):

  | kick tolerance | goal | kick triggered | foot touched ball | timeout |
  | --- | --- | --- | --- | --- |
  | 2 cm | 11.3 % | 13.0 % | 12.3 % | 86.7 % |
  | 3 cm | 30.0 % | 46.7 % | 36.7 % | 60.3 % |
  | 4 cm | 38.7 % | 75.7 % | 55.3 % | 39.0 % |
  | 5 cm (default) | 40.3 % | 94.3 % | 64.0 % | 24.7 % |
  | 6 cm | 41.3 % | 98.3 % | 67.7 % | 23.3 % |
  | 10 cm | 37.0 % | 97.0 % | 64.0 % | 26.3 % |

  Method A plateaus near 40 %, below the 70 % target, bounded by the skills
  (median placement error at the kick 3.0 cm; the foot touches the ball in
  ~2/3 of kicks). Next: §2.1's fallback of retraining the kick with ball/goal
  observations and a direction-aware reward.
- **Method B (PPO, no curriculum) scores 95.3 %** on the same 300 held-out
  episodes (seed 1, 2048 envs, 2000 iterations, commit 7fa6298): foot contact
  98.7 %, ball out 2.3 %, falls 0.7 %, timeouts 0.3 %, 3.2 s to goal, median
  placement error at the kick 1.4 cm (method A: 3.0 cm). Same skills, different
  use: near the ball 96 % of its commands are above the walking dead zone
  (median 0.31 m/s vs 0.07 m/s for A), it triggers the kick while still walking
  (0.19 m/s vs 0.001 m/s), and it faces 2.7° left of the shot line at the kick,
  consistent with partly offsetting the kick's 8° right bias. About one kick per
  episode; the video shows real shots. Training first collapsed to never
  kicking (episodes timed out from about iteration 5 to 140; the kick-attempt cost was active from the start)
  and found kicking before iteration 200. One seed so far.
- `soccer_goal_crossing` rejected every ball rolling on the floor: MuJoCo's soft
  contact rests the ball centre 0.4 mm below its radius, under the old
  `z >= ball_radius` check. Fixed with a 5 mm ground tolerance and a test.

## Handoff: train method C on another machine

Status 2026-10-24: method A is evaluated (≈40 %), method B seed 1 scores
95.3 % (see findings), and method C is ready to train. Run C with the exact same settings as B so the comparison only measures
the curriculum.

1. **Get the branch and dependencies.**

   ```bash
   git fetch origin && git checkout feat/shaolin-duck-soccer && git pull
   uv sync
   ```

2. **Download the frozen skills** (not tracked in git). Pin the Hub revision so
   every machine runs byte-identical skills:

   ```bash
   uv run hf download pollen-robotics/microduck-policies \
     alpha_walking.onnx ball_kick_right.onnx \
     --revision 1b56c396825c052a4e26e95cf2b8d8298af9e9b4 --local-dir policies
   md5sum policies/alpha_walking.onnx policies/ball_kick_right.onnx
   # 2b68446b8d9a2d70a709cbd06d8b3a09  policies/alpha_walking.onnx
   # 9695407b57e2a7f4f32edd3baaaf2958  policies/ball_kick_right.onnx
   ```

   If the checksums differ, stop: B and C would be using different skills.

3. **Check the install** (CPU, ~10 s). If pytest fails with `No module named
   'lark'` or similar, a sourced ROS workspace is on `PYTHONPATH`; prefix the
   command with `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

   ```bash
   uv run --with pytest pytest tests/test_soccer_*.py -q    # expect 27 passed
   ```

4. **Smoke test, then train** (~5 h at 2048 envs on a 6 GB laptop GPU, faster
   on a 4090; keep 2048 envs anyway, see the table below):

   ```bash
   uv run train Mjlab-Soccer-Curriculum-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
   uv run train Mjlab-Soccer-Curriculum-Flat-MicroDuck --env.scene.num-envs 2048 \
     --agent.max_iterations 2000 --agent.seed 1 --agent.run-name seed1
   ```

   While it runs, `Curriculum/reset_layout/stage` should step 0 → 1 → 2 by
   iteration 600 at the latest, and every penalty `Episode_Reward/*` (fell,
   out_of_bounds, kick_attempt, time) must stay ≤ 0.

5. **Evaluate and report back** the printed rates and `summary.json`:

   ```bash
   uv run scripts/eval_soccer.py \
     --checkpoint logs/rsl_rl/soccer_highlevel_curriculum/<run>/model_1999.pt \
     --repeats 3 --out logs/eval_soccer/C_seed1
   uv run scripts/view_soccer_policy.py \
     --checkpoint logs/rsl_rl/soccer_highlevel_curriculum/<run>/model_1999.pt --num-envs 4
   ```

   Watch the video too: a high goal rate with a strange motion (shoving,
   repeated kicks, falls right after scoring) is a reward exploit to report.
   Then repeat with `--agent.seed 2` and `3`.

| must match between B and C | value |
| --- | --- |
| frozen skills | the two checksums above |
| envs | `--env.scene.num-envs 2048` |
| budget | `--agent.max_iterations 2000` |
| seeds | 1, 2, 3 |
| evaluation | `scripts/eval_soccer.py --repeats 3` on the same 100 held-out layouts |

## Running the soccer stack

```bash
uv run scripts/view_soccer_policy.py --baseline --num-envs 2     # watch method A
uv run scripts/eval_soccer.py --baseline --out logs/eval_soccer/baseline
uv run train Mjlab-Soccer-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

### Methods B and C (high-level PPO)

| method | task id | logs |
| --- | --- | --- |
| B: PPO, full reset distribution | `Mjlab-Soccer-Flat-MicroDuck` | `logs/rsl_rl/soccer_highlevel/` |
| C: PPO + reset curriculum | `Mjlab-Soccer-Curriculum-Flat-MicroDuck` | `logs/rsl_rl/soccer_highlevel_curriculum/` |

Keep these identical across B and C and across machines, or the comparison is
confounded: `--env.scene.num-envs 2048 --agent.max_iterations 2000`, seeds
`--agent.seed 1`, `2`, `3`. 2048 envs fit a 6 GB laptop GPU (~4 GB, ~9 s per
iteration, ~5 h per run); a bigger GPU only makes it faster.

```bash
uv run train Mjlab-Soccer-Curriculum-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max_iterations 2000 --agent.seed 1 --agent.run-name seed1
uv run scripts/eval_soccer.py --checkpoint logs/rsl_rl/soccer_highlevel_curriculum/<run>/model_1999.pt \
  --repeats 3 --out logs/eval_soccer/C_seed1
```

Method C's stages (`CURRICULUM_STAGES` in `microduck_soccer_env_cfg.py`): near-ready
(0–5 cm, ±10°), nearby (5–25 cm, ±45°), then exactly method B's ranges. A stage
is left at a 50 % goal rate or after 300 iterations, whichever comes first, so C
trains on the full distribution from iteration 600 at the latest; 20 % of resets
keep drawing from earlier stages. The 70 % gate proposed above was lowered
because method A, with the same frozen skills, tops out near 40 %. Watch
`Curriculum/reset_layout/stage` and `.../goal_rate` in tensorboard.

The frozen skills default to `policies/alpha_walking.onnx` and
`policies/ball_kick_right.onnx` (official `pollen-robotics/microduck-policies`
files, not tracked in git); override with `MICRODUCK_SOCCER_WALK_ONNX` /
`MICRODUCK_SOCCER_KICK_ONNX`. `uv run play` cannot drive this task: its viewer
steps the 14D joint action, not the 4D high-level one.

## Verification

Current foundation tests:

```bash
uv run --with pytest pytest \
  tests/test_soccer_geometry.py \
  tests/test_soccer_cfg.py
```

Generate the fixed 100-episode held-out set once, before reward tuning, and
keep the resulting JSON unchanged across all experiment conditions:

```bash
uv run scripts/generate_soccer_eval_set.py \
  data/soccer_eval_set.json \
  --episodes 100 \
  --seed 20261023
```

Before any long training run:

```bash
uv run train Mjlab-Soccer-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

The fixed evaluation set must be generated and saved before reward tuning.

## Repository workflow

Development is isolated on `feat/shaolin-duck-soccer`. Existing local changes
to `scripts/infer_policy.py` are intentionally outside the soccer commit and
must remain unmodified until their purpose and desired upstream status are
confirmed.
