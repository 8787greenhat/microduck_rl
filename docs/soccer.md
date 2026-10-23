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
- [ ] Contact/separation latches and valid-shot termination implemented.
- [ ] Geometric state-machine baseline implemented.
- [ ] Frozen walking/kick action adapter implemented.
- [ ] `Mjlab-Soccer-Flat-MicroDuck` registered and smoke-tested.
- [ ] PPO and curriculum experiments completed.
- [ ] Evaluation plots, videos, and report completed.

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
