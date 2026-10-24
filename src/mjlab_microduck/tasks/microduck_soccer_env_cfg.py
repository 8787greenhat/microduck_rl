"""Shaolin Duck Soccer: the 50 Hz environment under the 10 Hz high-level controller.

The env keeps BallKick's robot, actuators, domain randomization and 61D
observation contract, and adds the goal, the soccer reset layout, the shot
latches and the high-level reward. It is stepped by
``mjlab_microduck.soccer_adapter.SoccerSkillAdapter``, which turns one
high-level action ``[vx_cmd, vy_cmd, yaw_rate_cmd, kick_trigger]`` into five
50 Hz steps of the frozen walking or kicking policy.

Observation groups:
  - ``lowlevel``: BallKick's 61D actor group, unchanged. The adapter feeds it
    to the frozen skills after writing the walking command into the twist slot.
  - ``actor``: the high-level policy input (proprioception + robot-relative
    ball, goal, and skill state). Simulator state, no perception noise yet.
  - ``critic``: actor terms plus privileged base velocity and the
    preparation-pose error.

Reward (identical for PPO with and without curriculum, see docs/soccer.md):
one-time valid-goal bonus, potential-based progress toward the right-foot
preparation pose and shot heading (pre-kick only), bounded goal-directed ball
speed after a foot-contact kick, and costs for kick attempts, time, falls and
leaving the field. Every positive shaping term is a rate of progress or
bounded, so standing near the ball cannot be farmed.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.robot.microduck_constants import MICRODUCK_SOCCER_GOAL_CFG
from mjlab_microduck.soccer_constants import (
    APPROACH_DISTANCE_RANGE,
    APPROACH_LATERAL_RANGE,
    BALL_LATERAL_RANGE,
    BALL_RADIUS,
    CROSSBAR_HEIGHT,
    FIELD_LENGTH,
    FIELD_WIDTH,
    GOAL_DISTANCE_RANGE,
    GOAL_WIDTH,
    HEADING_ERROR_RANGE,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── Episode ──────────────────────────────────────────────────────────────────
# Worst reset: 0.45 m behind the prep pose, 90° off heading, plus the kick and
# ~1 m of ball travel. 12 s is ~2x a direct approach at 0.3 m/s.
EPISODE_LENGTH_S = 12.0

# Pushes would make a shot's outcome depend on push timing, adding variance to
# the A/B/C comparison that has nothing to do with the decision policy.
ENABLE_PUSHES = False

# ── Reward weights (see module docstring) ────────────────────────────────────
W_GOAL = 10.0             # one-time
W_PREP_PROGRESS = 2.0     # per metre of approach to the prep pose
W_ALIGN_PROGRESS = 0.5    # per radian of heading error removed
W_BALL_TO_GOAL = 2.0      # per (m/s · s) of goal-directed ball motion after contact
W_KICK_ATTEMPT = -0.5     # per kick trigger
W_TIME = -0.1             # per second
W_FELL = -3.0             # one-time
W_OUT_OF_BOUNDS = -2.0    # one-time; > max time cost saved by ending early

FELL_LIMIT_ANGLE = 1.2217304763960306  # BallKick's fell_over limit (70°)


def make_microduck_soccer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the soccer env: BallKick physics + goal + high-level MDP."""
    cfg = make_microduck_ball_kick_env_cfg(play=play, kick_foot="right")

    # Preserve robot first: multiple inherited reset helpers rely on that
    # insertion order.  The fixed goal is auto-wrapped as a per-env mocap body.
    cfg.scene.entities = {
        "robot": cfg.scene.entities["robot"],
        "ball": cfg.scene.entities["ball"],
        "goal": MICRODUCK_SOCCER_GOAL_CFG,
    }
    cfg.scene.env_spacing = 2.5
    cfg.sim.nconmax = max(cfg.sim.nconmax, 80)
    cfg.episode_length_s = EPISODE_LENGTH_S

    kick_contact_cfg = ContactSensorCfg(
        name="kick_foot_ball_contact",
        primary=ContactMatch(mode="geom", pattern=r"^right_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    # Any robot part on the ball: a touch while walking voids the previous kick.
    body_contact_cfg = ContactSensorCfg(
        name="robot_ball_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, kick_contact_cfg, body_contact_cfg)

    # Replace BallKick's toe-relative ball placement with one event that owns
    # the robot/ball/goal geometry, preventing independently sampled states.
    cfg.events.pop("reset_ball")
    cfg.events["reset_soccer_scene"] = EventTermCfg(
        func=microduck_mdp.reset_soccer_scene,
        mode="reset",
        params={
            "field_length": FIELD_LENGTH,
            "field_width": FIELD_WIDTH,
            "goal_distance_range": GOAL_DISTANCE_RANGE,
            "ball_lateral_range": BALL_LATERAL_RANGE,
            "approach_distance_range": APPROACH_DISTANCE_RANGE,
            "approach_lateral_range": APPROACH_LATERAL_RANGE,
            "heading_error_range": HEADING_ERROR_RANGE,
            "ball_radius": BALL_RADIUS,
        },
    )
    cfg.events["reset_soccer_shot_state"] = EventTermCfg(
        func=microduck_mdp.reset_soccer_shot_state, mode="reset",
    )
    if not ENABLE_PUSHES:
        cfg.events.pop("push_robot", None)
        cfg.curriculum.pop("push_magnitude", None)

    cfg.terminations["robot_out_of_bounds"] = TerminationTermCfg(
        func=microduck_mdp.soccer_asset_out_of_bounds,
        time_out=False,
        params={
            "asset_name": "robot",
            "field_length": FIELD_LENGTH,
            "field_width": FIELD_WIDTH,
        },
    )
    cfg.terminations["ball_out_of_bounds"] = TerminationTermCfg(
        func=microduck_mdp.soccer_asset_out_of_bounds,
        time_out=False,
        params={
            "asset_name": "ball",
            "field_length": FIELD_LENGTH,
            "field_width": FIELD_WIDTH,
            "margin": BALL_RADIUS,
        },
    )
    # Also advances the shot latches every step: keep it in every variant.
    cfg.terminations["valid_goal"] = TerminationTermCfg(
        func=microduck_mdp.soccer_valid_goal,
        time_out=False,
        params={
            "sensor_name": kick_contact_cfg.name,
            "body_sensor_name": body_contact_cfg.name,
            "goal_width": GOAL_WIDTH,
            "crossbar_height": CROSSBAR_HEIGHT,
            "ball_radius": BALL_RADIUS,
        },
    )

    _configure_observations(cfg)
    _configure_rewards(cfg)

    # BallKick's reward-weight schedule targets terms that no longer exist.
    cfg.curriculum.pop("action_rate_weight", None)
    return cfg


def _configure_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    lowlevel = cfg.observations.pop("actor")
    cfg.observations.pop("critic")
    cfg.observations["lowlevel"] = lowlevel

    proprio = {
        name: deepcopy(lowlevel.terms[name])
        for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")
    }
    task = {
        "ball": ObservationTermCfg(func=microduck_mdp.soccer_ball_obs),
        "goal": ObservationTermCfg(func=microduck_mdp.soccer_goal_obs),
        "skill": ObservationTermCfg(func=microduck_mdp.soccer_skill_obs),
    }
    cfg.observations["actor"] = ObservationGroupCfg(
        terms={**proprio, **deepcopy(task)},
        concatenate_terms=True,
        enable_corruption=True,
    )

    clean_proprio = deepcopy(proprio)
    for term in clean_proprio.values():
        term.noise = None
        term.delay_min_lag = 0
        term.delay_max_lag = 0
    clean_proprio["joint_pos"].params["biased"] = False
    cfg.observations["critic"] = ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
            **clean_proprio,
            **deepcopy(task),
            "prep_error": ObservationTermCfg(func=microduck_mdp.soccer_prep_error_obs),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )


def _configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    oob = {"field_length": FIELD_LENGTH, "field_width": FIELD_WIDTH, "ball_radius": BALL_RADIUS}
    cfg.rewards = {
        "goal": RewardTermCfg(func=microduck_mdp.soccer_goal_bonus, weight=W_GOAL),
        "prep_progress": RewardTermCfg(func=microduck_mdp.soccer_prep_progress, weight=W_PREP_PROGRESS),
        "align_progress": RewardTermCfg(func=microduck_mdp.soccer_align_progress, weight=W_ALIGN_PROGRESS),
        "ball_to_goal": RewardTermCfg(func=microduck_mdp.soccer_ball_to_goal_speed, weight=W_BALL_TO_GOAL),
        "kick_attempt": RewardTermCfg(func=microduck_mdp.soccer_kick_attempt, weight=W_KICK_ATTEMPT),
        "time": RewardTermCfg(func=microduck_mdp.soccer_time_cost, weight=W_TIME),
        "fell": RewardTermCfg(
            func=microduck_mdp.soccer_fell, weight=W_FELL, params={"limit_angle": FELL_LIMIT_ANGLE},
        ),
        "out_of_bounds": RewardTermCfg(
            func=microduck_mdp.soccer_out_of_bounds_failure, weight=W_OUT_OF_BOUNDS, params=oob,
        ),
    }


# ── High-level PPO (docs/soccer.md: two 128-unit layers, normalized obs) ─────

MicroduckSoccerRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(128, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.5,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(128, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="soccer_highlevel",
    run_name="soccer_highlevel",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=2000,
)
