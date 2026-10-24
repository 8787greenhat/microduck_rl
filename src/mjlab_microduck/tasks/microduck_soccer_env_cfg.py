"""Soccer scene/reset foundation.

This module is intentionally not registered as a public task yet.  The action
path is still BallKick's 50 Hz joint-position interface; stage 4 will replace
it with the 10 Hz frozen-skill adapter before ``Mjlab-Soccer-*`` is exposed.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, TerminationTermCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_SOCCER_GOAL_CFG
from mjlab_microduck.soccer_constants import (
    APPROACH_DISTANCE_RANGE,
    APPROACH_LATERAL_RANGE,
    BALL_LATERAL_RANGE,
    BALL_RADIUS,
    FIELD_LENGTH,
    FIELD_WIDTH,
    GOAL_DISTANCE_RANGE,
    HEADING_ERROR_RANGE,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
)


def make_microduck_soccer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the unregistered soccer scene with coherent randomized resets."""
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
    return cfg
