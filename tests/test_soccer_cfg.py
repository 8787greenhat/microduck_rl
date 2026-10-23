import mujoco
from mjlab.scene import Scene

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_SOCCER_GOAL_CFG,
    get_soccer_goal_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_soccer_env_cfg import (
    BALL_LATERAL_RANGE,
    FIELD_LENGTH,
    FIELD_WIDTH,
    GOAL_DISTANCE_RANGE,
    make_microduck_soccer_env_cfg,
)


def _geom_names(model):
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    }


def test_soccer_goal_has_collidable_posts_crossbar_and_rails():
    model = get_soccer_goal_spec().compile()
    assert {
        "goal_left_post",
        "goal_right_post",
        "goal_crossbar",
        "goal_left_rail",
        "goal_right_rail",
    } <= _geom_names(model)
    for name in _geom_names(model):
        assert model.geom(name).contype[0] != 0


def test_soccer_cfg_adds_goal_and_coherent_reset_event():
    cfg = make_microduck_soccer_env_cfg()

    assert tuple(cfg.scene.entities) == ("robot", "ball", "goal")
    assert cfg.scene.entities["goal"] is MICRODUCK_SOCCER_GOAL_CFG
    assert "reset_ball" not in cfg.events
    reset = cfg.events["reset_soccer_scene"]
    assert reset.func is microduck_mdp.reset_soccer_scene
    assert reset.params["field_length"] == FIELD_LENGTH
    assert reset.params["field_width"] == FIELD_WIDTH
    assert reset.params["goal_distance_range"] == GOAL_DISTANCE_RANGE
    assert reset.params["ball_lateral_range"] == BALL_LATERAL_RANGE


def test_soccer_scene_compiles_with_two_vectorized_envs():
    cfg = make_microduck_soccer_env_cfg()
    cfg.scene.num_envs = 2

    model = Scene(cfg.scene, device="cpu").compile()

    assert model.nbody > 2
    assert model.ngeom > 5


def test_soccer_cfg_terminates_robot_and_ball_out_of_bounds():
    cfg = make_microduck_soccer_env_cfg()

    for name, asset_name in (
        ("robot_out_of_bounds", "robot"),
        ("ball_out_of_bounds", "ball"),
    ):
        term = cfg.terminations[name]
        assert term.func is microduck_mdp.soccer_asset_out_of_bounds
        assert term.params["asset_name"] == asset_name
        assert term.params["field_length"] == FIELD_LENGTH
        assert term.params["field_width"] == FIELD_WIDTH
