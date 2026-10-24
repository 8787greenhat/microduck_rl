"""Valid-shot latches and the adapter's kick lockout, on a minimal fake env (CPU)."""

from types import SimpleNamespace

import torch

from mjlab_microduck.soccer_adapter import SoccerSkillAdapter
from mjlab_microduck.tasks import mdp

GOAL_X = 1.0


class _Scene(dict):
    def __init__(self, ball, sensors):
        super().__init__(ball=ball)
        self.sensors = sensors


def _env():
    ball = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.5, 0.0, 0.0346]]),
        root_link_lin_vel_w=torch.zeros(1, 3),
    ))
    sensors = {
        "foot": SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 1))),
        "body": SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 1))),
    }
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02, scene=_Scene(ball, sensors))
    mdp.soccer_state(env)
    env._soccer_goal_base_pos_w[:] = torch.tensor([GOAL_X, 0.0, 0.0])
    env._soccer_previous_ball_pos_w[:] = ball.data.root_link_pos_w
    return env


def _step(env, *, kicking=False, foot=False, body=False, ball_x=None, ball_vx=0.0):
    env._soccer_skill_mode[:] = mdp.SOCCER_SKILL_KICK if kicking else mdp.SOCCER_SKILL_WALK
    if kicking:
        env._soccer_kick_triggered[:] = True
    env.scene.sensors["foot"].data.found[:] = float(foot)
    env.scene.sensors["body"].data.found[:] = float(body or foot)
    if ball_x is not None:
        env.scene["ball"].data.root_link_pos_w[0, 0] = ball_x
    env.scene["ball"].data.root_link_lin_vel_w[0, 0] = ball_vx
    return mdp.soccer_valid_goal(
        env, sensor_name="foot", body_sensor_name="body",
        goal_width=0.40, crossbar_height=0.20, ball_radius=0.035,
    ).item()


def test_kicked_ball_that_rolls_in_is_a_valid_goal():
    env = _env()
    _step(env, kicking=True, foot=True)
    _step(env, kicking=True, ball_x=0.7, ball_vx=1.0)
    assert _step(env, ball_x=GOAL_X + 0.02, ball_vx=1.0) is True


def test_walking_the_ball_in_after_a_kick_does_not_score():
    env = _env()
    _step(env, kicking=True, foot=True)
    _step(env, kicking=True, ball_x=0.7, ball_vx=1.0)
    _step(env, body=True, ball_x=0.9, ball_vx=0.3)  # pushed with the body while walking
    assert _step(env, ball_x=GOAL_X + 0.02, ball_vx=0.3) is False


def test_ball_pushed_in_without_any_kick_does_not_score():
    env = _env()
    _step(env, body=True, ball_x=0.9, ball_vx=0.3)
    _step(env, ball_x=0.95, ball_vx=0.3)
    assert _step(env, ball_x=GOAL_X + 0.02, ball_vx=0.3) is False


def test_adapter_kick_lockout_ignores_retriggers_and_returns_to_walking():
    env = _env()
    adapter = object.__new__(SoccerSkillAdapter)
    adapter.env = SimpleNamespace(unwrapped=env)
    adapter.kick_steps = 3
    adapter.command_low = torch.tensor([-0.4, -0.3, -1.0])
    adapter.command_high = torch.tensor([0.4, 0.3, 1.0])

    kick = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    adapter.apply_high_level(kick)
    assert env._soccer_skill_mode.item() == mdp.SOCCER_SKILL_KICK
    for _ in range(2):
        adapter._advance_kicks()
        adapter.apply_high_level(kick)  # retrigger during the lockout
    assert env._soccer_kick_count.item() == 1
    adapter._advance_kicks()
    assert env._soccer_skill_mode.item() == mdp.SOCCER_SKILL_WALK
