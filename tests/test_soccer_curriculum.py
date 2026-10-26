"""Method C (reset curriculum) must differ from method B only in its reset ranges."""

from dataclasses import asdict
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_soccer_env_cfg import (
    CURRICULUM_STAGES,
    MicroduckSoccerCurriculumRlCfg,
    MicroduckSoccerRlCfg,
    make_microduck_soccer_curriculum_env_cfg,
    make_microduck_soccer_env_cfg,
)

FIELD = {"field_length": 2.0, "field_width": 1.2}


def test_last_stage_is_method_b_reset_distribution():
    b_reset = make_microduck_soccer_env_cfg().events["reset_soccer_scene"].params
    for key, value in CURRICULUM_STAGES[-1].items():
        assert b_reset[key] == value, key


def test_curriculum_is_the_only_difference_from_method_b():
    b, c = make_microduck_soccer_env_cfg(), make_microduck_soccer_curriculum_env_cfg()
    assert set(c.curriculum) - set(b.curriculum) == {"reset_layout"}
    assert set(c.rewards) == set(b.rewards)
    assert all(c.rewards[k].weight == b.rewards[k].weight for k in b.rewards)
    b_rl, c_rl = asdict(MicroduckSoccerRlCfg), asdict(MicroduckSoccerCurriculumRlCfg)
    differing = {k for k in b_rl if b_rl[k] != c_rl[k]}
    assert differing == {"experiment_name", "run_name"}


def _env(n=64):
    # episode_length_buf > 0: every reset in these tests ends an episode that ran.
    env = SimpleNamespace(num_envs=n, device="cpu", common_step_counter=0,
                          episode_length_buf=torch.ones(n, dtype=torch.long))
    mdp.soccer_state(env)
    return env


def _reset(env, **kw):
    params = {"stages": CURRICULUM_STAGES, **FIELD, "min_episodes": 32,
              "max_iterations_per_stage": 10, "steps_per_iteration": 120, **kw}
    return mdp.soccer_reset_curriculum(env, slice(None), **params)


def test_starts_on_the_easiest_stage_and_fills_the_layout_override():
    env = _env()
    state = _reset(env)
    assert state["stage"] == 0.0
    approach = CURRICULUM_STAGES[0]["approach_distance_range"]
    prep = mdp.soccer_preparation_pose(env._soccer_layout_override["ball_xy"],
                                       env._soccer_layout_override["goal_xy_yaw"][:, :2])
    dist = torch.linalg.vector_norm(prep[:, :2] - env._soccer_layout_override["robot_xy_yaw"][:, :2], dim=-1)
    assert (dist <= approach[1] + 0.02 + 1e-5).all()  # approach + lateral offset


def test_promotes_on_goal_rate():
    env = _env()
    _reset(env)
    env._soccer_valid_goal[:] = True  # every episode that started in stage 0 scored
    assert _reset(env, promote_goal_rate=0.5)["stage"] == 1.0


def test_does_not_promote_on_low_goal_rate_before_the_cap():
    env = _env()
    _reset(env)
    env._soccer_valid_goal[:] = False
    assert _reset(env)["stage"] == 0.0


def test_startup_resets_do_not_count_as_finished_episodes():
    # Training resets every env twice before the first step (RslRlVecEnvWrapper,
    # then SoccerSkillAdapter); no episode has run, so nothing may be scored.
    env = _env()
    env.episode_length_buf[:] = 0
    _reset(env)
    _reset(env)
    assert env._soccer_curriculum_episodes == 0
    env.episode_length_buf[:] = 1  # now every env has played its episode
    env._soccer_valid_goal[:] = True
    assert _reset(env, promote_goal_rate=0.5)["stage"] == 1.0


def test_iteration_cap_promotes_even_without_goals():
    env = _env()
    _reset(env)
    env.common_step_counter = 10 * 120
    assert _reset(env)["stage"] == 1.0


def test_easy_fraction_keeps_some_resets_on_earlier_stages():
    env = _env(n=2000)
    _reset(env)
    env.common_step_counter = 10 * 120
    _reset(env, easy_fraction=0.2)
    earlier = (env._soccer_episode_stage == 0).float().mean().item()
    assert 0.15 < earlier < 0.25
