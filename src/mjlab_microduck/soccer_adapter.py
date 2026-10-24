"""Frozen-skill adapter: one 10 Hz high-level action -> five 50 Hz skill steps.

``SoccerSkillAdapter`` replaces ``RslRlVecEnvWrapper`` for the soccer task, so
PPO sees a 4D action ``[vx_cmd, vy_cmd, yaw_rate_cmd, kick_trigger]`` and the
high-level ``actor``/``critic`` observation groups. Underneath, every env step
is still the 50 Hz BallKick physics, driven by the frozen official skills:

  - WALK: ``alpha_walking.onnx`` with the clamped velocity command written into
    the twist slot of the 61D ``lowlevel`` observation.
  - KICK: ``ball_kick_right.onnx`` with an all-zero command, for a fixed
    lockout (``KICK_LOCKOUT_S``) once ``kick_trigger > 0``; triggers during the
    lockout are ignored, so skills never switch faster than that.

Both skills run on the GPU through ``OnnxMlp`` (bit-equivalent to onnxruntime
to ~1e-5), so thousands of envs step in one batched call. The skill state
lives on the env (``mdp.soccer_state``) where rewards, latches and the
geometric baseline read it.
"""

import os
from pathlib import Path

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from mjlab_microduck.onnx_mlp import OnnxMlp
from mjlab_microduck.tasks import mdp as microduck_mdp

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Relative paths resolve against the repo root, so scripts work from any cwd.
WALK_POLICY_ONNX = str(_REPO_ROOT / os.environ.get("MICRODUCK_SOCCER_WALK_ONNX", "policies/alpha_walking.onnx"))
KICK_POLICY_ONNX = str(_REPO_ROOT / os.environ.get("MICRODUCK_SOCCER_KICK_ONNX", "policies/ball_kick_right.onnx"))

SUBSTEPS = 5                 # 50 Hz skill steps per 10 Hz high-level step
KICK_LOCKOUT_S = 2.0         # kick + settle; the kick policy was trained on 5 s episodes
# The walking policy's training command ranges (velocity env); commands outside
# them are out of distribution for the frozen skill.
COMMAND_LOW = (-0.4, -0.3, -1.0)
COMMAND_HIGH = (0.4, 0.3, 1.0)
TWIST_SLOT = slice(48, 51)   # [twist(3), head(4), body(6)] after 48D proprioception
LOWLEVEL_OBS_DIM = 61


class SoccerSkillAdapter(RslRlVecEnvWrapper):
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        walk_onnx: str = WALK_POLICY_ONNX,
        kick_onnx: str = KICK_POLICY_ONNX,
    ):
        super().__init__(env)  # resets the env
        e = self.unwrapped
        self.num_actions = 4
        self.walk = OnnxMlp(walk_onnx, e.device)
        self.kick = OnnxMlp(kick_onnx, e.device)
        lowlevel_dim = e.observation_manager.group_obs_dim["lowlevel"][0]
        for policy in (self.walk, self.kick):
            if policy.obs_dim != LOWLEVEL_OBS_DIM or lowlevel_dim != LOWLEVEL_OBS_DIM:
                raise ValueError(
                    f"{policy.path}: expected the {LOWLEVEL_OBS_DIM}D obs contract, "
                    f"policy takes {policy.obs_dim}D, env lowlevel group is {lowlevel_dim}D"
                )
        microduck_mdp.soccer_state(e)
        self.kick_steps = round(KICK_LOCKOUT_S / e.step_dt)
        self.command_low = torch.tensor(COMMAND_LOW, device=e.device)
        self.command_high = torch.tensor(COMMAND_HIGH, device=e.device)
        self._obs = e.obs_buf

    def get_observations(self) -> TensorDict:
        return TensorDict(self._obs, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = self.env.reset()
        self._obs = obs
        return TensorDict(obs, batch_size=[self.num_envs]), extras

    def apply_high_level(self, actions: torch.Tensor) -> None:
        """Latch a new command and start kicks for walking envs that trigger."""
        e = self.unwrapped
        e._soccer_command[:] = torch.clamp(actions[:, :3], self.command_low, self.command_high)
        start = (actions[:, 3] > 0.0) & (e._soccer_skill_mode == microduck_mdp.SOCCER_SKILL_WALK)
        e._soccer_skill_mode[start] = microduck_mdp.SOCCER_SKILL_KICK
        e._soccer_kick_steps_left[start] = self.kick_steps
        e._soccer_kick_progress[start] = 0.0
        e._soccer_kick_triggered |= start
        e._soccer_kick_count += start.long()

    def skill_actions(self, lowlevel_obs: torch.Tensor) -> torch.Tensor:
        """Joint targets from the frozen skill each env is currently running."""
        e = self.unwrapped
        kicking = (e._soccer_skill_mode == microduck_mdp.SOCCER_SKILL_KICK).unsqueeze(-1)
        obs = lowlevel_obs.clone()
        obs[:, TWIST_SLOT] = torch.where(kicking, torch.zeros_like(e._soccer_command), e._soccer_command)
        return torch.where(kicking, self.kick(obs), self.walk(obs))

    def _advance_kicks(self) -> None:
        e = self.unwrapped
        kicking = e._soccer_skill_mode == microduck_mdp.SOCCER_SKILL_KICK
        e._soccer_kick_steps_left[kicking] -= 1
        e._soccer_kick_progress[kicking] = 1.0 - e._soccer_kick_steps_left[kicking].float() / self.kick_steps
        finished = kicking & (e._soccer_kick_steps_left <= 0)
        e._soccer_skill_mode[finished] = microduck_mdp.SOCCER_SKILL_WALK
        e._soccer_kick_progress[finished] = 0.0

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Run SUBSTEPS skill steps; rewards stop accumulating at an env's first done.

        An env that terminates mid-step is auto-reset by mjlab (which also
        resets its skill state), so its remaining substeps stand still at the
        new episode's start; their reward belongs to no transition and is dropped.
        """
        e = self.unwrapped
        self.apply_high_level(actions)
        reward = torch.zeros(self.num_envs, device=e.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=e.device)
        time_out = torch.zeros_like(done)
        logs: dict = {}
        extras: dict = {}
        terms = e.termination_manager
        # Which terminations ended each env's transition, captured at its done
        # substep (later substeps already belong to the auto-reset episode).
        done_terms = {name: torch.zeros_like(done) for name in terms.active_terms}
        for _ in range(SUBSTEPS):
            obs, rew, terminated, truncated, extras = e.step(self.skill_actions(self._obs["lowlevel"]))
            alive = ~done
            reward += rew * alive
            new_done = (terminated | truncated) & alive
            time_out |= truncated & new_done
            for name in done_terms:
                done_terms[name] |= new_done & terms.get_term(name)
            done |= new_done
            logs.update(extras.get("log", {}))
            self._advance_kicks()
            self._obs = obs
        extras = {**extras, "log": logs, "done_terms": done_terms}
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = time_out
        return TensorDict(self._obs, batch_size=[self.num_envs]), reward, done.long(), extras


# --------------------------------------------------------------------------- #
# Method A: geometric state machine on the same adapter                        #
# --------------------------------------------------------------------------- #

# Kick only inside BallKick's measured placement tolerance (scripts/eval_kick_
# tolerance.py: goal rate collapses beyond about ±2 cm laterally).
BASELINE_POS_TOL = 0.02      # m
BASELINE_YAW_TOL = 0.10      # rad
BASELINE_SPEED_TOL = 0.05    # m/s, settle before kicking
BASELINE_GAIN_XY = 2.0       # (m/s) per m of prep-pose error
BASELINE_GAIN_YAW = 2.0      # (rad/s) per rad of heading error


def geometric_baseline_actions(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Approach -> align -> slow down -> kick, from the shared geometry helpers.

    Proportional walking commands toward ``soccer_preparation_pose`` in the
    robot yaw frame, saturated at the walking policy's command range; the kick
    fires once position, heading and speed are all inside tolerance.
    """
    geo = microduck_mdp.soccer_geometry(env)
    c, s = torch.cos(geo["robot_yaw"]), torch.sin(geo["robot_yaw"])
    dx = c * geo["prep_delta"][:, 0] + s * geo["prep_delta"][:, 1]
    dy = -s * geo["prep_delta"][:, 0] + c * geo["prep_delta"][:, 1]
    cmd = torch.stack(
        (BASELINE_GAIN_XY * dx, BASELINE_GAIN_XY * dy, BASELINE_GAIN_YAW * geo["yaw_err"]), dim=-1,
    )
    cmd = torch.clamp(cmd, torch.tensor(COMMAND_LOW, device=env.device), torch.tensor(COMMAND_HIGH, device=env.device))
    speed = torch.linalg.vector_norm(env.scene["robot"].data.root_link_lin_vel_b[:, :2], dim=-1)
    ready = (
        (geo["prep_dist"] < BASELINE_POS_TOL)
        & (geo["yaw_err"].abs() < BASELINE_YAW_TOL)
        & (speed < BASELINE_SPEED_TOL)
    )
    trigger = torch.where(ready, 1.0, -1.0).unsqueeze(-1)
    return torch.cat((cmd, trigger), dim=-1)
