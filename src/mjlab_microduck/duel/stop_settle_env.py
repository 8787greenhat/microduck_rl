"""Opt-in inference candidate; does not alter DuelV2 or claim a new PPO model."""
import torch
from mjlab_microduck.tasks import mdp
from .v2_env import DuelV2
from .stop_settle import StopSettleController


class DuelStopSettle(DuelV2):
    def __init__(self, num_envs=64, seed=1, learner_index=0, base_env=None):
        self.stop_settle = None
        self.last_control = None
        super().__init__(num_envs, seed, learner_index, 'ppo', base_env)

    def reset(self, ids=None):
        if self.stop_settle is None:
            self.stop_settle = StopSettleController(self.num_envs, self.device)
        indices = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        self.stop_settle.reset(indices)
        return super().reset(ids)

    def step(self, actions):
        li = self.learner_index
        distance, error, delta = mdp.duel_geometry(self.env, li)
        ball_speed = torch.linalg.vector_norm(self.env.scene['ball'].data.root_link_lin_vel_w[:, :2], dim=1)
        ready = mdp.duel_v2_ready(self.env, li) & ~self.skills.mode[:, li]
        applied = self.stop_settle.prepare(actions, distance, error, delta, ball_speed,
                                           self.skills.mode[:, li], ready)
        before = self.skills.kick_count[:, li].clone()
        phase = self.stop_settle.phase.clone()
        # Original physics, official skills, real speed guard and scoring path.
        obs, reward, done, extra = super().step(applied)
        stats = extra['duel_episode']
        starts = stats['kick_count'][:, li] > before
        self.stop_settle.accepted(starts & ~done.bool())
        self.last_control = {'requested_actions': actions.detach().clone(), 'applied_actions': applied.detach().clone(),
                             'phase_before_physics': phase, 'kick_starts': starts.detach().clone()}
        return obs, reward, done, extra
