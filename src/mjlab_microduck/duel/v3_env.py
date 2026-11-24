"""Trainable fast-approach/stop controller with its memory exposed to PPO."""
from dataclasses import replace
import torch
from tensordict import TensorDict
from .stop_settle import StopSettleConfig, StopSettleController
from .stop_settle_env import DuelStopSettle
from .v2_env import DuelV2

TASK_CONTRACT='duel-v3-actor76-critic76-normalized4'
CONTROL_CONFIG=replace(StopSettleConfig(),capture_distance_m=.12,release_distance_m=.22,approach_speed_mps=.30)
OBS_EXTRA=('phase_follow','phase_align','phase_settle','intent_age_fraction','settle_fraction','cooldown_fraction')


class FastStopController(StopSettleController):
    def __init__(self,n,device):
        super().__init__(n,device,CONTROL_CONFIG)

    def prepare(self,actions,distance,error,delta,ball_speed,kicking,ready,dt=.1):
        result=super().prepare(actions,distance,error,delta,ball_speed,kicking,ready,dt)
        align=(self.phase==self.ALIGN)&~kicking
        # Nominal frozen-walk calibration: .08-.20 forward commands barely
        # moved; .30 moved reliably. Avoid an ineffective tiny-command crawl.
        norm=torch.linalg.vector_norm(delta,dim=1).clamp(min=1e-9)
        velocity=delta/norm.unsqueeze(-1)*self.config.approach_speed_mps
        scale=torch.tensor([.4,.3],device=actions.device)
        result[align,:2]=(velocity/scale)[align]
        return result


class DuelTrainV3(DuelStopSettle):
    def reset(self,ids=None):
        if self.stop_settle is None:
            self.stop_settle=FastStopController(self.num_envs,self.device)
        return super().reset(ids)

    def get_observations(self):
        base=DuelV2.get_observations(self)['actor']
        state=self.stop_settle;c=state.config
        phase=torch.nn.functional.one_hot(state.phase,num_classes=3).to(base.dtype)
        timers=torch.stack((state.age/c.intent_timeout_seconds,
                            state.settle/c.settle_seconds,state.cooldown/c.cooldown_seconds),1).clamp(0,1)
        value=torch.cat((base,phase,timers),1)
        assert value.shape==(self.num_envs,76) and torch.isfinite(value).all()
        return TensorDict({'actor':value,'critic':value.clone()},batch_size=[self.num_envs])

    def step(self,actions):
        _,reward,done,extra=super().step(actions)
        # Parent computes observations before clearing an accepted intent.
        # Expose the final post-transition/reset controller state to the actor.
        state=self.stop_settle
        extra['log'].update({'DuelV3/phase_align':(state.phase==1).float().mean(),
                             'DuelV3/phase_settle':(state.phase==2).float().mean(),
                             'DuelV3/kick_starts':self.last_control['kick_starts'].float().mean()})
        return self.get_observations(),reward,done,extra
