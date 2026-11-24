"""Inference-only matched v3 helper for a geometric high-level rule."""
from dataclasses import asdict
import torch
from mjlab_microduck.duel.v3_env import FastStopController,CONTROL_CONFIG
from mjlab_microduck.tasks.microduck_duel_v2_env_cfg import ACTION_SCALE,READY_HOLD_SECONDS
from mjlab_microduck.tasks import mdp

CONDITIONS={
    'P_H_vs_A':dict(learner='PPO',opponent='original_A',description='Frozen PPO + v3 helper versus original geometric A'),
    'P_H_vs_R_H':dict(learner='PPO',opponent='matched_rule',description='Frozen PPO versus geometric rule, both with the exact v3 helper and readiness guard'),
    'R_H_vs_A':dict(learner='geometric_rule',opponent='original_A',description='Geometric rule + v3 helper versus original geometric A'),
    'R_H_vs_R_H':dict(learner='geometric_rule',opponent='matched_rule',description='Both sides use geometric rule + exact same v3 helper; symmetry reference'),
}

def rule_request(delta,error):
    scale=torch.tensor(ACTION_SCALE,device=delta.device)
    command=torch.cat((2*delta,(2*error).unsqueeze(1)),1).clamp(-scale,scale)
    # Greedy deterministic high-level request. Helper captures only eligible near-ball intents.
    return torch.cat((command/scale,torch.ones(len(delta),1,device=delta.device)),1)

def rule_actions_for_env(env,index):
    _,error,delta=mdp.duel_geometry(env,index)
    return rule_request(delta,error)

class MatchedRuleControl:
    def __init__(self,n,device):
        self.controller=FastStopController(n,device)
        self.ready_time=torch.zeros(n,device=device)
        self.last_starts=torch.zeros(n,dtype=torch.bool,device=device)
        self.total_guard_verified_starts=0
        self.guard_violations=0

    def reset(self,ids):
        self.controller.reset(ids);self.ready_time[ids]=0;self.last_starts[ids]=False

    def prepare(self,distance,error,delta,ball_speed,kicking,ready):
        actions=rule_request(delta,error)
        applied=self.controller.prepare(actions,distance,error,delta,ball_speed,kicking,ready)
        self.ready_time[:]=torch.where(ready,self.ready_time+.1,torch.zeros_like(self.ready_time))
        allowed=self.ready_time>=READY_HOLD_SECONDS
        request=(applied[:,3]>0)&allowed
        assert torch.isfinite(applied).all() and (applied.abs()<=1+1e-6).all()
        return applied,request

    def after_physics(self,done):
        self.controller.accepted(self.last_starts & ~done.bool())

class InstalledMatchedOpponent:
    def __init__(self,env):
        self.env=env;self.index=env.opponent_index
        self.control=MatchedRuleControl(env.num_envs,env.device)
        assert asdict(self.control.controller.config)==asdict(env.stop_settle.config)==asdict(CONTROL_CONFIG)
        self.original_rule=mdp.duel_geometric_actions
        self.original_reset=env.reset;self.original_step=env.step
        mdp.duel_geometric_actions=self.rule
        env.reset=self.reset;env.step=self.step

    def reset(self,ids=None):
        indices=torch.arange(self.env.num_envs,device=self.env.device) if ids is None else ids
        self.control.reset(indices)
        return self.original_reset(ids)

    def rule(self,low_env,skills,**unused):
        oi=self.index
        distance,error,delta=mdp.duel_geometry(low_env,oi)
        ball_speed=torch.linalg.vector_norm(low_env.scene['ball'].data.root_link_lin_vel_w[:,:2],dim=1)
        kicking=skills.mode[:,oi]
        ready=mdp.duel_v2_ready(low_env,oi)&~kicking
        applied,request=self.control.prepare(distance,error,delta,ball_speed,kicking,ready)
        scale=torch.tensor(ACTION_SCALE,device=applied.device)
        skills.command[:,oi]=applied[:,:3]*scale
        triggers=torch.zeros_like(skills.mode);triggers[:,oi]=request
        starts=skills.start(triggers)
        self.control.last_starts[:]=starts[:,oi]
        fired=starts[:,oi]
        if fired.any():
            valid=(ready & (self.control.ready_time>=READY_HOLD_SECONDS)
                   & (self.control.controller.phase==self.control.controller.SETTLE)
                   & (self.control.controller.settle>=self.control.controller.config.settle_seconds))
            self.control.guard_violations+=int((fired&~valid).sum())
            assert valid[fired].all() and (applied[fired,:3]==0).all()
            self.control.total_guard_verified_starts+=int(fired.sum())
        self.control.ready_time[fired]=0
        return starts

    def step(self,actions):
        result=self.original_step(actions)
        self.control.after_physics(result[2])
        return result

    def close(self):
        mdp.duel_geometric_actions=self.original_rule
        self.env.reset=self.original_reset;self.env.step=self.original_step

def install_matched_opponent(env):return InstalledMatchedOpponent(env)
