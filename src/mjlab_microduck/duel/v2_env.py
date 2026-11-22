"""Finite duel v2: normalized bounded controls, recoverable shaping, ready kicks."""
import math
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_duel_v2_env_cfg import (
    make_microduck_duel_v2_env_cfg,ACTION_SCALE,GAMMA,READY_HOLD_SECONDS,
)
from .env import DuelVsA,primary_end_code
from .constants import ROBOT_NAMES
from .contact import SENSORS


class DuelV2(DuelVsA):
    def __init__(self,num_envs=64,seed=1,learner_index=0,learner_mode='ppo',base_env=None):
        self.ready_time=None
        if base_env is None:
            cfg=make_microduck_duel_v2_env_cfg(num_envs);cfg.seed=seed
            base_env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        super().__init__(num_envs,seed,learner_index,learner_mode,base_env)

    def reset(self,ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if ids is None else ids
        if self.ready_time is None:self.ready_time=torch.zeros(self.num_envs,device=self.device)
        self.ready_time[ids]=0
        return super().reset(ids)

    def get_observations(self):
        return TensorDict(mdp.duel_v2_observations(self.env,self.skills,self.low_obs,
            self.ready_time,self.elapsed,self.contact.last_touch,self.learner_index),batch_size=[self.num_envs])

    def step(self,actions):
        assert actions.shape==(self.num_envs,4) and torch.isfinite(actions).all()
        assert (actions.abs()<=1+1e-6).all(),'v2 requires normalized bounded actions'
        li,oi=self.learner_index,self.opponent_index
        self.episode_length_buf+=1
        potential_before=mdp.duel_v2_potential(self.env,li)
        ready=mdp.duel_v2_ready(self.env,li)&~self.skills.mode[:,li]
        self.ready_time[:]=torch.where(ready,self.ready_time+.1,torch.zeros_like(self.ready_time))
        allowed=self.ready_time>=READY_HOLD_SECONDS
        rule_starts=mdp.duel_geometric_actions(self.env,self.rule)
        for name in ('mode','left','command','settled','kick_count'):
            target=getattr(self.skills,name);source=getattr(self.rule,name)
            target[:,oi]=source[:,oi]
            if self.learner_mode=='A':target[:,li]=source[:,li]
        requests=actions[:,3]>0
        if self.learner_mode=='A':starts=rule_starts[:,li]
        else:
            self.skills.command[:,li]=actions[:,:3]*torch.tensor(ACTION_SCALE,device=self.device)
            triggers=torch.zeros(self.num_envs,2,dtype=torch.bool,device=self.device)
            triggers[:,li]=requests&allowed
            starts=self.skills.start(triggers)[:,li]
        self.ready_time[starts]=0
        self.ever_kicked|=starts  # retained diagnostic state; never gates rewards
        terms={'kick_attempt':torch.zeros(self.num_envs,device=self.device)}
        done=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        end_flags=torch.zeros(self.num_envs,8,dtype=torch.bool,device=self.device)
        end_elapsed=torch.zeros_like(self.elapsed)
        sampled200=torch.zeros(self.num_envs,5,dtype=torch.bool,device=self.device)
        sampled50=torch.zeros_like(sampled200)
        for _ in range(5):
            active=~done;previous=self.env.scene['ball'].data.root_link_pos_w.clone()
            self.contact.begin(active)
            self.low_obs,_,_,_,_=self.env.step(self.low_actions())
            assert self.contact.samples==4
            self.contact_counts+=self.contact.count;sampled200|=self.contact.any
            sampled50|=torch.stack([mdp.duel_contact(self.env,n) for n in SENSORS],1)&active.unsqueeze(-1)
            self.skills.tick();self.rule.tick();self.skill_steps+=active.long();self.elapsed[:]=.02*self.skill_steps
            for name in (*ROBOT_NAMES,'ball'):
                data=self.env.scene[name].data
                assert all(torch.isfinite(x).all() for x in (data.root_link_pos_w,data.root_link_quat_w,data.root_link_lin_vel_w,data.root_link_ang_vel_b))
            ball=self.env.scene['ball'].data.root_link_pos_w
            goal=mdp.duel_goal_crossings(previous,ball,self.env.scene.env_origins)
            win=active&goal[:,li];loss=active&goal[:,oi]&~win
            local_ball=ball-self.env.scene.env_origins
            ball_out=(local_ball[:,0].abs()>1.035)|(local_ball[:,1].abs()>.635)
            falls=[];outs=[]
            for name in (ROBOT_NAMES[li],ROBOT_NAMES[oi]):
                asset=self.env.scene[name];pos=asset.data.root_link_pos_w-self.env.scene.env_origins
                falls.append(asset.data.projected_gravity_b[:,2]>-math.cos(math.radians(70)))
                outs.append((pos[:,0].abs()>1)|(pos[:,1].abs()>.6))
            unsafe=active&(ball_out|falls[0]|falls[1]|outs[0]|outs[1])&~win&~loss
            timeout=active&(self.skill_steps>=1000)&~win&~loss&~unsafe
            new_done=win|loss|unsafe|timeout
            flags=torch.stack((win,loss,ball_out,falls[0],falls[1],outs[0],outs[1],timeout),1)&active.unsqueeze(-1)
            end_flags[new_done]=flags[new_done];end_elapsed[new_done]=self.elapsed[new_done]
            values=mdp.duel_v2_terminal_rewards(win,loss,unsafe,timeout,ball_out,falls[0],falls[1],outs[0],outs[1],self.contact.last_touch,li)
            values['time']=-.1*.02*active.float()
            for key,value in values.items():terms[key]=terms.get(key,torch.zeros_like(value))+value
            done|=new_done
        potential_after=mdp.duel_v2_potential(self.env,li)
        terms['potential']=mdp.duel_v2_potential_reward(potential_before,potential_after,done,GAMMA)
        reward=sum(terms.values());assert torch.isfinite(reward).all()
        self.episode_rewards+=reward;codes=primary_end_code(end_flags)
        # Every timeout is a true finite-match terminal. Do NOT provide the
        # rsl-rl time_outs bootstrap key for these 20-second match outcomes.
        extras={'log':{f'DuelV2_Reward/{key}':value.mean() for key,value in terms.items()}}
        extras['duel_episode']={'code':codes,'flags':end_flags,'elapsed_s':end_elapsed,
            'return':self.episode_rewards.clone(),'last_touch':self.contact.last_touch.clone(),
            'kick_count':self.skills.kick_count.clone(),'contact_samples':self.contact_counts.clone()}
        extras['log'].update({'Duel/wins':(codes==1).float().mean(),'Duel/losses':(codes==2).float().mean(),
            'Duel/ends':done.float().mean(),'DuelV2/ready_fraction':allowed.float().mean(),
            'DuelV2/kick_starts':starts.float().mean(),'DuelV2/blocked_kick_requests':(requests&~allowed).float().mean(),
            'DuelV2/actions_near_bound':(actions[:,:3].abs()>.95).float().mean(),
            'DuelV2/contact_windows_missed_at_50Hz':(sampled200&~sampled50).float().sum()})
        self.total_wins+=int((codes==1).sum());self.total_losses+=int((codes==2).sum());self.total_completed+=int(done.sum())
        if done.any():
            extras['log']['Duel/completed_return']=self.episode_rewards[done].mean()
            self.reset(done.nonzero(as_tuple=False).squeeze(-1))
        obs=self.get_observations()
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all()
        return obs,reward,done.long(),extras
