"""One 4D high-level learner versus frozen method A; independent dual-robot worlds."""
import math
import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.onnx_mlp import OnnxMlp
from mjlab_microduck.soccer_adapter import WALK_POLICY_ONNX, KICK_POLICY_ONNX
from mjlab_microduck.tasks.microduck_duel_env_cfg import make_microduck_duel_env_cfg
from mjlab_microduck.tasks import mdp
from .constants import ROBOT_NAMES
from .state import SkillState
from .contact import PhysicsContactLatch, SENSORS


def primary_end_code(flags):
    """One mutually exclusive outcome; raw overlapping flags are retained too."""
    win, loss, ball, fell, opp_fell, out, opp_out, timeout = flags.unbind(1)
    code = torch.zeros(len(flags), dtype=torch.long, device=flags.device)
    for condition, value in ((timeout,8),(ball,3),(opp_out,7),(out,6),(out & opp_out,10),
                             (opp_fell,5),(fell,4),(fell & opp_fell,9),(loss,2),(win,1)):
        code = torch.where(condition, value, code)
    return code


class DuelVsA(VecEnv):
    @property
    def unwrapped(self):
        return self.env

    def __init__(self, num_envs=64, seed=1, learner_index=0, learner_mode='ppo', base_env=None):
        if learner_index not in (0,1) or learner_mode not in ('ppo','A'):
            raise ValueError('Expected side 0/1 and learner mode ppo/A')
        self.learner_index=learner_index; self.opponent_index=1-learner_index
        self.learner_mode=learner_mode
        if base_env is None:
            cfg=make_microduck_duel_env_cfg(num_envs); cfg.seed=seed
            base_env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        self.env=base_env; self.cfg=base_env.cfg; self.num_envs=base_env.num_envs
        self.num_actions=4; self.device=torch.device(base_env.device); n=self.num_envs
        self.max_episode_length=200
        self.episode_length_buf=torch.zeros(n,dtype=torch.long,device=self.device)
        self.elapsed=torch.zeros(n,device=self.device)
        self.skill_steps=torch.zeros(n,dtype=torch.long,device=self.device)
        self.skills=SkillState(n,self.device); self.rule=SkillState(n,self.device)
        self.walk=OnnxMlp(WALK_POLICY_ONNX,str(self.device)); self.kick=OnnxMlp(KICK_POLICY_ONNX,str(self.device))
        self.contact=PhysicsContactLatch(self.env)
        self.contact_counts=torch.zeros(n,len(SENSORS),dtype=torch.long,device=self.device)
        self.ever_kicked=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.episode_rewards=torch.zeros(n,device=self.device)
        self.total_completed=0; self.total_wins=0; self.total_losses=0
        assert abs(self.env.physics_dt-.005)<1e-9 and self.cfg.decimation==4
        self.reset(); mdp.duel_assert_static_alignment(self.env)

    def reset(self,ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if ids is None else ids
        self.low_obs,_=self.env.reset(env_ids=ids)
        self.skills.reset(ids); self.rule.reset(ids); self.contact.reset(ids)
        for value in (self.episode_length_buf,self.elapsed,self.skill_steps,self.ever_kicked,self.episode_rewards,self.contact_counts):
            value[ids]=0
        return self.get_observations()

    def get_observations(self):
        return TensorDict(mdp.duel_observations(self.env,self.skills,self.low_obs,self.learner_index),batch_size=[self.num_envs])

    def low_actions(self):
        obs=torch.stack((self.low_obs['skill_a'],self.low_obs['skill_b']),dim=1).clone()
        obs[:,:,48:51]=torch.where(self.skills.mode.unsqueeze(-1),torch.zeros_like(self.skills.command),self.skills.command)
        obs=obs.reshape(-1,61)
        values=torch.where(self.skills.mode.reshape(-1,1),self.kick(obs),self.walk(obs))
        return values.reshape(self.num_envs,28)

    def step(self,actions):
        assert actions.shape==(self.num_envs,4) and torch.isfinite(actions).all(), 'Invalid high-level actions'
        li,oi=self.learner_index,self.opponent_index
        self.episode_length_buf+=1
        rule_starts=mdp.duel_geometric_actions(self.env,self.rule)  # exactly once at 10 Hz
        for name in ('mode','left','command','settled','kick_count'):
            source=getattr(self.rule,name); target=getattr(self.skills,name)
            target[:,oi]=source[:,oi]
            if self.learner_mode=='A': target[:,li]=source[:,li]
        if self.learner_mode=='A':
            starts=rule_starts[:,li]
        else:
            self.skills.command[:,li]=torch.clamp(actions[:,:3],torch.tensor([-.4,-.3,-1.],device=self.device),torch.tensor([.4,.3,1.],device=self.device))
            triggers=torch.zeros(self.num_envs,2,dtype=torch.bool,device=self.device); triggers[:,li]=actions[:,3]>0
            starts=self.skills.start(triggers)[:,li]
        self.ever_kicked|=starts
        terms={'kick_attempt':-.5*starts.float()}
        done=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device); timeouts=torch.zeros_like(done)
        end_flags=torch.zeros(self.num_envs,8,dtype=torch.bool,device=self.device)
        end_elapsed=torch.zeros_like(self.elapsed)
        sampled200=torch.zeros(self.num_envs,5,dtype=torch.bool,device=self.device); sampled50=torch.zeros_like(sampled200)
        for _ in range(5):
            active=~done
            prev_dist,prev_err,_=mdp.duel_geometry(self.env,li)
            previous=self.env.scene['ball'].data.root_link_pos_w.clone()
            self.contact.begin(active)
            self.low_obs,_,_,_,_=self.env.step(self.low_actions())
            assert self.contact.samples==4
            self.contact_counts+=self.contact.count
            sampled200|=self.contact.any
            sampled50|=torch.stack([mdp.duel_contact(self.env,n) for n in SENSORS],dim=1)&active.unsqueeze(-1)
            self.skills.tick(); self.rule.tick(); self.skill_steps+=active.long(); self.elapsed[:]=.02*self.skill_steps
            ball=self.env.scene['ball'].data.root_link_pos_w
            for name in (*ROBOT_NAMES,'ball'):
                data=self.env.scene[name].data
                assert all(torch.isfinite(value).all() for value in (data.root_link_pos_w,data.root_link_quat_w,data.root_link_lin_vel_w,data.root_link_ang_vel_b)), f'Nonfinite {name} state'
            goal=mdp.duel_goal_crossings(previous,ball,self.env.scene.env_origins)
            win=active&goal[:,li]; loss=active&goal[:,oi]&~win
            local_ball=ball-self.env.scene.env_origins
            ball_out=(local_ball[:,0].abs()>1.035)|(local_ball[:,1].abs()>.635)
            falls=[]; outs=[]
            for name in (ROBOT_NAMES[li],ROBOT_NAMES[oi]):
                asset=self.env.scene[name]; pos=asset.data.root_link_pos_w-self.env.scene.env_origins
                falls.append(asset.data.projected_gravity_b[:,2]>-math.cos(math.radians(70)))
                outs.append((pos[:,0].abs()>1)|(pos[:,1].abs()>.6))
            unsafe=active&(ball_out|falls[0]|falls[1]|outs[0]|outs[1])&~win&~loss
            timeout=active&(self.skill_steps>=1000)&~win&~loss&~unsafe
            new_done=win|loss|unsafe|timeout
            flags=torch.stack((win,loss,ball_out,falls[0],falls[1],outs[0],outs[1],timeout),1)&active.unsqueeze(-1)
            end_flags[new_done]=flags[new_done]; end_elapsed[new_done]=self.elapsed[new_done]
            dist,err,_=mdp.duel_geometry(self.env,li)
            reward_terms=mdp.duel_rewards(prev_dist,dist,prev_err,err,~self.ever_kicked,active,win,loss,unsafe,timeout)
            for key,value in reward_terms.items(): terms[key]=terms.get(key,torch.zeros_like(value))+value
            done|=new_done; timeouts|=timeout
        reward=sum(terms.values()); assert torch.isfinite(reward).all()
        self.episode_rewards+=reward
        codes=primary_end_code(end_flags)
        extras={'time_outs':timeouts,'log':{f'Duel_Reward/{k}':v.mean() for k,v in terms.items()}}
        extras['duel_episode']={'code':codes,'flags':end_flags,'elapsed_s':end_elapsed,
            'return':self.episode_rewards.clone(),'last_touch':self.contact.last_touch.clone(),
            'kick_count':self.skills.kick_count.clone(),'contact_samples':self.contact_counts.clone()}
        extras['log'].update({'Duel/contact_samples_missed_at_50Hz':(sampled200&~sampled50).float().sum(),
            'Duel/wins':(codes==1).float().mean(),'Duel/losses':(codes==2).float().mean(),'Duel/ends':done.float().mean()})
        self.total_wins+=int((codes==1).sum()); self.total_losses+=int((codes==2).sum()); self.total_completed+=int(done.sum())
        if done.any():
            extras['log']['Duel/completed_return']=self.episode_rewards[done].mean()
            self.reset(done.nonzero(as_tuple=False).squeeze(-1))
        observations=self.get_observations()
        assert torch.isfinite(observations['actor']).all() and torch.isfinite(observations['critic']).all()
        return observations,reward,done.long(),extras

    def close(self):
        self.contact.close(); self.env.close()
