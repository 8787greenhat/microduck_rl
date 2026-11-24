"""Bounded inference checks of the stop/settle candidate; never starts training."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source',choices=('synthetic_request','policy'),required=True)
    p.add_argument('--layout',choices=('approach','fixed'),default='approach')
    p.add_argument('--controller',choices=('stop_settle','original_v2'),default='stop_settle')
    p.add_argument('--side',type=int,choices=(0,1),default=0)
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.num_envs<=64 or not 1<=a.steps<=200:p.error('Short inference checks are limited to <=64 envs and <=200 high-level steps')
    a.out.mkdir(parents=True,exist_ok=False)
    gc.disable()  # Bounded diagnostic workaround for the previously observed GC/import crash.
    from mjlab_microduck.duel.runtime import initialize,verify_skills
    initialize(0)
    import torch
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.duel.v2_env import DuelV2
    from mjlab_microduck.duel.stop_settle_env import DuelStopSettle
    from mjlab_microduck.duel.v2_runner import DuelV2OnPolicyRunner,MicroduckDuelV2RlCfg
    from mjlab_microduck.duel.protocol import load_protocol,role_layout,END_NAMES
    root=Path(__file__).resolve().parents[1]
    checkpoint_hash=sha(a.checkpoint)
    frozen_hash=verify_skills()
    cls=DuelStopSettle if a.controller=='stop_settle' else DuelV2
    env=cls(a.num_envs,0,learner_index=a.side)
    began=time.monotonic()
    try:
        data=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        assert data['infos']['task']=='duel_ppo_vs_fixed_A_v2' and data['iter']==1999
        cfg=asdict(MicroduckDuelV2RlCfg);cfg.update(logger='tensorboard',obs_groups={'actor':['actor'],'critic':['critic']})
        runner=DuelV2OnPolicyRunner(env,cfg,None,'cuda:0')
        actor=runner.alg.actor;actor.load_state_dict(data['actor_state_dict'],strict=True)
        before={key:value.clone() for key,value in actor.state_dict().items()}
        policy=runner.get_inference_policy(device='cuda:0')
        if a.layout=='approach':
            ball=torch.zeros(a.num_envs,2,device=env.device)
            prep=mdp.soccer_preparation_pose(ball,torch.tensor([[1.,0.]],device=env.device).repeat(a.num_envs,1))
            poses=torch.zeros(a.num_envs,2,3,device=env.device);poses[:,0]=prep
            # Four prespecified starts: exact ready pose and increasingly farther approaches.
            offsets=torch.tensor([[0.,0.,0.],[-.04,.005,.04],[-.10,-.01,-.08],[-.16,.015,.12]],device=env.device)
            group=torch.arange(a.num_envs,device=env.device)//max(1,a.num_envs//4);group.clamp_(max=3)
            poses[:,0]+=offsets[group]
            poses[:,1]=torch.tensor([.75,.45,3.141592653589793],device=env.device)
            if a.side==1:
                rotated=poses.clone();rotated[:,:,:2]*=-1
                rotated[:,:,2]=mdp.duel_wrap(rotated[:,:,2]+3.141592653589793)
                poses=rotated[:,[1,0]];ball=-ball
            labels=[f'approach_group_{int(g)}' for g in group.cpu()]
        else:
            manifest,layouts=load_protocol(root/'benchmarks/duel/v1')
            assignments=[(scene,repeat) for scene in layouts[:32] for repeat in range(2)][:a.num_envs]
            placed=[role_layout(scene,a.side) for scene,_ in assignments]
            poses=torch.tensor([v['robots'] for v in placed],device=env.device)
            ball=torch.tensor([v['ball'] for v in placed],device=env.device)
            labels=[f'scene_{scene["scene_id"]}_repeat_{repeat}' for scene,repeat in assignments]
        env.env._duel_fixed_layout_override={'robots':poses,'ball':ball}
        torch.manual_seed(0);obs=env.reset()
        finished=torch.zeros(a.num_envs,dtype=torch.bool,device=env.device)
        any_kick=torch.zeros_like(finished);episodes=[];trace=[];kick_events=[]
        foot_before_first_kick=torch.full((a.num_envs,),-1,dtype=torch.long,device=env.device)
        for step in range(a.steps):
            with torch.inference_mode():
                alive=~finished
                if a.source=='policy':actions=policy(obs)
                else:
                    actions=torch.tensor([[.9,.3,.4,1.]],device=env.device).repeat(a.num_envs,1)
                distance,error,_=mdp.duel_geometry(env.env,a.side)
                speed=torch.linalg.vector_norm(env.env.scene['robot' if a.side==0 else 'opponent'].data.root_link_lin_vel_b[:,:2],dim=1)
                ball_speed=torch.linalg.vector_norm(env.env.scene['ball'].data.root_link_lin_vel_w[:,:2],dim=1)
                ready=mdp.duel_v2_ready(env.env,a.side)
                before_count=env.skills.kick_count[:,a.side].clone()
                foot_before_step=env.contact_counts[:,a.side+2].clone()
                obs,reward,done,extra=env.step(actions);stats=extra['duel_episode']
                starts=(stats['kick_count'][:,a.side]>before_count)&alive
                assert (ready[starts]).all(), 'Candidate bypassed original physical readiness guard'
                first_start=starts&(foot_before_first_kick<0)
                foot_before_first_kick[first_start]=foot_before_step[first_start]
                for index in starts.nonzero(as_tuple=False).flatten().tolist():
                    kick_events.append({'env_index':index,'high_step':step,'distance_m':float(distance[index]),
                        'abs_heading_rad':float(error[index].abs()),'base_speed_mps':float(speed[index]),
                        'ball_speed_mps':float(ball_speed[index]),'original_ready_guard_passed':bool(ready[index])})
                any_kick|=starts
                if step<40 or starts.any():
                    for index in (0,min(a.num_envs-1,a.num_envs//2)):
                        if not alive[index]:continue
                        control=env.last_control if a.controller=='stop_settle' else None
                        trace.append({'high_step':step,'env_index':index,'start_group':labels[index],
                            'distance':float(distance[index]),'heading_error':float(error[index]),
                            'base_speed':float(speed[index]),'ball_speed':float(ball_speed[index]),
                            'raw_request':float(actions[index,3]),'phase':int(control['phase_before_physics'][index]) if control else None,
                            'physical_command':(control['applied_actions'][index,:3]*torch.tensor([.4,.3,1.],device=env.device)).tolist() if control else None,
                            'kick_start':bool(starts[index])})
                fresh=done.bool()&alive
                for index in fresh.nonzero(as_tuple=False).flatten().tolist():
                    episodes.append({'env_index':index,'start_group':labels[index],'end_reason':END_NAMES[int(stats['code'][index])],
                        'elapsed_s':float(stats['elapsed_s'][index]),'kick_count':int(stats['kick_count'][index,a.side]),
                        'rightfoot_contact_samples':int(stats['contact_samples'][index,a.side+2]),
                        'rightfoot_contact_samples_after_first_kick':max(0,int(stats['contact_samples'][index,a.side+2]-foot_before_first_kick[index])) if foot_before_first_kick[index]>=0 else 0,
                        'last_touch':int(stats['last_touch'][index])})
                finished|=fresh
                if step%20==0 or finished.all():print(f'[short] {a.source}/{a.controller} step {step+1}: done {int(finished.sum())}, kick rounds {int(any_kick.sum())}',flush=True)
                if finished.all():break
        assert all(torch.equal(value,actor.state_dict()[key]) for key,value in before.items())
        assert sha(a.checkpoint)==checkpoint_hash and verify_skills()==frozen_hash
        assert not runner.alg.optimizer.state, 'Inference must not perform any PPO optimizer step'
        goals=[r for r in episodes if r['end_reason']=='goal_for']
        skill_goals=[r for r in goals if r['kick_count']>0 and r['rightfoot_contact_samples_after_first_kick']>0 and r['last_touch']==a.side]
        result={'complete':True,'training':False,'optimizer_steps':0,'source':a.source,'controller':a.controller,
            'layout':a.layout,'side':a.side,'environments':a.num_envs,'step_cap':a.steps,
            'completed_episodes':len(episodes),'unfinished':a.num_envs-len(episodes),
            'kick_skill_rounds':int(any_kick.sum()),'goal_rounds':len(goals),'goals_after_kick_and_rightfoot_contact':len(skill_goals),
            'checkpoint_sha256':checkpoint_hash,'actor_and_normalizer_unchanged':True,'official_skill_sha256':frozen_hash,
            'original_v2_readiness_gate_not_bypassed':True,'not_official_fixed600_score':True,
            'controller_is_hand_designed_not_new_PPO_learning':True,
            'candidate_source_sha256':{name:sha(root/name) for name in ('src/mjlab_microduck/duel/stop_settle.py','src/mjlab_microduck/duel/stop_settle_env.py')},
            'episodes':episodes,'trace':trace,'kick_events':kick_events,'wall_seconds':time.monotonic()-began}
        save(a.out/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('trace','episodes','kick_events','candidate_source_sha256')},indent=2),flush=True)
    finally:env.close()

if __name__=='__main__':main()
