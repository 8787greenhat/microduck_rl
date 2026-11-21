"""Portable PPO-vs-fixed-A entry point; defaults are a 64 x 5 smoke test."""
import argparse
from dataclasses import asdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs',type=int,default=64); p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--seed',type=int,default=1); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.num_envs<1 or a.iterations<1: p.error('Positive environment/iteration counts required')
    a.out.mkdir(parents=True,exist_ok=False)
    from mjlab_microduck.duel.runtime import initialize,provenance,save,sha
    initialize(a.seed)
    import torch
    from mjlab_microduck.duel.env import DuelVsA
    from mjlab_microduck.duel.runner import DuelOnPolicyRunner,MicroduckDuelRlCfg
    from mjlab_microduck.duel.constants import REWARD_SPEC,TASK_CONTRACT
    from mjlab_microduck.tasks.mdp import duel_assert_static_alignment
    env=DuelVsA(a.num_envs,a.seed)
    cfg=asdict(MicroduckDuelRlCfg)
    cfg.update(seed=a.seed,logger='tensorboard',max_iterations=a.iterations,save_interval=min(100,a.iterations),obs_groups={'actor':['actor'],'critic':['critic']})
    manifest={**provenance(),'task':'duel_ppo_vs_fixed_A_v1','contract':TASK_CONTRACT,'seed':a.seed,'num_envs':a.num_envs,
        'iterations':a.iterations,'PPO':cfg,'reward_spec':REWARD_SPEC,'curriculum':False,'training_side':0,
        'opponent':'fixed geometric A','checkpoint_resume_supported_by_this_entrypoint':False}
    save(a.out/'manifest.json',manifest)
    runner=DuelOnPolicyRunner(env,cfg,str(a.out/'checkpoints'),'cuda:0')
    initial={k:v.detach().cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
    critic={k:v.detach().cpu().clone() for k,v in runner.alg.critic.state_dict().items()}
    frozen={name:{k:v.detach().cpu().clone() for k,v in policy.weights.items()} for name,policy in (('walk',env.walk),('kick',env.kick))}
    try:
        runner.learn(a.iterations)
        final=a.out/f'checkpoints/model_{a.iterations-1}.pt'
        data=torch.load(final,map_location='cpu',weights_only=True)
        changed=[k for k in initial if k.startswith('mlp.') and not torch.equal(initial[k],data['actor_state_dict'][k])]
        critic_changed=[k for k in critic if k.startswith('mlp.') and not torch.equal(critic[k],data['critic_state_dict'][k])]
        assert changed and critic_changed and data['iter']==a.iterations-1
        assert all(torch.isfinite(v).all() for group in ('actor_state_dict','critic_state_dict') for v in data[group].values() if isinstance(v,torch.Tensor))
        steps={int(v['step']) for v in data['optimizer_state_dict']['state'].values() if 'step' in v}
        assert steps=={a.iterations*cfg['algorithm']['num_learning_epochs']*cfg['algorithm']['num_mini_batches']}
        assert all(torch.equal(v.detach().cpu(),frozen[name][k]) for name,policy in (('walk',env.walk),('kick',env.kick)) for k,v in policy.weights.items())
        duel_assert_static_alignment(env.env)
        save(a.out/'verification.json',{'verified':True,'actor_changed':changed,'critic_changed':critic_changed,
            'optimizer_steps':sorted(steps),'checkpoint_sha256':sha(final),'frozen_skills_unchanged':True,
            'parallel_goals_and_pitch_aligned':True,'finite':True,'competitive_performance_claimed':False})
    finally: env.close()

if __name__=='__main__': main()
