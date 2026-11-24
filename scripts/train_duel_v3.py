"""Fresh v3 PPO-vs-A training; defaults are 64 environments x 5 iterations."""
import argparse
from dataclasses import asdict
import gc
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs',type=int,default=64);p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.num_envs<1 or a.iterations<1:p.error('Positive environment and iteration counts required')
    a.out.mkdir(parents=True,exist_ok=False)
    # Avoid cyclic-GC during the previously problematic native-module import,
    # but keep normal GC enabled during the multi-hour training itself.
    enabled=gc.isenabled();gc.disable()
    from mjlab_microduck.duel.runtime import initialize,save,sha
    initialize(a.seed)
    if enabled:gc.enable()
    import torch
    from mjlab_microduck.duel.v3_env import DuelTrainV3,CONTROL_CONFIG,TASK_CONTRACT,OBS_EXTRA
    from mjlab_microduck.duel.v3_runner import DuelV3OnPolicyRunner,MicroduckDuelV3RlCfg
    from mjlab_microduck.duel.v3_runtime import v3_provenance
    from mjlab_microduck.tasks.microduck_duel_v2_env_cfg import REWARD_SPEC
    from mjlab_microduck.tasks.mdp import duel_assert_static_alignment
    env=DuelTrainV3(a.num_envs,a.seed)
    cfg=asdict(MicroduckDuelV3RlCfg)
    cfg.update(seed=a.seed,logger='tensorboard',max_iterations=a.iterations,save_interval=100,
               obs_groups={'actor':['actor'],'critic':['critic']})
    manifest={**v3_provenance(),'task':'duel_ppo_vs_fixed_A_v3','contract':TASK_CONTRACT,'seed':a.seed,
        'num_envs':a.num_envs,'iterations':a.iterations,'PPO':cfg,'reward_spec':REWARD_SPEC,
        'control_config':asdict(CONTROL_CONFIG),'obs_dim':76,'observation_extra':OBS_EXTRA,
        'curriculum':False,'self_play':False,'opponent':'fixed geometric A','fresh_initialization':True,
        'old_or_smoke_checkpoint_loaded':False,'resume_supported':False,'gc_enabled_during_training':gc.isenabled(),
        'nominal_physics_and_fixed_benchmark_unchanged':True}
    save(a.out/'manifest.json',manifest)
    runner=DuelV3OnPolicyRunner(env,cfg,str(a.out/'checkpoints'),'cuda:0')
    initial={k:v.detach().cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
    critic={k:v.detach().cpu().clone() for k,v in runner.alg.critic.state_dict().items()}
    frozen={name:{k:v.detach().cpu().clone() for k,v in policy.weights.items()} for name,policy in (('walk',env.walk),('kick',env.kick))}
    try:
        torch.save(initial,a.out/'initial_actor_state.pt')
        runner.learn(a.iterations)
        final=a.out/f'checkpoints/model_{a.iterations-1}.pt'
        data=torch.load(final,map_location='cpu',weights_only=True)
        assert data['iter']==a.iterations-1 and data['infos']['contract']==TASK_CONTRACT
        changed=[k for k in initial if k.startswith('mlp.') and not torch.equal(initial[k],data['actor_state_dict'][k])]
        critic_changed=[k for k in critic if k.startswith('mlp.') and not torch.equal(critic[k],data['critic_state_dict'][k])]
        assert changed and critic_changed
        assert data['actor_state_dict']['mlp.0.weight'].shape==(128,76)
        assert all(torch.isfinite(v).all() for group in ('actor_state_dict','critic_state_dict') for v in data[group].values() if isinstance(v,torch.Tensor))
        steps={int(v['step']) for v in data['optimizer_state_dict']['state'].values() if 'step' in v}
        assert steps=={a.iterations*20}
        assert all(torch.equal(v.detach().cpu(),frozen[name][k]) for name,policy in (('walk',env.walk),('kick',env.kick)) for k,v in policy.weights.items())
        duel_assert_static_alignment(env.env)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import math
        acc=EventAccumulator(str(a.out/'checkpoints'),size_guidance={'scalars':0});acc.Reload()
        assert all(math.isfinite(e.value) for tag in acc.Tags()['scalars'] for e in acc.Scalars(tag))
        save(a.out/'verification.json',{'verified':True,'actor_changed':changed,'critic_changed':critic_changed,
            'optimizer_steps':sorted(steps),'finite':True,'frozen_skills_unchanged':True,'obs_dim':76,
            'checkpoint_sha256':sha(final),'fresh_initialization':True,'training_performance_not_formal_score':True})
    finally:env.close()

if __name__=='__main__':main()
