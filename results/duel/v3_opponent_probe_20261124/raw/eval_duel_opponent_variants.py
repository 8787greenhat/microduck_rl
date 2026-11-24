"""Frozen v3 actor robustness probe. A separate diagnostic, not original fixed600."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
from duel_opponent_variants import opponent_rule,VARIANTS

ROOT=Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
REPO=Path('/home/lingchentseng/projects/microduck_duel_v3_seed1_e8b9756')
PIN='e8b9756c03ae66a693275ee84b61a36aab367b2f'
CHECKPOINT=ROOT/'outputs/duel_v3_PPO_vs_A_seed1_20261124_091746/training/checkpoints/model_1999.pt'
EXPECTED='f50f7fade3a12abea06f12acdfc904d2029a57238ed864aaa940e73a1df9cfc2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def worker(a):
    assert sha(CHECKPOINT)==EXPECTED
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    from mjlab_microduck.duel.protocol import load_protocol,role_layout,EVAL_RNG_SEED,END_NAMES,FLAG_NAMES,summarize
    from mjlab_microduck.duel.runtime import initialize
    initialize(EVAL_RNG_SEED)
    import torch
    from mjlab_microduck.duel.v3_env import DuelTrainV3
    from mjlab_microduck.duel.v3_runner import DuelV3OnPolicyRunner,MicroduckDuelV3RlCfg
    from mjlab_microduck.duel.v3_runtime import v3_provenance
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.duel.contact import SENSORS
    protocol,layouts=load_protocol(REPO/'benchmarks/duel/v1')
    # Predeclared small diagnostic: original scene IDs 0..31, repeats 0 and 1.
    assignments=[(row,r) for row in layouts[:32] for r in range(2)]
    a.out.mkdir(exist_ok=False)
    manifest=dict(source=v3_provenance(),checkpoint_sha256=EXPECTED,official_fixed600=False,
        diagnostic=True,new_training=False,curriculum=False,self_play=False,variant=a.variant,
        opponent=VARIANTS[a.variant],learner_side=a.side,num_envs=64,scene_ids=list(range(32)),repeats=[0,1],
        eval_rng_seed=0,round_seconds=20,original_protocol=protocol,
        comparison='Only compare new diagnostic conditions with each other; batch64/subset differs from historical fixed600 batch300.',
        wrapper_sha256=sha(Path(__file__)),opponent_wrapper_sha256=sha(Path(__file__).with_name('duel_opponent_variants.py')))
    save(a.out/'manifest.json',manifest)
    original=mdp.duel_geometric_actions
    env=DuelTrainV3(64,EVAL_RNG_SEED,learner_index=a.side)
    rows=[];began=time.monotonic()
    try:
        checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        assert checkpoint['infos']['contract']=='duel-v3-actor76-critic76-normalized4'
        cfg=asdict(MicroduckDuelV3RlCfg);cfg.update(logger='tensorboard',obs_groups={'actor':['actor'],'critic':['critic']})
        runner=DuelV3OnPolicyRunner(env,cfg,None,'cuda:0')
        runner.alg.actor.load_state_dict(checkpoint['actor_state_dict'],strict=True)
        before={k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
        policy=runner.get_inference_policy(device='cuda:0')
        mdp.duel_geometric_actions=opponent_rule(original,a.variant,1-a.side)
        placed=[role_layout(row,a.side) for row,_ in assignments]
        env.env._duel_fixed_layout_override={'robots':torch.tensor([x['robots'] for x in placed],device=env.device),
            'ball':torch.tensor([x['ball'] for x in placed],device=env.device)}
        torch.manual_seed(EVAL_RNG_SEED);obs=env.reset();mdp.duel_assert_static_alignment(env.env)
        for index,name in enumerate(('robot','opponent')):
            assert torch.allclose(env.env.scene[name].data.root_link_pos_w[:,:2]-env.env.scene.env_origins[:,:2],env.env._duel_fixed_layout_override['robots'][:,index,:2],atol=1e-5)
        finished=torch.zeros(64,dtype=torch.bool,device=env.device)
        with torch.inference_mode():
            for step in range(200):
                obs,_,done,extra=env.step(policy(obs));fresh=done.bool()&~finished;stats=extra['duel_episode']
                for i in fresh.nonzero(as_tuple=False).flatten().tolist():
                    scene,repeat=assignments[i]
                    row=dict(scene_id=scene['scene_id'],repeat=repeat,learner_side=a.side,opponent_variant=a.variant,
                        end_reason=END_NAMES[int(stats['code'][i])],elapsed_s=round(float(stats['elapsed_s'][i]),5),
                        last_touch=int(stats['last_touch'][i]),learner_kick_attempts=int(stats['kick_count'][i,a.side]),
                        opponent_kick_attempts=int(stats['kick_count'][i,1-a.side]),training_reward_not_score=float(stats['return'][i]))
                    row.update({f'flag_{name}':bool(stats['flags'][i,j]) for j,name in enumerate(FLAG_NAMES)})
                    row.update({f'physics_samples_{name}':int(stats['contact_samples'][i,j]) for j,name in enumerate(SENSORS)})
                    rows.append(row)
                finished|=fresh
                if step%40==0 or finished.all():
                    print(f'{a.variant} side {a.side} step {step+1}: {int(finished.sum())}/64',flush=True)
                    save(a.out/'progress.json',dict(high_step=step+1,finished=int(finished.sum()),wall_seconds=time.monotonic()-began))
                if finished.all():break
        assert finished.all() and len(rows)==64
        assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in before.items())
        assert sha(CHECKPOINT)==EXPECTED
        rows.sort(key=lambda r:(r['scene_id'],r['repeat']))
        with (a.out/'per_episode.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        save(a.out/'episodes.json',rows)
        save(a.out/'summary.json',dict(complete=True,official=False,total=summarize(rows),actor_and_normalizer_unchanged=True,
            checkpoint_sha256=EXPECTED,per_episode_csv_sha256=sha(a.out/'per_episode.csv'),new_training=False,wall_seconds=time.monotonic()-began))
    finally:
        mdp.duel_geometric_actions=original;env.close()

def parent(a):
    a.out.mkdir(exist_ok=False)
    states=[]
    for variant in VARIANTS:
        for side in (0,1):
            folder=a.out/f'{variant}_side{side}'
            command=[__import__('sys').executable,str(Path(__file__)),'--variant',variant,'--side',str(side),'--out',str(folder)]
            save(a.out/'status.json',dict(state='running',variant=variant,side=side,finished=states,training=False))
            with (a.out/f'{variant}_side{side}.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            state=dict(variant=variant,side=side,returncode=result.returncode);states.append(state)
            if result.returncode:
                save(a.out/'status.json',dict(state='failed',workers=states,automatic_restart=False));raise RuntimeError(state)
            print(json.dumps(state),flush=True)
    save(a.out/'status.json',dict(state='complete',workers=states,training=False,checkpoint_sha256=EXPECTED))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=list(VARIANTS));p.add_argument('--side',type=int,choices=(0,1));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.variant is None:parent(a)
    else:worker(a)
