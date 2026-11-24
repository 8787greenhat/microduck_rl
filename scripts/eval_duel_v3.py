"""Fixed duel benchmark: 100 layouts x 3 repeats x 2 attack directions."""
import argparse
import csv
from pathlib import Path
import time
import json
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'benchmarks/duel/v1')
    p.add_argument('--prepare',action='store_true',help='Write/check the immutable protocol on CPU, then exit')
    source=p.add_mutually_exclusive_group(); source.add_argument('--baseline',action='store_true'); source.add_argument('--checkpoint',type=Path)
    p.add_argument('--smoke',action='store_true',help='Only 4 layouts x 1 repeat x 2 sides; NOT an official score')
    p.add_argument('--out',type=Path)
    p.add_argument('--worker-side',type=int,choices=(0,1),help=argparse.SUPPRESS)
    a=p.parse_args()
    from mjlab_microduck.duel.protocol import write_protocol,load_protocol,role_layout,summarize,sha,END_NAMES,FLAG_NAMES,EVAL_RNG_SEED
    if a.prepare:
        print(write_protocol(a.protocol)); return
    if not (a.baseline or a.checkpoint) or a.out is None: p.error('Use --baseline or --checkpoint, and a new --out folder')
    if a.worker_side is None: return parent(a)
    manifest,layouts=load_protocol(a.protocol)
    if a.smoke: layouts=layouts[:4]
    repeats=1 if a.smoke else manifest['repeats']
    assignments=[(row,repeat) for row in layouts for repeat in range(repeats)]
    a.out.mkdir(parents=True,exist_ok=False)
    from mjlab_microduck.duel.runtime import initialize,save
    from mjlab_microduck.duel.v3_runtime import v3_provenance as provenance
    initialize(EVAL_RNG_SEED)
    import torch
    from dataclasses import asdict
    from mjlab_microduck.duel.v3_env import DuelTrainV3 as DuelVsA
    from mjlab_microduck.duel.v3_runner import DuelV3OnPolicyRunner as DuelOnPolicyRunner,MicroduckDuelV3RlCfg as MicroduckDuelRlCfg
    from mjlab_microduck.tasks.mdp import duel_assert_static_alignment
    rows=[]; started=time.monotonic()
    evaluation_manifest={**provenance(),'protocol':manifest,'protocol_manifest_sha256':sha(a.protocol/'manifest.json'),
        'checkpoint_sha256':sha(a.checkpoint) if a.checkpoint else None,'policy':'PPO vs fixed A' if a.checkpoint else 'A vs A',
        'official':False,'single_direction_worker':True,'checkpoint_loading':'actor and frozen normalization only; no optimizer, clock or physics restore',
        'repeats_are':'same physical initial state; reset-level BAM RNG may vary; not independent training seeds'}
    save(a.out/'manifest.json',evaluation_manifest)
    for side in (a.worker_side,):
        if a.checkpoint:
            env=DuelVsA(len(assignments),EVAL_RNG_SEED,learner_index=side)
        else:
            from mjlab_microduck.duel.v2_env import DuelV2
            env=DuelV2(len(assignments),EVAL_RNG_SEED,learner_index=side,learner_mode='A')
        try:
            runner=None; actor_before=None
            if a.checkpoint:
                data=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
                if (data.get('infos') or {}).get('task')!='duel_ppo_vs_fixed_A_v3':
                    raise ValueError('Expected a duel PPO checkpoint, not a single-robot B/C checkpoint')
                evaluation_manifest['checkpoint_iteration']=int(data['iter'])
                evaluation_manifest['checkpoint_infos']=data.get('infos')
                save(a.out/'manifest.json',evaluation_manifest)
                cfg=asdict(MicroduckDuelRlCfg); cfg.update(logger='tensorboard',obs_groups={'actor':['actor'],'critic':['critic']})
                runner=DuelOnPolicyRunner(env,cfg,None,'cuda:0')
                runner.alg.actor.load_state_dict(data['actor_state_dict'],strict=True)
                actor_before={k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
                policy=runner.get_inference_policy(device='cuda:0')
            placed=[role_layout(row,side) for row,_ in assignments]
            env.env._duel_fixed_layout_override={
                'robots':torch.tensor([x['robots'] for x in placed],device=env.device),
                'ball':torch.tensor([x['ball'] for x in placed],device=env.device)}
            # Actor construction consumes RNG. Restore common RNG AFTER it, so BAM
            # reset samples are paired across baseline/checkpoints on each side.
            torch.manual_seed(EVAL_RNG_SEED)
            obs=env.reset(); duel_assert_static_alignment(env.env)
            for index,name in enumerate(('robot','opponent')):
                assert torch.allclose(env.env.scene[name].data.root_link_pos_w[:,:2]-env.env.scene.env_origins[:,:2],env.env._duel_fixed_layout_override['robots'][:,index,:2],atol=1e-5)
            assert torch.allclose(env.env.scene['ball'].data.root_link_pos_w[:,:2]-env.env.scene.env_origins[:,:2],env.env._duel_fixed_layout_override['ball'],atol=1e-5)
            finished=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
            with torch.inference_mode():
                for step in range(200):
                    actions=policy(obs) if a.checkpoint else torch.zeros(env.num_envs,4,device=env.device)
                    obs,_,done,extra=env.step(actions)
                    fresh=done.bool()&~finished; stats=extra['duel_episode']
                    for i in fresh.nonzero(as_tuple=False).flatten().tolist():
                        scene,repeat=assignments[i]; code=int(stats['code'][i]); assert code in END_NAMES
                        record={'scene_id':scene['scene_id'],'repeat':repeat,'learner_side':side,
                            'attack_direction':'+X' if side==0 else '-X','end_reason':END_NAMES[code],
                            'elapsed_s':round(float(stats['elapsed_s'][i]),5),'last_touch':int(stats['last_touch'][i]),
                            'learner_kick_attempts':int(stats['kick_count'][i,side]),'opponent_kick_attempts':int(stats['kick_count'][i,1-side]),
                            'training_reward_not_score':float(stats['return'][i])}
                        record.update({f'flag_{name}':bool(stats['flags'][i,j]) for j,name in enumerate(FLAG_NAMES)})
                        from mjlab_microduck.duel.contact import SENSORS
                        record.update({f'physics_samples_{name}':int(stats['contact_samples'][i,j]) for j,name in enumerate(SENSORS)})
                        rows.append(record)
                    finished|=fresh
                    if step%20==0 or finished.all():
                        save(a.out/'progress.json',{'side':side,'high_step':step+1,'first_episodes_finished':int(finished.sum()),'expected_this_side':env.num_envs,'elapsed_wall_s':time.monotonic()-started})
                        print(f'[duel eval] side={side} step={step+1} first episodes={int(finished.sum())}/{env.num_envs}',flush=True)
                    if finished.all(): break
            assert finished.all(), 'Evaluation exceeded the 20-second round budget'
            if actor_before is not None:
                assert all(torch.equal(value,runner.alg.actor.state_dict()[key]) for key,value in actor_before.items()), 'Evaluation mutated actor/normalization'
            duel_assert_static_alignment(env.env)
            save(a.out/f'side_{side}_episodes.json',sorted([r for r in rows if r['learner_side']==side],key=lambda r:(r['scene_id'],r['repeat'])))
        finally: env.close()
    rows.sort(key=lambda r:(r['learner_side'],r['scene_id'],r['repeat']))
    assert len(rows)==len(assignments) and len({(r['scene_id'],r['repeat'],r['learner_side']) for r in rows})==len(rows)
    with (a.out/'per_episode.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result={'complete':True,'official':False,'single_direction_worker':True,'policy':evaluation_manifest['policy'],'total':summarize(rows),
        'by_side':{str(s):summarize([r for r in rows if r['learner_side']==s]) for s in (a.worker_side,)},
        'wall_seconds':time.monotonic()-started,'per_episode_csv_sha256':sha(a.out/'per_episode.csv'),
        'training_started':False,'comparative_RL_performance_claimed':False}
    save(a.out/'summary.json',result); print(result,flush=True)


def parent(a):
    from mjlab_microduck.duel.protocol import load_protocol,summarize,sha
    from mjlab_microduck.duel.runtime import save
    protocol,_=load_protocol(a.protocol)
    a.out.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();workers=[];rows=[];manifests=[]
    for side in (0,1):
        command=[sys.executable,str(Path(__file__).resolve()),'--worker-side',str(side),'--protocol',str(a.protocol),
                 '--out',str(a.out/f'side_{side}')]
        if a.checkpoint:command+=['--checkpoint',str(a.checkpoint.resolve())]
        else:command+=['--baseline']
        if a.smoke:command+=['--smoke']
        state={'state':'running','current_side':side,'completed_sides':workers}
        save(a.out/'status.json',state)
        with (a.out/f'side_{side}.log').open('w') as log:
            result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        workers.append({'side':side,'returncode':result.returncode})
        if result.returncode:
            save(a.out/'status.json',{'state':'failed','workers':workers})
            raise RuntimeError(f'Side {side} failed; inspect {a.out}/side_{side}.log. No automatic retry.')
        child=a.out/f'side_{side}'
        data=json.loads((child/'summary.json').read_text(encoding='utf-8'))
        assert data['complete'] and data['total']['episodes']==(4 if a.smoke else 300)
        rows+=json.loads((child/f'side_{side}_episodes.json').read_text(encoding='utf-8'))
        manifests.append(json.loads((child/'manifest.json').read_text(encoding='utf-8')))
    assert manifests[0]['source_sha256']==manifests[1]['source_sha256']
    assert manifests[0]['checkpoint_sha256']==manifests[1]['checkpoint_sha256']
    rows.sort(key=lambda r:(r['learner_side'],r['scene_id'],r['repeat']))
    expected=8 if a.smoke else 600
    assert len(rows)==expected and len({(r['learner_side'],r['scene_id'],r['repeat']) for r in rows})==expected
    with (a.out/'per_episode.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'complete':True,'official':not a.smoke,'policy':'PPO v3 vs fixed A' if a.checkpoint else 'A vs A in v3',
        'total':summarize(rows),'by_side':{str(s):summarize([r for r in rows if r['learner_side']==s]) for s in (0,1)},
        'wall_seconds':time.monotonic()-began,'per_episode_csv_sha256':sha(a.out/'per_episode.csv'),
        'training_started':False,'execution':'Fresh process per direction; same v1 layouts, game rules and RNG seed 0'}
    save(a.out/'manifest.json',{**manifests[0],'single_direction_worker':False,'official':not a.smoke,
        'execution':'fresh process per direction','workers':workers,'protocol':protocol,
        'v3_controller_changes':['nearby effective-speed approach','observable stop/settle memory','original readiness guard'],
        'scene_geometry_and_game_scoring_changed':False})
    save(a.out/'summary.json',summary);save(a.out/'status.json',{'state':'complete','workers':workers})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
