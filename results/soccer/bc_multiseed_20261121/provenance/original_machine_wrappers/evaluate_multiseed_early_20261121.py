"""Pinned fixed-test evaluation of saved early checkpoints; no training or publication."""
import argparse
import collections
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
REPO=Path('/home/lingchentseng/projects/microduck_rl_4060_multiseed_1a8a105')
OLD=Path('/home/lingchentseng/projects/microduck_rl_4060_7fa6298')
OUT=ROOT/'outputs/bc_multiseed_analysis_20261121'
TRAIN=ROOT/'outputs/multiseed_20261120_legacy_seed1'
FIXED=ROOT/'outputs/seed1_evaluation/soccer_eval_set.json'
PIN='1a8a1053533210d6695454cf51026551f3e4dbbc'
FIXED_HASH='3a59dee2c22b5129fcb10685192630443a29550790ee5ccbffc04e414d053464'
SCRIPT_HASH='2d8bd0f9fc840aa54fa858181e16ccb88813b4fc4579db42c6b49c7a85ba0757'
DRIVER='/usr/lib/wsl/lib:/usr/lib/wsl/drivers/nvacegpu.inf_amd64_75303a74d1223c32'
POINTS=list(range(0,601,100))+[1999]
FLAGS=('goal','fell','robot_oob','ball_oob','timeout','kicked','contact')

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    pending.replace(path)

def protocol():
    assert sha(FIXED)==FIXED_HASH
    assert sha(REPO/'scripts/eval_soccer.py')==SCRIPT_HASH
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==PIN
    assert not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=REPO,text=True).strip()
    assert json.loads((TRAIN/'queue_status.json').read_text())['state']=='finished'

def plan():
    protocol()
    oldplan=json.loads((ROOT/'outputs/bc_checkpoint_eval_20261025/plan.json').read_text())
    oldjobs={(j['method'],j['iteration_0based']):j for j in oldplan['jobs']}
    jobs=[]
    for seed in (1,2,3):
        for it in POINTS:
            for method in ('B','C'):
                reuse=None
                if seed==1:
                    old=oldjobs[method,it]
                    source=Path(old['checkpoint'])
                    assert sha(source)==old['checkpoint_sha256']
                    reuse=ROOT/f'outputs/bc_checkpoint_eval_20261025/{method}/model_{it}'
                    assert (reuse/'verification.json').exists()
                else:
                    folder=TRAIN/f'formal_{method}_seed{seed}'
                    verification=json.loads((folder/'verification.json').read_text())
                    assert verification['verified'] and verification['iterations']==2000
                    source=Path(verification['actual_run_dir'])/f'model_{it}.pt'
                    if it==1999: reuse=folder/'evaluation'
                assert source.exists()
                jobs.append({'method':method,'training_seed':seed,'iteration_0based':it,
                    'completed_iterations':it+1,'training_high_level_transitions':(it+1)*49152,
                    'checkpoint':str(source),'checkpoint_sha256':sha(source),
                    'out':str(OUT/f'seed{seed}/{method}/model_{it}'),
                    'reuse':str(reuse) if reuse else None})
    return {'created_at_utc':now(),'commit_for_new_evaluations':PIN,'fixed_eval_set_sha256':FIXED_HASH,
        'evaluation_script_sha256':SCRIPT_HASH,'evaluation_rng_seed':0,'points_0based':POINTS,
        'early_window_completed_iterations':[1,601],'thresholds':[.7,.9],
        'layouts':100,'repeats':3,'new_GPU_evaluations':28,'reused_evaluations':20,
        'new_training':False,'modified_evaluator':False,'github_updates':False,
        'subgroup_boundaries_prespecified':{'robot_ball_distance_m':[.3,.45],
          'initial_heading_error_deg':[30,60],'ball_goal_distance_m':[.7,.9],
          'shot_line_obliqueness_deg':[10,20]},
        'notes':['model_0 is after one PPO update, not an untrained policy.',
          'Seed1 C resumed after model_200; physical episodes and RNG reinitialized.',
          'Original evaluator restores checkpoint clock; shared COM randomization changes with clock, matched B/C points share the same schedule.',
          'Final endpoints are reused; no interpolation across 601 to 2000 for early AUC.',
          'Thresholds are observed on saved checkpoint grid; stability refers only to measured points.'],
        'jobs':jobs}

def verify(job):
    import torch
    folder=Path(job['out'])
    summary=json.loads((folder/'summary.json').read_text())
    data=torch.load(folder/'per_episode.pt',map_location='cpu',weights_only=True)
    assert summary['envs']==300 and all(data[k].numel()==300 for k in (*FLAGS,'steps'))
    rows=[]
    for i in range(300):
        row={'layout_index_0based':i%100,'repeat_index_1based':i//100+1,
             **{k:int(data[k][i]) for k in FLAGS},'elapsed_s':round(float(data['steps'][i])*.1,3)}
        assert any(row[k] for k in FLAGS[:5]) and 0<row['elapsed_s']<=12.2
        rows.append(row)
    counts={k:sum(r[k] for r in rows) for k in FLAGS}
    assert all(math.isclose(summary['rates'][k],counts[k]/300,abs_tol=1e-6) for k in FLAGS)
    times=sorted(r['elapsed_s'] for r in rows if r['goal'])
    mean=sum(times)/len(times) if times else None
    assert mean is None and summary['mean_goal_time_s'] is None or mean is not None and math.isclose(mean,summary['mean_goal_time_s'],abs_tol=1e-5)
    with (folder/'per_episode.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    assert sha(job['checkpoint'])==job['checkpoint_sha256']
    result={**job,'verified':True,'goal_count':counts['goal'],'goal_rate':counts['goal']/300,
        'mean_goal_time_s':mean,'counts':counts,'summary_sha256':sha(folder/'summary.json'),
        'per_episode_sha256':sha(folder/'per_episode.pt'),'all_episodes_terminated':True,
        'fixed_eval_set_sha256':FIXED_HASH,'evaluation_rng_seed':0,
        'failure_combinations':dict(collections.Counter('+'.join(k for k in FLAGS[1:5] if r[k]) for r in rows if not r['goal']))}
    save(folder/'verification.json',result)
    return result

def child(job):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    protocol()
    assert os.environ.get('LD_LIBRARY_PATH')==DRIVER
    for k in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','MICRODUCK_WARM_START','MICRODUCK_SOCCER_WALK_ONNX','MICRODUCK_SOCCER_KICK_ONNX','CUDA_VISIBLE_DEVICES'):
        os.environ.pop(k,None)
    os.environ.update(CUDA_MODULE_LOADING='EAGER',WARP_CACHE_PATH=str(OLD/'.local-runtime/warp-cache-driver610'),CUDA_CACHE_PATH=str(OLD/'.local-runtime/cuda-cache-driver610'))
    os.chdir(REPO)
    import torch
    import mjlab.tasks
    import mjlab_microduck.tasks as tasks
    import mujoco_warp
    import warp as wp
    wp.init()
    for name in ('smooth','constraint','collision_driver','sensor','forward','passive','support','solver','derivative','io'):
        wp.load_module(module='mujoco_warp._src.'+name,device='cuda:0')
    original_load=torch.load
    expected={}
    def cpu_load(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).resolve()==Path(job['checkpoint']).resolve():
            kwargs.update(map_location='cpu',weights_only=True)
            state=original_load(path,*args,**kwargs)
            assert state['iter']==job['iteration_0based']
            assert state['infos']['env_state']['common_step_counter']==job['completed_iterations']*120
            expected.update(state['actor_state_dict'])
            return state
        return original_load(path,*args,**kwargs)
    torch.load=cpu_load
    original_policy=tasks.SoccerOnPolicyRunner.get_inference_policy
    def checked_policy(runner,*args,**kwargs):
        actual=runner.alg.actor.state_dict()
        assert expected and set(actual)==set(expected)
        assert all(torch.equal(actual[k].detach().cpu(),v) for k,v in expected.items())
        print('[analysis] Exact actor / normalization match; clock verified',flush=True)
        return original_policy(runner,*args,**kwargs)
    tasks.SoccerOnPolicyRunner.get_inference_policy=checked_policy
    spec=importlib.util.spec_from_file_location('unchanged_soccer_eval',REPO/'scripts/eval_soccer.py')
    evaluator=importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    evaluator.EVAL_SET=FIXED
    assert not (Path(job['out'])/'summary.json').exists()
    sys.argv=['eval_soccer.py','--checkpoint',job['checkpoint'],'--episodes','100','--repeats','3','--seed','0','--device','cuda:0','--out',job['out']]
    with torch.inference_mode(): evaluator.main()
    torch.load=original_load
    print(json.dumps(verify(job)),flush=True)

def batch(p):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    lock=(OUT/'.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'state':'running','pid':os.getpid(),'started_at_utc':now(),'completed':0,'total':len(p['jobs']),'current':None,'training':False}
    results=[]
    for i,job in enumerate(p['jobs']):
        folder=Path(job['out']); folder.mkdir(parents=True,exist_ok=True)
        status.update(current={k:job[k] for k in ('method','training_seed','iteration_0based')},updated_at_utc=now())
        save(OUT/'status.json',status)
        if (folder/'verification.json').exists():
            previous=json.loads((folder/'verification.json').read_text())
            assert previous['checkpoint_sha256']==job['checkpoint_sha256']
            assert previous['summary_sha256']==sha(folder/'summary.json')
            assert previous['per_episode_sha256']==sha(folder/'per_episode.pt')
            result=verify(job)
        elif job['reuse']:
            assert not (folder/'summary.json').exists()
            for name in ('summary.json','per_episode.pt'): shutil.copyfile(Path(job['reuse'])/name,folder/name)
            result=verify(job)
        else:
            assert not (folder/'summary.json').exists(), 'Partial output must be inspected; never overwrite a score.'
            save(folder/'job.json',job)
            env=os.environ.copy(); env.update(LD_LIBRARY_PATH=DRIVER,CUDA_MODULE_LOADING='EAGER',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1')
            with (folder/'console.log').open('ab',buffering=0) as log:
                process=subprocess.Popen([str(REPO/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--child',str(folder/'job.json')],cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.monotonic(); fatal_at=None
                while process.poll() is None:
                    content=(folder/'console.log').read_bytes()[-40000:]
                    if b'Fatal Python error:' in content: fatal_at=fatal_at or time.monotonic()
                    if fatal_at is not None and time.monotonic()-fatal_at>15 or time.monotonic()-start>240:
                        os.killpg(process.pid,signal.SIGTERM)
                        try: process.wait(timeout=5)
                        except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL)
                    status.update(child_pid=process.pid,updated_at_utc=now()); save(OUT/'status.json',status)
                    time.sleep(3)
                code=process.wait()
            if code:
                status.update(state='failed',returncode=code,updated_at_utc=now(),failed_job=job)
                save(OUT/'status.json',status)
                return 1
            result=verify(job)
        results.append(result); status.update(completed=i+1,updated_at_utc=now())
        save(OUT/'status.json',status); save(OUT/'all_results.json',results)
        print(f"[{i+1}/{len(p['jobs'])}] seed{job['training_seed']} {job['method']} model_{job['iteration_0based']}: {result['goal_count']}/300",flush=True)
    status.update(state='finished',current=None,returncode=0,finished_at_utc=now());save(OUT/'status.json',status)
    return 0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--child')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.child: child(json.loads(Path(args.child).read_text()));return 0
    p=plan()
    if (OUT/'plan.json').exists():
        saved=json.loads((OUT/'plan.json').read_text());assert p['jobs']==saved['jobs'];p=saved
    else: save(OUT/'plan.json',p)
    if args.prepare: print(json.dumps({k:v for k,v in p.items() if k!='jobs'},indent=2))
    if args.run: return batch(p)
    return 0

if __name__=='__main__': sys.exit(main())
