"""Original fixed 100x3 final evaluation for a completed new seed; no training."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
REPO=Path('/home/lingchentseng/projects/microduck_rl_4060_multiseed_1a8a105')
OLD=Path('/home/lingchentseng/projects/microduck_rl_4060_7fa6298')
OUT=ROOT/'outputs/multiseed_20261120_legacy_seed1'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=('B','C'),required=True)
    parser.add_argument('--seed',type=int,choices=(2,3),required=True)
    args=parser.parse_args()
    folder=OUT/f'formal_{args.method}_seed{args.seed}'
    verified=json.loads((folder/'verification.json').read_text())
    assert verified['verified'] and verified['iterations']==2000
    source=Path(verified['checkpoint'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==verified['checkpoint_sha256']
    target=folder/'evaluation'
    target.mkdir(exist_ok=True)
    assert not (target/'summary.json').exists(), 'Preserve the first completed evaluation.'
    fixed=ROOT/'outputs/seed1_evaluation/soccer_eval_set.json'
    assert hashlib.sha256(fixed.read_bytes()).hexdigest()=='3a59dee2c22b5129fcb10685192630443a29550790ee5ccbffc04e414d053464'
    assert hashlib.sha256((REPO/'scripts/eval_soccer.py').read_bytes()).hexdigest()=='2d8bd0f9fc840aa54fa858181e16ccb88813b4fc4579db42c6b49c7a85ba0757'
    assert os.environ.get('LD_LIBRARY_PATH')=='/usr/lib/wsl/lib:/usr/lib/wsl/drivers/nvacegpu.inf_amd64_75303a74d1223c32'
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','MICRODUCK_WARM_START','CUDA_VISIBLE_DEVICES'):
        os.environ.pop(key,None)
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
    def cpu_load(path,*values,**fields):
        if isinstance(path,(str,Path)) and Path(path).resolve()==source.resolve():
            fields.update(map_location='cpu',weights_only=True)
            state=original_load(path,*values,**fields)
            expected.update(state['actor_state_dict'])
            return state
        return original_load(path,*values,**fields)
    torch.load=cpu_load
    original_policy=tasks.SoccerOnPolicyRunner.get_inference_policy
    def checked_policy(runner,*values,**fields):
        actual=runner.alg.actor.state_dict()
        assert set(actual)==set(expected) and all(torch.equal(actual[k].detach().cpu(),v) for k,v in expected.items())
        return original_policy(runner,*values,**fields)
    tasks.SoccerOnPolicyRunner.get_inference_policy=checked_policy
    spec=importlib.util.spec_from_file_location('unchanged_official_eval',REPO/'scripts/eval_soccer.py')
    evaluator=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.EVAL_SET=fixed
    sys.argv=['eval_soccer.py','--checkpoint',str(source),'--episodes','100','--repeats','3','--seed','0','--device','cuda:0','--out',str(target)]
    with torch.inference_mode(): evaluator.main()
    torch.load=original_load
    data=torch.load(target/'per_episode.pt',map_location='cpu',weights_only=True)
    flags=('goal','fell','robot_oob','ball_oob','timeout','kicked','contact')
    rows=[]
    for i in range(300):
        row={'layout_index_0based':i%100,'repeat_index_1based':i//100+1,**{k:int(data[k][i]) for k in flags},'elapsed_s':round(float(data['steps'][i])*.1,3)}
        assert any(row[k] for k in flags[:5])
        rows.append(row)
    summary=json.loads((target/'summary.json').read_text())
    assert summary['envs']==300
    count=sum(r['goal'] for r in rows)
    assert abs(summary['rates']['goal']-count/300)<1e-6
    with (target/'per_episode.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'verified':True,'method':args.method,'training_seed':args.seed,'evaluation_rng_seed':0,'goal_count':count,
        'episodes':300,'layouts':100,'repeats':3,'goal_rate':count/300,'mean_goal_time_s':summary['mean_goal_time_s'],
        'checkpoint_sha256':verified['checkpoint_sha256'],'fixed_eval_set_sha256':hashlib.sha256(fixed.read_bytes()).hexdigest(),
        'source_commit':'1a8a1053533210d6695454cf51026551f3e4dbbc','evaluator_identical_to_seed1':True,
        'full_task_without_reset_curriculum':True,'actor_exact_match_to_source':True}
    (target/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
