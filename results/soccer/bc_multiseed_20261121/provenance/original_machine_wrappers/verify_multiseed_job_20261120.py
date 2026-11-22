"""CPU-only verification after a smoke or formal job; never loads an env."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
OUT=ROOT/'outputs/multiseed_20261120_legacy_seed1'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('B','C'),required=True)
    parser.add_argument('--seed',type=int,choices=(2,3),required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    folder=OUT/f"{'smoke' if args.smoke else 'formal'}_{args.method}_seed{args.seed}"
    launch=json.loads((folder/'launch.json').read_text())
    run=Path(json.loads((folder/'run_directory.json').read_text())['path'])
    assert json.loads((folder/'process_success.json').read_text())['returncode']==0
    expected=5 if args.smoke else 2000
    assert launch['iterations']==expected and launch['envs']==(64 if args.smoke else 2048)
    assert launch['resume'] is False and launch['fresh_initialization'] is True
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    checkpoint=run/f'model_{expected-1}.pt'
    final=torch.load(checkpoint,map_location='cpu',weights_only=True)
    first=torch.load(run/'model_0.pt',map_location='cpu',weights_only=True)
    assert final['iter']==expected-1 and final['infos']['env_state']['common_step_counter']==expected*120
    assert first['iter']==0
    changed={}
    for key in ('actor_state_dict','critic_state_dict'):
        assert all(torch.isfinite(v).all().item() for v in final[key].values() if isinstance(v,torch.Tensor))
        names=[n for n in final[key] if n.startswith('mlp.')]
        changed[key]=sum(not torch.equal(final[key][n],first[key][n]) for n in names)
        assert changed[key]>0, 'No learned MLP parameter change after model_0'
    steps={int(v['step']) for v in final['optimizer_state_dict']['state'].values() if 'step' in v}
    assert steps=={20*expected}
    c_state=final['infos'].get('soccer_curriculum')
    assert (c_state is not None)==(args.method=='C')
    if c_state is not None:
        assert 0<=c_state['stage']<=2 and all(k in c_state for k in ('stage_start_step','goals','episodes'))
    accumulator=EventAccumulator(str(run),size_guidance={'scalars':0})
    accumulator.Reload()
    tags=accumulator.Tags()['scalars']
    assert tags
    all_finite=all(math.isfinite(event.value) for tag in tags for event in accumulator.Scalars(tag))
    assert all_finite
    curve=accumulator.Scalars('Train/mean_reward')
    assert [e.step for e in curve]==list(range(expected))
    if 'Episode_Termination/nan_state' in tags:
        assert max(e.value for e in accumulator.Scalars('Episode_Termination/nan_state'))==0
    for name in ('fell','out_of_bounds','kick_attempt','time'):
        tag='Episode_Reward/'+name
        assert tag in tags and max(e.value for e in accumulator.Scalars(tag))<=1e-8
    result={'verified':True,'method':args.method,'seed':args.seed,'smoke':args.smoke,'iterations':expected,
        'actual_run_dir':str(run),'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'MLP_tensors_changed_vs_model0':changed,'optimizer_steps':list(steps),'common_step_counter':expected*120,
        'all_model_and_logged_scalars_finite':True,'nan_state_max':0,'curriculum_state_saved':c_state,
        'fresh_seed_not_warm_start':True,'formal_evaluation_completed':False}
    (folder/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
