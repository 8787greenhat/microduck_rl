"""One fresh B/C seed run; isolation, diagnostics and atomic snapshots only."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

ROOT=Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
REPO=Path('/home/lingchentseng/projects/microduck_rl_4060_multiseed_1a8a105')
OLD=Path('/home/lingchentseng/projects/microduck_rl_4060_7fa6298')
PIN='1a8a1053533210d6695454cf51026551f3e4dbbc'
DRIVER='/usr/lib/wsl/lib:/usr/lib/wsl/drivers/nvacegpu.inf_amd64_75303a74d1223c32'
OUT=ROOT/'outputs/multiseed_20261120_legacy_seed1'

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    pending.replace(path)

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('B','C'),required=True)
    parser.add_argument('--seed',type=int,choices=(2,3),required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('LD_LIBRARY_PATH')==DRIVER, 'Set driver path before Python starts'
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','MICRODUCK_WARM_START','MICRODUCK_SOCCER_WALK_ONNX','MICRODUCK_SOCCER_KICK_ONNX'):
        os.environ.pop(key,None)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_MODULE_LOADING='EAGER',TZ='Asia/Taipei',PYTHONUNBUFFERED='1',
                      WARP_CACHE_PATH=str(OLD/'.local-runtime/warp-cache-driver610'),CUDA_CACHE_PATH=str(OLD/'.local-runtime/cuda-cache-driver610'))
    os.chdir(REPO)
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','diff','--name-only','HEAD'],text=True).strip()
    assert hashlib.sha256((REPO/'uv.lock').read_bytes()).hexdigest()=='2eeeb680025baa737e7ccf3a87da3695a9d49b228de0bb60d8af0c022ab5f1aa'
    for name,digest in {'alpha_walking.onnx':'e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c',
                        'ball_kick_right.onnx':'147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af'}.items():
        assert hashlib.sha256((REPO/'policies'/name).read_bytes()).hexdigest()==digest
    import torch
    import mjlab.tasks
    import mjlab_microduck.tasks as tasks
    from mjlab.scripts.train import TrainConfig,launch_training
    assert Path(tasks.__file__).resolve().is_relative_to(REPO/'src')
    assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory>7_000_000_000
    import warp as wp
    import mujoco_warp
    wp.init()
    for name in ('smooth','constraint','collision_driver','sensor','forward','passive','support','solver','derivative','io'):
        wp.load_module(module='mujoco_warp._src.'+name,device='cuda:0')
    name=f"{'smoke' if args.smoke else 'formal'}_{args.method}_seed{args.seed}"
    folder=OUT/name
    folder.mkdir(exist_ok=True)
    assert not (folder/'launch.json').exists(), 'A run already exists; inspect instead of overwriting/restarting.'
    task='Mjlab-Soccer-Flat-MicroDuck' if args.method=='B' else 'Mjlab-Soccer-Curriculum-Flat-MicroDuck'
    cfg=TrainConfig.from_task(task)
    cfg.env.scene.num_envs=64 if args.smoke else 2048
    cfg.agent.seed=args.seed
    cfg.agent.max_iterations=5 if args.smoke else 2000
    cfg.agent.save_interval=100
    cfg.agent.logger='tensorboard'
    cfg.agent.run_name=f'{args.method}_seed{args.seed}_legacyrep_20261120'+('_smoke' if args.smoke else '')
    cfg.agent.resume=False
    assert cfg.agent.num_steps_per_env==24
    assert cfg.agent.algorithm.num_mini_batches==4 and cfg.agent.algorithm.num_learning_epochs==5
    assert cfg.agent.algorithm.learning_rate==3e-4
    expected_steps=cfg.agent.max_iterations
    original_save=tasks.SoccerOnPolicyRunner.save
    def atomic_save(runner,path,infos=None):
        for model in (runner.alg.actor,runner.alg.critic):
            assert all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite model; preserving prior snapshot'
        info={**(infos or {}),'multiseed_recovery':{'method':args.method,'training_seed':args.seed,
            'algorithm_learning_rate':float(runner.alg.learning_rate),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'next_iteration':int(runner.current_learning_iteration)+1}}
        target=Path(path)
        pending=target.with_name('.'+target.name+'.pending')
        original_save(runner,str(pending),info)
        pending.replace(target)
        write_json(folder/'latest_checkpoint.json',{'path':str(target.resolve()),'iteration_0based':int(runner.current_learning_iteration),
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'saved_at_utc':now(),'actual_run_dir':str(target.parent.resolve())})
    tasks.SoccerOnPolicyRunner.save=atomic_save
    original_log=None
    original_init=tasks.SoccerOnPolicyRunner.__init__
    def init_with_progress(runner,*positional,**kwargs):
        original_init(runner,*positional,**kwargs)
        original_log=runner.logger.log
        def log_progress(*values,**fields):
            original_log(*values,**fields)
            it=int(fields['it'])
            if it%5==0 or it==expected_steps-1:
                write_json(folder/'progress.json',{'method':args.method,'seed':args.seed,'state':'training',
                    'iteration_0based':it,'completed_iterations':it+1,'target_iterations':expected_steps,
                    'collection_time_s':float(fields['collect_time']),'PPO_update_time_s':float(fields['learn_time']),
                    'actual_run_dir':str(Path(runner.logger.log_dir).resolve()),'updated_at_utc':now()})
        runner.logger.log=log_progress
        write_json(folder/'run_directory.json',{'path':str(Path(runner.logger.log_dir).resolve())})
    tasks.SoccerOnPolicyRunner.__init__=init_with_progress
    manifest={'created_at_utc':now(),'commit':PIN,'method':args.method,'seed':args.seed,'task':task,
        'repo':str(REPO),'python':sys.version,'envs':cfg.env.scene.num_envs,'iterations':expected_steps,
        'rollout_steps_per_env':24,'minibatches':4,'epochs':5,'save_interval':100,'resume':False,
        'logger':'tensorboard','run_name':cfg.agent.run_name,'fresh_initialization':True,
        'original_seed1_compatible_fresh_training_behavior':True,'pending_PR4_startup_change_applied':False,
        'versions':{key:importlib.metadata.version(key) for key in ('torch','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib')},
        'driver_library_path':DRIVER,'atomic_checkpoint_wrapper':True,'smoke':args.smoke}
    write_json(folder/'launch.json',manifest)
    print('[MULTISEED] '+json.dumps(manifest,ensure_ascii=False),flush=True)
    launch_training(task,cfg)
    write_json(folder/'process_success.json',{'returncode':0,'ended_at_utc':now()})

if __name__=='__main__':
    main()
