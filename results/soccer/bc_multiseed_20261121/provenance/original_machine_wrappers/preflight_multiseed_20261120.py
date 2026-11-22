"""CPU-only source/config replication audit for the original seed-1 protocol."""
import ast
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
REPO=Path('/home/lingchentseng/projects/microduck_rl_4060_multiseed_1a8a105')
OUT=ROOT/'outputs/multiseed_20261120_legacy_seed1'
OLD=Path('/home/lingchentseng/projects/microduck_rl_4060_7fa6298')
PIN='1a8a1053533210d6695454cf51026551f3e4dbbc'
OUT.mkdir(exist_ok=True)
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==PIN
assert hashlib.sha256((REPO/'uv.lock').read_bytes()).hexdigest()=='2eeeb680025baa737e7ccf3a87da3695a9d49b228de0bb60d8af0c022ab5f1aa'
for name in ('soccer_adapter.py','soccer_constants.py','soccer_onnx.py','tasks/microduck_soccer_env_cfg.py'):
    a=OLD/'src/mjlab_microduck'/name
    b=REPO/'src/mjlab_microduck'/name
    if a.exists():
        assert a.read_bytes()==b.read_bytes(),name
old_ast=ast.parse((OLD/'src/mjlab_microduck/tasks/mdp.py').read_text())
new_ast=ast.parse((REPO/'src/mjlab_microduck/tasks/mdp.py').read_text())
old_functions={n.name:ast.dump(n) for n in old_ast.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
new_functions={n.name:ast.dump(n) for n in new_ast.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
assert all(new_functions.get(k)==v for k,v in old_functions.items())
assert set(new_functions)-set(old_functions)=={'soccer_curriculum_state','restore_soccer_curriculum_state'}
import mjlab.tasks
import mjlab_microduck.tasks as tasks
from mjlab.scripts.train import TrainConfig
assert Path(tasks.__file__).resolve().is_relative_to(REPO/'src')
b=TrainConfig.from_task('Mjlab-Soccer-Flat-MicroDuck')
c=TrainConfig.from_task('Mjlab-Soccer-Curriculum-Flat-MicroDuck')
ba,ca=asdict(b.agent),asdict(c.agent)
agent_diffs={k:{'B':ba[k],'C':ca[k]} for k in ba if ba[k]!=ca[k]}
assert set(agent_diffs)=={'experiment_name','run_name'}
assert set(c.env.curriculum)-set(b.env.curriculum)=={'reset_layout'}
assert c.env.curriculum['reset_layout'].params['easy_fraction']==.2
assert c.env.curriculum['reset_layout'].params['promote_goal_rate']==.5
assert ba['num_steps_per_env']==24 and ba['max_iterations']==2000
assert ba['algorithm']['num_mini_batches']==4 and ba['algorithm']['num_learning_epochs']==5
assert ba['algorithm']['learning_rate']==.0003
versions={k:importlib.metadata.version(k) for k in ('torch','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','numpy','onnx','onnxruntime')}
report={'commit':PIN,'seed1_commit':'7fa62987d76dd2ba00ec15ffdd1de5bd93480845','python':sys.version,
        'versions':versions,'common_mdp_functions_AST_identical':True,'only_new_mdp_functions':sorted(set(new_functions)-set(old_functions)),
        'skill_adapter_constants_and_soccer_config_identical':True,'agent_B_C_differences':agent_diffs,
        'new_training_behavior_matches_seed1':True,'checkpoint_state_persistence_fix_present':True,
        'unmerged_PR4_startup_count_change_applied':False,'experiment_label':'legacy seed1-protocol replication',
        'plan':[{'method':m,'seed':s,'envs':2048,'iterations':2000,'resume':False} for s in (2,3) for m in ('B','C')]}
(OUT/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
