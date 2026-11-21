"""Shared pinned-runtime initialization and provenance; no machine-specific paths."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from .protocol import MODEL_HASHES, sha

ROOT=Path(__file__).resolve().parents[3]

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def verify_skills():
    from mjlab_microduck.soccer_adapter import WALK_POLICY_ONNX, KICK_POLICY_ONNX
    paths=[Path(WALK_POLICY_ONNX),Path(KICK_POLICY_ONNX)]
    actual={name:sha(path) for name,path in zip(MODEL_HASHES,paths)}
    if actual!=MODEL_HASHES:
        raise ValueError(f'Frozen official skill hash mismatch: {actual}')
    return actual

def initialize(seed=0):
    # Set driver-specific LD_LIBRARY_PATH externally before launching Python.
    os.environ.setdefault('CUDA_MODULE_LOADING','EAGER')
    import torch
    torch.set_num_threads(1); torch.manual_seed(seed)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA PyTorch is required for duel rollouts')
    verify_skills()
    import mjlab.tasks
    import mjlab_microduck.tasks
    import mujoco_warp
    import warp as wp
    wp.init()
    for name in ('smooth','constraint','collision_driver','sensor','forward','passive','support','solver','derivative','io'):
        wp.load_module(module='mujoco_warp._src.'+name,device='cuda:0')

def provenance():
    import torch
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True).strip()
    paths=list((ROOT/'src/mjlab_microduck/duel').glob('*.py'))
    paths += [ROOT/'src/mjlab_microduck/tasks'/name for name in ('mdp.py','microduck_duel_env_cfg.py','__init__.py')]
    paths += list((ROOT/'scripts').glob('*duel.py'))
    return {'source_commit':commit,'source_dirty':bool(dirty),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'uv_lock_sha256':sha(ROOT/'uv.lock'),'python':sys.version,'device':torch.cuda.get_device_name(0),
        'versions':{name:importlib.metadata.version(name) for name in ('torch','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib')},
        'official_skill_sha256':verify_skills()}

