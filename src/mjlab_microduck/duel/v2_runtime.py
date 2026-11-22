"""Extend existing provenance to cover every v2 entry point and configuration."""
from .runtime import ROOT,provenance
from .protocol import sha

def v2_provenance():
    value=provenance()
    for name in ('src/mjlab_microduck/tasks/microduck_duel_v2_env_cfg.py',
                 'scripts/train_duel_v2.py','scripts/eval_duel_v2.py'):
        value['source_sha256'][name]=sha(ROOT/name)
    return value
