"""Provenance for the independent v3 controller/training entry points."""
from .runtime import ROOT
from .protocol import sha
from .v2_runtime import v2_provenance

def v3_provenance():
    value=v2_provenance()
    for name in ('src/mjlab_microduck/duel/v3_env.py','src/mjlab_microduck/duel/v3_runner.py',
                 'src/mjlab_microduck/duel/v3_runtime.py','src/mjlab_microduck/duel/stop_settle.py',
                 'src/mjlab_microduck/duel/stop_settle_env.py','scripts/train_duel_v3.py','scripts/eval_duel_v3.py'):
        value['source_sha256'][name]=sha(ROOT/name)
    return value
