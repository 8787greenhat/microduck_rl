"""Verify archived bytes and recompute 384-episode metrics. No GPU needed."""
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files=read(ROOT/'file_manifest.json')['files']
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p!=ROOT/'file_manifest.json' and '__pycache__' not in p.parts}
    assert actual==set(files)
    for name,v in files.items():
        p=ROOT/name;assert p.resolve().is_relative_to(ROOT)
        assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
    raw=ROOT/'raw'
    for name,v in read(raw/'file_manifest.json')['files'].items():
        p=raw/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
    spec=importlib.util.spec_from_file_location('duel_protocol',REPO/'src/mjlab_microduck/duel/protocol.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=read(raw/'comparison_summary.json')
    with (raw/'all_384_episodes.csv').open(encoding='utf-8',newline='') as f:combined=list(csv.DictReader(f))
    assert len(combined)==384
    for variant,goals in [('original_A',36),('short_dwell_A',37),('direct_chase_A',22)]:
        rows=[]
        for side in (0,1):
            folder=raw/f'{variant}_side{side}';summary=read(folder/'summary.json');items=read(folder/'episodes.json')
            assert summary['complete'] and summary['actor_and_normalizer_unchanged'] and not summary['new_training']
            assert sha(folder/'per_episode.csv')==summary['per_episode_csv_sha256']
            assert module.summarize(items)==summary['total'];rows+=items
        assert len(rows)==128 and {(r['scene_id'],r['repeat'],r['learner_side']) for r in rows}==set(itertools.product(range(32),range(2),range(2)))
        assert module.summarize(rows)==report['results'][variant]['total']
        assert sum(r['end_reason']=='goal_for' for r in rows)==goals
        expected={(r['scene_id'],r['repeat'],r['learner_side']):r for r in rows}
        from_csv=[r for r in combined if r['opponent_variant']==variant]
        assert len(from_csv)==128
        for row in from_csv:
            prior=expected[int(row['scene_id']),int(row['repeat']),int(row['learner_side'])]
            assert all(str(prior[key])==value for key,value in row.items())
    assert len(read(raw/'status.json')['workers'])==6 and all(w['returncode']==0 for w in read(raw/'status.json')['workers'])
    checkpoint=ROOT.parent/'v3_seed1_20261124/checkpoints/model_1999.pt'
    assert sha(checkpoint)==report['checkpoint_sha256']
    print(json.dumps(dict(verified=True,files=len(files)+1,raw_files=45,episodes=384,checkpoint_sha256=sha(checkpoint),new_training=False)))
if __name__=='__main__':main()
