"""Verify the archived matched-helper experiment. Standard library only."""
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
    raw=ROOT/'raw';original=read(raw/'file_manifest.json')['files']
    assert len(original)==56
    for name,v in original.items():
        p=raw/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
    spec=importlib.util.spec_from_file_location('duel_protocol',REPO/'src/mjlab_microduck/duel/protocol.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=read(raw/'comparison_summary.json')
    with (raw/'all_512_episodes.csv').open(encoding='utf-8',newline='') as f:combined=list(csv.DictReader(f))
    assert len(combined)==512
    reference=None
    for condition,goals,against in [('P_H_vs_A',44,3),('P_H_vs_R_H',23,1),('R_H_vs_A',38,6),('R_H_vs_R_H',11,13)]:
        rows=[]
        for side in (0,1):
            folder=raw/f'{condition}_side{side}';summary=read(folder/'summary.json');items=read(folder/'episodes.json');manifest=read(folder/'manifest.json')
            assert summary['complete'] and summary['actor_and_normalizer_unchanged'] and not summary['new_training']
            assert summary['actor_used']==manifest['actor_used']==condition.startswith('P_')
            assert sha(folder/'per_episode.csv')==summary['per_episode_csv_sha256']
            assert module.summarize(items)==summary['total'];rows+=items
            assert manifest['shared_v3_helper_config']==report['shared_helper_config']
            assert manifest['wrapper_sha256']==sha(raw/'eval_duel_matched_helper.py')
            assert manifest['opponent_wrapper_sha256']==sha(raw/'duel_matched_helper.py')
            if manifest['opponent_helper_active']:assert summary['helper_opponent_guard_violations']==0
            paired={k:manifest[k] for k in ('source','checkpoint_sha256','scene_ids','repeats','num_envs','eval_rng_seed','round_seconds','original_protocol','shared_v3_helper_config','rule_request')}
            if reference is None:reference=paired
            else:assert paired==reference
        assert len(rows)==128 and {(r['scene_id'],r['repeat'],r['learner_side']) for r in rows}==set(itertools.product(range(32),range(2),range(2)))
        assert module.summarize(rows)==report['results'][condition]['total']
        assert sum(r['end_reason']=='goal_for' for r in rows)==goals
        assert sum(r['end_reason']=='goal_against' for r in rows)==against
        expected={(r['scene_id'],r['repeat'],r['learner_side']):r for r in rows}
        csv_rows=[r for r in combined if r['opponent_variant']==condition]
        assert len(csv_rows)==128
        for row in csv_rows:
            prior=expected[int(row['scene_id']),int(row['repeat']),int(row['learner_side'])]
            assert all(str(prior[k])==v for k,v in row.items())
    workers=read(raw/'status.json')['workers']
    assert len(workers)==8 and all(w['returncode']==0 for w in workers)
    checkpoint=ROOT.parent/'v3_seed1_20261124/checkpoints/model_1999.pt'
    assert sha(checkpoint)==report['checkpoint_sha256']
    print(json.dumps(dict(verified=True,files=len(files)+1,raw_files=57,episodes=512,all_matched_opponent_guard_violations_zero=True,checkpoint_sha256=sha(checkpoint),new_training=False)))
if __name__=='__main__':main()
