"""Verify preserved bytes and recompute headline results. Standard library only; no GPU."""
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))

def metrics(folder):
    manifest = read(folder/'manifest.json')
    summary = read(folder/'summary.json')
    assert summary['complete'] and summary['official']
    assert sha(folder/'per_episode.csv') == summary['per_episode_csv_sha256']
    with (folder/'per_episode.csv').open(encoding='utf-8',newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 600
    assert {(int(r['scene_id']),int(r['repeat']),int(r['learner_side'])) for r in rows} == set(itertools.product(range(100),range(3),range(2)))
    counts = collections.Counter(r['end_reason'] for r in rows)
    assert counts == collections.Counter(summary['total']['end_counts'])
    assert sum(counts.values()) == 600
    wins = [r for r in rows if r['end_reason']=='goal_for']
    times = sorted(float(r['elapsed_s']) for r in wins)
    x = (len(times)-1)*0.9
    p90 = times[int(x)] + (times[min(int(x)+1,len(times)-1)]-times[int(x)])*(x-int(x))
    for key,value in dict(mean=statistics.mean(times),median=statistics.median(times),p90=p90).items():
        assert abs(value-summary['total']['winning_time_s'][key]) < 1e-10
    assert counts['goal_for']/600 == summary['total']['goal_for_rate']
    for side in range(2):
        side_counts = collections.Counter(r['end_reason'] for r in rows if int(r['learner_side'])==side)
        assert side_counts == collections.Counter(summary['by_side'][str(side)]['end_counts'])
    result = dict(summary=summary['total'],source_commit=manifest['source_commit'],
                  checkpoint_sha256=manifest['checkpoint_sha256'],
                  rounds_with_learner_kick_skill=sum(int(r['learner_kick_attempts'])>0 for r in rows),
                  total_learner_kick_skill_starts=sum(int(r['learner_kick_attempts']) for r in rows),
                  goal_rounds_with_learner_kick_skill=sum(int(r['learner_kick_attempts'])>0 for r in wins),
                  goal_rounds_with_any_learner_right_foot_contact=sum(int(r[f"physics_samples_ball_foot_{r['learner_side']}"])>0 for r in wins))
    return manifest,result

def verify():
    inventory = read(ROOT/'file_manifest.json')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='file_manifest.json' and '__pycache__' not in p.parts}
    assert actual == set(inventory['files'])
    for name,entry in inventory['files'].items():
        path = ROOT/name
        assert path.stat().st_size == entry['bytes'] and sha(path)==entry['sha256'], name
    a,v2 = metrics(ROOT/'comparison/v2_fixed600')
    b,v3 = metrics(ROOT/'evaluation/fixed600')
    assert a['protocol']==b['protocol']
    assert a['official_skill_sha256']==b['official_skill_sha256']
    assert sha(ROOT/'evaluation_set/layouts.json') == b['protocol']['layouts_sha256']
    assert read(ROOT/'comparison/comparison.json')['v2']==v2
    assert read(ROOT/'comparison/comparison.json')['v3']==v3
    checkpoints = read(ROOT/'checkpoint_index.json')
    assert len(checkpoints)==21
    for c in checkpoints:
        assert sha(ROOT/c['path'])==c['sha256']
    assert checkpoints[-1]['sha256']==b['checkpoint_sha256']
    print(json.dumps(dict(verified=True,files=len(actual),checkpoints=len(checkpoints),
                         v2_goals=v2['summary']['end_counts']['goal_for'],v3_goals=v3['summary']['end_counts']['goal_for'],
                         v3_kick_rounds=v3['rounds_with_learner_kick_skill']),indent=2))

if __name__=='__main__':
    verify()
