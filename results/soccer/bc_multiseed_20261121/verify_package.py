"""Verify published data using only the Python standard library; no GPU work."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))

def main():
    manifest = read(ROOT/'file_manifest.json')
    for item in manifest['files']:
        path = ROOT/item['path']
        assert path.resolve().is_relative_to(ROOT), item['path']
        assert path.stat().st_size == item['bytes'], item['path']
        assert sha(path) == item['sha256'], item['path']
    protocol = read(ROOT/'protocol.json')
    assert sha(ROOT/'evaluation_set/soccer_eval_set.json') == protocol['eval_set_sha256']
    checkpoints = read(ROOT/'checkpoint_index.json')['checkpoints']
    assert len(checkpoints) == 84
    for method in ('B','C'):
        for seed in (2,3):
            selected = [r for r in checkpoints if r['method']==method and r['training_seed']==seed]
            assert [r['iteration_0based'] for r in selected] == list(range(0,2000,100))+[1999]
            for item in selected:
                assert sha(ROOT/item['path']) == item['sha256']
                assert item['completed_iterations'] == item['iteration_0based']+1
    evaluations = protocol['evaluation_records']
    assert len(evaluations) == 48
    assert len({(j['method'],j['training_seed'],j['iteration_0based']) for j in evaluations}) == 48
    finals = {(r['method'],int(r['training_seed'])):r for r in rows(ROOT/'analysis/final_results.csv')}
    curve = {(r['method'],int(r['training_seed']),int(r['iteration_0based'])):r
             for r in rows(ROOT/'analysis/learning_curve.csv')}
    total = 0
    for job in evaluations:
        folder = ROOT/job['path']
        summary = read(folder/'summary.json')
        verification = read(folder/'verification.json')
        assert sha(folder/'summary.json') == verification['summary_sha256']
        assert sha(folder/'per_episode.pt') == verification['per_episode_sha256']
        assert verification['checkpoint_sha256'] == job['checkpoint_sha256']
        episode = rows(folder/'per_episode.csv')
        assert len(episode) == 300 and summary['envs'] == 300
        for i,row in enumerate(episode):
            assert int(row['layout_index_0based']) == i%100
            assert int(row['repeat_index_1based']) == i//100+1
            assert any(int(row[k]) for k in ('goal','fell','robot_oob','ball_oob','timeout'))
        count = sum(int(r['goal']) for r in episode)
        assert count == job['goal_count'] == verification['goal_count']
        assert math.isclose(summary['rates']['goal'], count/300, abs_tol=1e-6)
        key = (job['method'],job['training_seed'],job['iteration_0based'])
        assert math.isclose(float(curve[key]['goal_rate']), count/300, abs_tol=1e-9)
        if job['training_seed'] in (2,3):
            item = next(r for r in checkpoints if (r['method'],r['training_seed'],r['iteration_0based'])==key)
            assert item['sha256'] == job['checkpoint_sha256']
        if job['iteration_0based'] == 1999:
            final = finals[key[:2]]
            assert int(final['goal_count']) == count
            successful = [float(r['elapsed_s']) for r in episode if int(r['goal'])]
            assert math.isclose(float(final['mean_goal_time_s']),sum(successful)/len(successful),abs_tol=1e-5)
        total += len(episode)
    assert total == 14400
    print(json.dumps({'verified':True,'files':len(manifest['files']),'checkpoints':84,
                      'evaluations':48,'episodes':total,'GPU_used':False}, indent=2))

if __name__ == '__main__':
    main()
