"""Matched checkpoint evaluation. Reuses the pinned official evaluator, never trains."""
import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
REPO = Path('/home/lingchentseng/projects/microduck_rl_4060_7fa6298')
OUT = ROOT / 'outputs/bc_checkpoint_eval_20261025'
FIXED = ROOT / 'outputs/seed1_evaluation/soccer_eval_set.json'
COMMIT = '7fa62987d76dd2ba00ec15ffdd1de5bd93480845'
FIXED_HASH = '3a59dee2c22b5129fcb10685192630443a29550790ee5ccbffc04e414d053464'
SCRIPT_HASH = '2d8bd0f9fc840aa54fa858181e16ccb88813b4fc4579db42c6b49c7a85ba0757'
PYTHON = REPO / '.venv/bin/python'
DRIVER = '/usr/lib/wsl/lib:/usr/lib/wsl/drivers/nvacegpu.inf_amd64_75303a74d1223c32'
FLAGS = ('goal', 'fell', 'robot_oob', 'ball_oob', 'timeout', 'kicked', 'contact')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def check_protocol():
    assert sha(FIXED) == FIXED_HASH
    assert len(json.loads(FIXED.read_text())['episodes']) == 100
    assert sha(REPO / 'scripts/eval_soccer.py') == SCRIPT_HASH
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(REPO), 'diff', '--name-only', 'HEAD'], text=True).strip()
    assert json.loads((ROOT / 'outputs/formal_C_seed1_20261024_164942/status.json').read_text())['state'] == 'finished'
    versions = json.loads((ROOT / 'outputs/seed1_evaluation/final_audit.json').read_text())
    assert sha(REPO / 'uv.lock') == versions['uv_lock_sha256']
    # These are the adapter's actual default skill paths, unchanged since final evaluation.
    for name, expected in versions['frozen_model_sha256'].items():
        assert sha(REPO / 'policies' / name) == expected, name


def make_plan():
    check_protocol()
    b_audit = json.loads((ROOT / 'outputs/pr_review_20261025/data_audit.json').read_text())
    assert b_audit['manifest_files_verified'] == b_audit['manifest_files_checked'] == 50
    c_audit = json.loads((ROOT / 'outputs/checkpoint_learning_speed/checkpoint_inventory.json').read_text())
    b_models = {row['iteration_0based']: row for row in b_audit['checkpoint_records']}
    c_models = {row['iteration_0based']: row for row in c_audit['checkpoint_records'] if row['method'] == 'C'}
    points = list(range(0, 2000, 100)) + [1999]
    jobs = []
    for it in points:
        for method, models in (('B', b_models), ('C', c_models)):
            row = models[it]
            assert sha(row['path']) == row['sha256']
            reuse = None
            if it == 1999:
                reuse = str(ROOT / ('outputs/bc_comparison_20261025/B_seed1_on_4060' if method == 'B' else 'outputs/seed1_evaluation/metrics'))
            jobs.append({'method': method, 'iteration_0based': it,
                         'completed_iterations': it + 1, 'training_high_level_transitions': (it+1)*49152,
                         'checkpoint': row['path'], 'checkpoint_sha256': row['sha256'],
                         'out': str(OUT / method / f'model_{it}'), 'reuse_existing_final_output': reuse})
    return {'created_at_utc': now(), 'training_commit': COMMIT,
            'reviewed_PR1_head': '80c4725f0ba918741252b2e675aa75bc2ad6da45',
            'B_checkpoints_source_commit': b_audit['reviewed_commit'],
            'B_results_unchanged_in_latest_commit': True,
            'evaluation_script_sha256': SCRIPT_HASH, 'fixed_eval_set_sha256': FIXED_HASH,
            'evaluation_rng_seed': 0, 'training_seed': 1, 'layouts': 100, 'repeats': 3,
            'episodes_per_checkpoint': 300, 'checkpoint_points_per_method': 21,
            'new_GPU_evaluations': 40, 'reused_final_evaluations': 2,
            'new_training': False, 'modified_evaluation_settings': False,
            'thresholds_declared_before_evaluation': [0.7, 0.9],
            'C_recovery_note': 'Retain original 0-200 then resumed 201-1999; exclude discarded original 201-267 and recovery smoke. Physics episodes/RNG and legacy curriculum counters reinitialized on resume.',
            'jobs': jobs}


def verify_result(job, folder, reused=False):
    import torch
    folder = Path(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    data = torch.load(folder / 'per_episode.pt', weights_only=True, map_location='cpu')
    assert summary['envs'] == 300 and all(data[key].numel() == 300 for key in (*FLAGS, 'steps'))
    rows = []
    for i in range(300):
        row = {'layout_index_0based': i % 100, 'repeat_index_1based': i // 100 + 1,
               **{key: int(data[key][i]) for key in FLAGS},
               'elapsed_s': round(float(data['steps'][i]) * 0.1, 3)}
        assert any(row[key] for key in FLAGS[:5]), ('unfinished or unrecorded termination', i)
        assert 0 < row['elapsed_s'] <= 12.2
        rows.append(row)
    for key in FLAGS:
        assert math.isclose(summary['rates'][key], sum(row[key] for row in rows)/300, abs_tol=1e-6)
    goal_times = torch.tensor([row['elapsed_s'] for row in rows if row['goal']], dtype=torch.float64)
    mean = float(goal_times.mean()) if len(goal_times) else None
    if mean is None:
        assert summary['mean_goal_time_s'] is None
    else:
        assert math.isclose(mean, summary['mean_goal_time_s'], abs_tol=1e-5)
    with (folder / 'per_episode.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {key: sum(row[key] for row in rows) for key in FLAGS}
    result = {'method': job['method'], 'iteration_0based': job['iteration_0based'],
              'completed_iterations': job['completed_iterations'],
              'training_high_level_transitions': job['training_high_level_transitions'],
              'checkpoint_sha256': job['checkpoint_sha256'], 'checkpoint': job['checkpoint'],
              'evaluation_script_sha256': SCRIPT_HASH, 'fixed_eval_set_sha256': FIXED_HASH,
              'training_commit': COMMIT, 'evaluation_rng_seed': 0, 'training_seed': 1,
              'envs': 300, 'goal_count': counts['goal'], 'goal_rate': counts['goal']/300,
              'terminal_and_action_counts': counts, 'all_episodes_terminated': True,
              'mean_goal_time_s': mean,
              'median_goal_time_s': float(torch.quantile(goal_times, .5)) if len(goal_times) else None,
              'p90_goal_time_s': float(torch.quantile(goal_times, .9)) if len(goal_times) else None,
              'repeat_goal_counts': [sum(row['goal'] for row in rows if row['repeat_index_1based'] == r) for r in (1,2,3)],
              'exclusive_failure_combinations': dict(collections.Counter(
                  '+'.join(key for key in FLAGS[1:5] if row[key]) for row in rows if not row['goal'])),
              'reused_original_final_evaluation': reused, 'new_training': False,
              'summary_sha256': sha(folder / 'summary.json'), 'per_episode_pt_sha256': sha(folder / 'per_episode.pt'),
              'verified_at_utc': now()}
    assert sha(job['checkpoint']) == job['checkpoint_sha256']
    write_json(folder / 'verification.json', result)
    return result


def child(job):
    print(f"[checkpoint-eval] starting {job['method']} model_{job['iteration_0based']}", flush=True)
    check_protocol()
    assert os.environ.get('LD_LIBRARY_PATH') == DRIVER
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'MICRODUCK_WARM_START',
                 'MICRODUCK_SOCCER_WALK_ONNX', 'MICRODUCK_SOCCER_KICK_ONNX', 'CUDA_VISIBLE_DEVICES'):
        os.environ.pop(name, None)
    os.environ['CUDA_MODULE_LOADING'] = 'EAGER'
    os.environ['WARP_CACHE_PATH'] = str(REPO / '.local-runtime/warp-cache-driver610')
    os.environ['CUDA_CACHE_PATH'] = str(REPO / '.local-runtime/cuda-cache-driver610')
    os.chdir(REPO)
    print('[checkpoint-eval] protocol checked; importing runtime', flush=True)
    import torch
    import mjlab.tasks
    import mjlab_microduck.tasks
    import mujoco_warp
    import warp as wp
    assert torch.cuda.is_available()
    wp.init()
    for name in ('smooth', 'constraint', 'collision_driver', 'sensor', 'forward', 'passive',
                 'support', 'solver', 'derivative', 'io'):
        wp.load_module(module='mujoco_warp._src.'+name, device='cuda:0')
    spec = importlib.util.spec_from_file_location('official_fixed_soccer_eval', REPO / 'scripts/eval_soccer.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.EVAL_SET = FIXED
    original_load = torch.load
    expected_actor = {}
    def safe_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() == Path(job['checkpoint']).resolve():
            kwargs['weights_only'] = True
            # Load serialized tensors on CPU first; runner copies the same actor state
            # to CUDA. Avoid temporary GPU optimizer allocations during deserialization.
            kwargs['map_location'] = 'cpu'
            result = original_load(path, *args, **kwargs)
            expected_actor.update({key: value.clone() for key,value in result['actor_state_dict'].items()})
            return result
        return original_load(path, *args, **kwargs)
    torch.load = safe_load
    runner_class = mjlab_microduck.tasks.SoccerOnPolicyRunner
    original_get_policy = runner_class.get_inference_policy
    def checked_get_policy(runner, *args, **kwargs):
        assert expected_actor
        actual = runner.alg.actor.state_dict()
        assert set(actual) == set(expected_actor)
        assert all(torch.equal(actual[key].detach().cpu(), value) for key,value in expected_actor.items())
        print('[checkpoint-eval] CPU-deserialized actor equals GPU actor including normalization', flush=True)
        return original_get_policy(runner, *args, **kwargs)
    runner_class.get_inference_policy = checked_get_policy
    assert not (Path(job['out']) / 'summary.json').exists()
    sys.argv = ['eval_soccer.py', '--checkpoint', job['checkpoint'], '--episodes', '100', '--repeats', '3',
                '--seed', '0', '--device', 'cuda:0', '--out', job['out']]
    with torch.inference_mode():
        evaluator.main()
    torch.load = original_load
    runner_class.get_inference_policy = original_get_policy
    result = verify_result(job, job['out'])
    print(json.dumps({'verified': True, 'method': job['method'], 'iteration': job['iteration_0based'],
                      'goals': result['goal_count'], 'mean_goal_time_s': result['mean_goal_time_s']}), flush=True)


def batch(plan):
    import fcntl
    lock = (OUT / '.batch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'state': 'running', 'pid': os.getpid(), 'started_at_utc': now(), 'total': len(plan['jobs']),
             'completed': 0, 'current': None, 'training': False, 'results': []}
    began = time.monotonic()
    for index, job in enumerate(plan['jobs']):
        folder = Path(job['out'])
        folder.mkdir(parents=True, exist_ok=True)
        state['current'] = {'method': job['method'], 'iteration': job['iteration_0based']}
        state['updated_at_utc'] = now()
        write_json(OUT / 'status.json', state)
        if (folder / 'verification.json').exists():
            saved = json.loads((folder / 'verification.json').read_text())
            assert saved['checkpoint_sha256'] == job['checkpoint_sha256']
            assert saved['fixed_eval_set_sha256'] == FIXED_HASH
            result = verify_result(job, folder, saved['reused_original_final_evaluation'])
        elif job['reuse_existing_final_output']:
            source = Path(job['reuse_existing_final_output'])
            assert not (folder / 'summary.json').exists()
            for filename in ('summary.json', 'per_episode.pt'):
                shutil.copyfile(source / filename, folder / filename)
            result = verify_result(job, folder, True)
        else:
            assert not (folder / 'summary.json').exists(), 'Partial completed output retained; inspect before any retry.'
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = DRIVER
            env['PYTHONFAULTHANDLER'] = '1'
            env['PYTHONUNBUFFERED'] = '1'
            job_path = folder / 'job.json'
            write_json(job_path, job)
            started = time.monotonic()
            with (folder / 'console.log').open('w') as log:
                completed = subprocess.run([str(PYTHON), str(Path(__file__).resolve()), '--child', str(job_path)],
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                state.update(state='failed', failed_job=job, returncode=completed.returncode, updated_at_utc=now())
                write_json(OUT / 'status.json', state)
                raise RuntimeError(f"Evaluation failed: {job['method']} model_{job['iteration_0based']}; see {folder / 'console.log'}")
            result = json.loads((folder / 'verification.json').read_text())
            result['elapsed_wall_s'] = time.monotonic() - started
            write_json(folder / 'verification.json', result)
        state['results'].append(result)
        state['completed'] = index + 1
        state['elapsed_wall_s'] = time.monotonic() - began
        state['updated_at_utc'] = now()
        write_json(OUT / 'status.json', state)
        print(f"[{index+1}/{len(plan['jobs'])}] {job['method']} model_{job['iteration_0based']}: {result['goal_count']}/300, mean_goal_time={result['mean_goal_time_s']}, reused={result['reused_original_final_evaluation']}", flush=True)
    state.update(state='finished', current=None, returncode=0, finished_at_utc=now())
    write_json(OUT / 'status.json', state)
    write_json(OUT / 'all_results.json', state['results'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--child')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.child:
        child(json.loads(Path(args.child).read_text()))
        return
    plan = make_plan()
    if (OUT / 'plan.json').exists():
        old = json.loads((OUT / 'plan.json').read_text())
        assert old['jobs'] == plan['jobs']
        plan = old
    else:
        write_json(OUT / 'plan.json', plan)
    if args.prepare:
        print(json.dumps({k: v for k, v in plan.items() if k != 'jobs'}, ensure_ascii=False, indent=2))
    if args.run:
        batch(plan)


if __name__ == '__main__':
    main()
