"""Verify published seed-1 results without GPU; optionally verify PTs or redraw CSVs.

Default: Python standard library only. --tensors requires the existing locked
PyTorch; --plot requires existing matplotlib. Neither option runs a simulation.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

DATA = Path(__file__).resolve().parents[1]
POINTS = list(range(0, 2000, 100)) + [1999]
FLAGS = ('goal', 'fell', 'robot_oob', 'ball_oob', 'timeout', 'kicked', 'contact')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def verify(tensors=False):
    manifest = read_json(DATA / 'file_manifest.json')
    listed = {}
    for item in manifest['files']:
        path = (DATA / item['path']).resolve()
        assert path.is_relative_to(DATA) and path.is_file(), item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['path']
        assert item['path'] not in listed
        listed[item['path']] = item['sha256']
    actual = {p.relative_to(DATA).as_posix() for p in DATA.rglob('*')
              if p.is_file() and p.name != 'file_manifest.json' and '__pycache__' not in p.parts}
    assert set(listed) == actual, sorted(set(listed) ^ actual)
    protocol = read_json(DATA / 'protocol.json')
    assert sha(DATA / 'evaluation_set/soccer_eval_set.json') == protocol['fixed_eval_set_sha256']
    assert protocol['fixed_eval_set_sha256'] == '3a59dee2c22b5129fcb10685192630443a29550790ee5ccbffc04e414d053464'
    assert len(read_json(DATA / 'evaluation_set/soccer_eval_set.json')['episodes']) == 100
    assert protocol['layouts'] == 100 and protocol['repeats'] == 3 and protocol['evaluation_rng_seed'] == 0
    checkpoint_manifest = read_json(DATA / 'inputs/checkpoint_manifest.json')
    checkpoint_records = {(row['method'], row['iteration_0based']): row for row in checkpoint_manifest['checkpoint_records']}
    assert len(checkpoint_records) == 42
    c_zip = DATA / 'inputs/method_C_seed1_selected_checkpoints.zip'
    assert sha(c_zip) == checkpoint_manifest['C_archive_sha256']
    with zipfile.ZipFile(c_zip) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {f'model_{i}.pt' for i in POINTS}
        for it in POINTS:
            assert hashlib.sha256(archive.read(f'model_{it}.pt')).hexdigest() == checkpoint_records[('C', it)]['checkpoint_sha256']
    torch = None
    if tensors:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
    curve = read_csv(DATA / 'learning_curve.csv')
    assert len(curve) == 42
    curve_lookup = {(row['method'], int(row['iteration_0based'])): row for row in curve}
    all_rows = read_csv(DATA / 'all_per_episode.csv')
    assert len(all_rows) == 12600
    flat = collections.defaultdict(list)
    for row in all_rows:
        flat[(row['method'], int(row['iteration_0based']))].append(
            {key: value for key, value in row.items() if key not in ('method', 'iteration_0based')})
    for method in ('B', 'C'):
        for it in POINTS:
            key = (method, it)
            folder = DATA / 'evaluations' / method / f'model_{it}'
            rows = read_csv(folder / 'per_episode.csv')
            summary = read_json(folder / 'summary.json')
            verification = read_json(folder / 'verification.json')
            assert len(rows) == summary['envs'] == 300 and rows == flat[key]
            assert verification['checkpoint_sha256'] == checkpoint_records[key]['checkpoint_sha256']
            assert int(curve_lookup[key]['completed_iterations']) == it + 1
            assert int(curve_lookup[key]['training_high_level_transitions']) == (it + 1) * 49152
            for i, row in enumerate(rows):
                assert int(row['layout_index_0based']) == i % 100 and int(row['repeat_index_1based']) == i // 100 + 1
                assert all(int(row[k]) in (0, 1) for k in FLAGS)
                assert any(int(row[k]) for k in FLAGS[:5])
                assert 0 < float(row['elapsed_s']) <= 12.2
            goals = sum(int(row['goal']) for row in rows)
            assert goals == verification['goal_count'] == int(curve_lookup[key]['goal_count'])
            assert math.isclose(goals / 300, summary['rates']['goal'], abs_tol=1e-6)
            for flag in FLAGS:
                assert math.isclose(sum(int(row[flag]) for row in rows) / 300, summary['rates'][flag], abs_tol=1e-6)
            times = [float(row['elapsed_s']) for row in rows if int(row['goal'])]
            if times:
                assert math.isclose(sum(times) / len(times), summary['mean_goal_time_s'], abs_tol=1e-5)
            else:
                assert summary['mean_goal_time_s'] is None
            if tensors:
                pt = torch.load(folder / 'per_episode.pt', weights_only=True, map_location='cpu')
                assert all(pt[k].numel() == 300 for k in (*FLAGS, 'steps'))
                for i, row in enumerate(rows):
                    assert all(int(pt[k][i]) == int(row[k]) for k in FLAGS)
                    assert math.isclose(float(pt['steps'][i]) * .1, float(row['elapsed_s']), abs_tol=1e-5)
    comparison = read_json(DATA / 'comparison_summary.json')
    for method in ('B', 'C'):
        for percent in (70, 90):
            first = next(i for i in POINTS if int(curve_lookup[(method, i)]['goal_count']) / 300 >= percent / 100)
            assert first == comparison['thresholds'][method][f'{percent}pct']['first_observed']['iteration_0based']
    pairs = read_csv(DATA / 'paired_comparison.csv')
    assert len(pairs) == 21 and [int(row['iteration_0based']) for row in pairs] == POINTS
    for row in pairs:
        it = int(row['iteration_0based'])
        assert int(row['B_goals']) == int(curve_lookup[('B', it)]['goal_count'])
        assert int(row['C_goals']) == int(curve_lookup[('C', it)]['goal_count'])
    return {'manifest_files': len(listed), 'checkpoint_pairs': 21, 'evaluation_outputs': 42,
            'per_episode_rows': 12600, 'C_checkpoint_zip_crc_and_hashes': 'passed',
            'csv_summary_thresholds': 'passed', 'PT_tensor_comparison': 'passed' if tensors else 'not requested (PT bytes SHA256 verified)'}


def plot(output):
    output = output.expanduser().resolve()
    assert not output.exists(), 'Choose a new output path; preserve existing plots.'
    assert not output.is_relative_to(DATA), 'Write rebuilt figures outside the audited evidence directory.'
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib.pyplot as plt
    curve = read_csv(DATA / 'learning_curve.csv')
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
    for method, color, label in (('B', '#dc7b16', 'B: PPO'), ('C', '#2074b0', 'C: PPO + reset curriculum')):
        rows = sorted((r for r in curve if r['method'] == method), key=lambda r: int(r['completed_iterations']))
        x = [int(r['completed_iterations']) for r in rows]
        axes[0].plot(x, [float(r['goal_rate']) * 100 for r in rows], '-o', color=color, label=label, markersize=4)
        axes[1].plot(x, [float(r['mean_goal_time_s']) if r['mean_goal_time_s'] else math.nan for r in rows], '-o', color=color, markersize=4)
    for level in (70, 90):
        axes[0].axhline(level, color='#999999', ls='--', lw=.8)
        axes[0].text(1990, level-2, f'{level}%', ha='right', va='top', color='#666666', fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvline(201, color='#7e5599', ls=':', lw=1)
    axes[0].text(220, 5, 'C resumed after 201 updates', color='#7e5599', fontsize=8)
    axes[0].set(ylim=(-3, 103), ylabel='Valid goals (%)')
    axes[0].legend(loc='lower right')
    axes[1].set(xlim=(0, 2020), ylabel='Mean time among goals (s)',
                xlabel='Completed PPO iterations (49,152 high-level transitions per iteration)')
    fig.suptitle('Fixed-test learning curves | training seed 1\n100 layouts x 3 repeats, evaluation seed 0, identical official evaluator')
    fig.tight_layout(rect=(0, 0, 1, .94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tensors', action='store_true', help='Also decode PT tensors with existing PyTorch on CPU.')
    parser.add_argument('--plot', type=Path, help='Rebuild the figure from CSV to a new path outside this package.')
    args = parser.parse_args()
    result = verify(args.tensors)
    if args.plot:
        plot(args.plot)
        result['rebuilt_plot'] = str(args.plot)
    print(json.dumps(result, ensure_ascii=False, indent=2))
