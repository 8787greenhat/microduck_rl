"""Summarize unchanged fixed-test checkpoint evaluations; CPU plotting only."""
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
OUT = ROOT / 'outputs/bc_checkpoint_eval_20261025'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    status = json.loads((OUT / 'status.json').read_text())
    assert status['state'] == 'finished' and status['returncode'] == 0 and status['completed'] == 42
    data = json.loads((OUT / 'all_results.json').read_text())
    assert len(data) == 42
    all_rows = []
    episode_rows = []
    by_method = {method: [] for method in ('B', 'C')}
    episodes = {}
    for record in data:
        folder = OUT / record['method'] / f"model_{record['iteration_0based']}"
        assert hashlib.sha256((folder / 'summary.json').read_bytes()).hexdigest() == record['summary_sha256']
        assert hashlib.sha256((folder / 'per_episode.pt').read_bytes()).hexdigest() == record['per_episode_pt_sha256']
        row = {key: record[key] for key in ('method', 'iteration_0based', 'completed_iterations',
                'training_high_level_transitions', 'goal_count', 'goal_rate', 'mean_goal_time_s',
                'median_goal_time_s', 'p90_goal_time_s', 'reused_original_final_evaluation')}
        row.update({key+'_count': record['terminal_and_action_counts'][key]
                    for key in ('fell', 'robot_oob', 'ball_oob', 'timeout', 'kicked', 'contact')})
        all_rows.append(row)
        by_method[row['method']].append(record)
        ep = list(csv.DictReader((folder / 'per_episode.csv').open(encoding='utf-8-sig')))
        assert len(ep) == 300
        episodes[(row['method'], row['iteration_0based'])] = ep
        episode_rows.extend({'method': row['method'], 'iteration_0based': row['iteration_0based'], **r} for r in ep)
    write_csv(OUT / 'learning_curve.csv', all_rows)
    write_csv(OUT / 'all_per_episode.csv', episode_rows)
    pairs = []
    for it in list(range(0, 2000, 100)) + [1999]:
        a, b = (next(r for r in data if r['method'] == method and r['iteration_0based'] == it) for method in ('B', 'C'))
        paired = collections.Counter()
        for ba, ca in zip(episodes[('B', it)], episodes[('C', it)]):
            assert ba['layout_index_0based'] == ca['layout_index_0based']
            assert ba['repeat_index_1based'] == ca['repeat_index_1based']
            bg, cg = int(ba['goal']), int(ca['goal'])
            paired['both_goal' if bg and cg else 'B_only_goal' if bg else 'C_only_goal' if cg else 'neither_goal'] += 1
        pairs.append({'iteration_0based': it, 'completed_iterations': it+1,
                      'training_high_level_transitions': (it+1)*49152,
                      'B_goals': a['goal_count'], 'C_goals': b['goal_count'],
                      'B_goal_pct': 100*a['goal_rate'], 'C_goal_pct': 100*b['goal_rate'],
                      'C_minus_B_percentage_points': 100*(b['goal_rate']-a['goal_rate']),
                      'B_mean_goal_time_s': a['mean_goal_time_s'], 'C_mean_goal_time_s': b['mean_goal_time_s'],
                      **{k: paired[k] for k in ('both_goal', 'B_only_goal', 'C_only_goal', 'neither_goal')}})
    write_csv(OUT / 'paired_comparison.csv', pairs)
    thresholds = {}
    auc = {}
    for method, records in by_method.items():
        records.sort(key=lambda r: r['completed_iterations'])
        thresholds[method] = {}
        for threshold in plan['thresholds_declared_before_evaluation']:
            key = str(round(threshold*100)) + 'pct'
            first = next((r for r in records if r['goal_rate'] >= threshold), None)
            sustained = next((r for i, r in enumerate(records)
                              if all(s['goal_rate'] >= threshold for s in records[i:])), None)
            thresholds[method][key] = {
                'first_observed': {k: first[k] for k in ('iteration_0based', 'completed_iterations',
                        'training_high_level_transitions', 'goal_count', 'goal_rate')} if first else None,
                'first_observed_without_later_sampled_points_below_threshold':
                    {k: sustained[k] for k in ('iteration_0based', 'completed_iterations', 'goal_count')} if sustained else None}
        area = sum((a['goal_rate']+b['goal_rate'])/2 * (b['completed_iterations']-a['completed_iterations'])
                   for a,b in zip(records, records[1:]))
        auc[method] = {'trapezoidal_area_iteration_units': area,
                       'normalized_mean_goal_rate_over_observed_range': area/(records[-1]['completed_iterations']-records[0]['completed_iterations']),
                       'observed_completed_iteration_range': [records[0]['completed_iterations'], records[-1]['completed_iterations']]}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': {k:v for k,v in plan.items() if k != 'jobs'},
              'thresholds': thresholds, 'exploratory_trapezoidal_AUC': auc,
              'paired_checkpoint_results': pairs, 'total_scored_episodes': 12600,
              'new_scored_episodes': 12000, 'reused_original_final_episodes': 600,
              'incomplete_process_attempts': [
                  {'method':'B','iteration':300,'attempts':2,'scored_output_saved':False,'evidence':'incident_B300_startup_20261025_1555'},
                  {'method':'B','iteration':1800,'attempts':1,'scored_output_saved':False,'evidence':'incident_B1800_torch_import'}],
              'training_weights_unchanged': True,
              'existing_evaluator_counter_behavior': {
                  'checkpoint_common_step_counter_restored_even_for_actor_only_load': True,
                  'matched_B_C_points_have_equal_common_step_counter': True,
                  'inherited_COM_randomization_curricula_retained_unchanged': True,
                  'trunk_COM_half_range_m_by_common_step_threshold': [[0,.003],[12000,.005],[24000,.01],[36000,.015]],
                  'head_COM_half_range_m_by_common_step_threshold': [[0,.003],[12000,.005],[24000,.01]],
                  'note': 'Fixed reset JSON is identical at all points. The original evaluator also restores the checkpoint clock, so inherited COM randomization schedules can differ across training progress. B and C at the same checkpoint index have the same clock and schedule. No curriculum or evaluation behavior was changed for this batch.'},
              'interpretation_limitations': [
                  'One training seed for each method; no algorithm-level superiority claim.',
                  'Checkpoint sampling is every 100 iterations; first crossing is observed on this grid, not exact convergence time.',
                  'model_0 is after one update, not an untrained checkpoint.',
                  'Repeated layouts are not independent training seeds.',
                  'C resumed after model_200 with physical episodes/RNG reinitialized; discarded branch excluded.',
                  'Final endpoint evaluations are the preserved original RTX4060 results, not fresh reruns.',
                  'Original evaluator restores checkpoint clock and retains inherited COM curricula; matching indices compare the same schedule, but physical randomization strength is not frozen across the entire training curve.',
                  'Successful goal time conditions on scoring and cannot alone measure overall task quality.',
                  'AUC interpolates sparsely measured points and is descriptive, not a formal significance test.']}
    (OUT / 'comparison_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(11,8),sharex=True,gridspec_kw={'height_ratios':[1.5,1]})
    colors = {'B':'#dc7b16','C':'#2074b0'}
    labels = {'B':'B: PPO', 'C':'C: PPO + reset curriculum'}
    for method, records in by_method.items():
        x = [r['completed_iterations'] for r in records]
        axes[0].plot(x,[100*r['goal_rate'] for r in records],'-o',color=colors[method],label=labels[method],markersize=4)
        axes[1].plot(x,[r['mean_goal_time_s'] if r['mean_goal_time_s'] is not None else math.nan for r in records],
                     '-o',color=colors[method],markersize=4)
    for threshold in (70,90):
        axes[0].axhline(threshold,color='#999999',ls='--',lw=.8)
        axes[0].text(1990,threshold-2,f'{threshold}%',ha='right',va='top',color='#666666',fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvline(201,color='#7e5599',ls=':',lw=1)
    axes[0].text(220,5,'C resumed after 201 updates',color='#7e5599',fontsize=8)
    axes[0].set_ylim(-3,103)
    axes[0].set_ylabel('Valid goals (%)')
    axes[0].legend(loc='lower right')
    axes[1].set_ylabel('Mean time among goals (s)')
    axes[1].set_xlabel('Completed PPO iterations (49,152 high-level transitions per iteration)')
    axes[1].set_xlim(0,2020)
    fig.suptitle('Fixed-test learning curves | training seed 1\n100 layouts x 3 repeats, evaluation seed 0, identical official evaluator')
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(OUT / 'learning_speed.png',dpi=170)
    plt.close(fig)
    def point_text(method, threshold):
        item=thresholds[method][threshold]['first_observed']
        return f"model_{item['iteration_0based']}（已完成 {item['completed_iterations']} 次更新，{item['goal_count']}/300 球）" if item else '未在存檔點觀察到達標'
    lines=['B/C seed 1 固定評測學習速度比較', '',
           '原本固定 100 場景各測 3 次、eval RNG seed 0、同一台 RTX4060、同一套官方技能及 commit 7fa6298；沒有重新訓練或改評測設定。',
           '21 個共同 checkpoint，每份 300 局。新跑 40 份中途模型，最後 2 份沿用既有正式評測；合計 12,600 局（本次新增 12,000 局）。', '',
           f"首次觀察到達 70%：B {point_text('B','70pct')}；C {point_text('C','70pct')}。",
           f"首次觀察到達 90%：B {point_text('B','90pct')}；C {point_text('C','90pct')}。", '',
           '各共同存檔結果（完整逐局 CSV 另存 all_per_episode.csv）：',
           'model / 已完成更新 / B 進球 / C 進球 / B 成功平均秒數 / C 成功平均秒數']
    for row in pairs:
        tb=f"{row['B_mean_goal_time_s']:.3f}" if row['B_mean_goal_time_s'] is not None else '-'
        tc=f"{row['C_mean_goal_time_s']:.3f}" if row['C_mean_goal_time_s'] is not None else '-'
        lines.append(f"{row['iteration_0based']} / {row['completed_iterations']} / {row['B_goals']}/300 ({row['B_goal_pct']:.1f}%) / {row['C_goals']}/300 ({row['C_goal_pct']:.1f}%) / {tb} / {tc}")
    lines += ['', '解讀限制：',
              '這是各一個 training seed 的實測學習曲線；三次重複場景不是三個訓練 seeds，不能直接推論所有 B/C 訓練結果。',
              '每 100 iterations 才有共同 checkpoint，因此達標只代表第一個測得的存檔點，沒有證明確切在哪個 iteration 學會或此後連續穩定。',
              'model_100 標號為 100，但已完成 101 次更新；model_0 已完成第一次更新，不是未訓練模型。',
              'C 在 model_200 後恢復；有效曲線使用原始 0–200 加恢復後 201–1999，排除舊 201–267 與短測試，物理 episode/RNG 重初始化已揭露。',
              '成功進球時間只統計進球局，必須與進球率一起看。最後點沿用先前保存的評測，不挑選重跑結果。',
              '原評測腳本會恢復 checkpoint 的步數，並保留共用的 CoM 隨機化課程。因此相同存檔點的 B/C 使用相同幅度，但不同訓練進度之間的物理隨機化幅度並非完全凍結；固定場景 JSON 始終相同。這是既有流程行為，本次未為此修改設定；達標點應解讀為此原流程下的首次觀察。',
              'B model_300 有兩次未完成的原生程序崩潰，沒有保留的評測分數；保留診斷後改用 CPU 反序列化再載入 GPU，逐張量核對權重相同後才續跑。所有已完成分數採首次完成結果，未挑選成功重跑。',
              'B model_1800 另在 import torch 階段崩潰一次，尚未載入模型；保存堆疊後以相同設定接續未完成點，沒有重跑已完成分數。上述未完成程序的額外時間不列為已完成評測局數。',
              '曲線面積是共同存檔點的梯形近似，只作描述，不是統計顯著性檢定。',
              '正式最終模型仍固定為 model_1999.pt；沒有把本次曲線最高分的中途模型挑出來取代原結果。',
              '所有 checkpoint 原始 SHA256 保持不變；這次沒有改 repo 訓練來源或評測來源。']
    (OUT / '學習速度比較報告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(json.dumps({'thresholds':thresholds,'AUC':auc,'final':pairs[-1]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
