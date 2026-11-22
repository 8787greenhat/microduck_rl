"""CPU-only descriptive analysis; fixed geometric bins, no benchmark changes."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as stats

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
ROOT=Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
OUT=ROOT/'outputs/bc_multiseed_analysis_20261121'
TRAIN=ROOT/'outputs/multiseed_20261120_legacy_seed1'
PLAN=json.loads((OUT/'plan.json').read_text())

def save(name,data): (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def csvout(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def wrapped(a): return (a+math.pi)%(2*math.pi)-math.pi
def bin_index(value,bounds): return sum(value>=b for b in bounds)
def getrows(folder):
    rows=list(csv.DictReader((Path(folder)/'per_episode.csv').open(encoding='utf-8-sig')))
    assert len(rows)==300
    for row in rows:
        for key in row: row[key]=float(row[key]) if key=='elapsed_s' else int(row[key])
    assert {(r['layout_index_0based'],r['repeat_index_1based']) for r in rows}=={(i,r) for i in range(100) for r in (1,2,3)}
    return rows

def geometry():
    fixed=ROOT/'outputs/seed1_evaluation/soccer_eval_set.json'
    assert hashlib.sha256(fixed.read_bytes()).hexdigest()==PLAN['fixed_eval_set_sha256']
    layouts=json.loads(fixed.read_text())['episodes']
    result=[]
    for i,layout in enumerate(layouts):
        rx,ry,yaw=layout['robot_xy_yaw'];bx,by=layout['ball_xy'];gx,gy,gyaw=layout['goal_xy_yaw']
        direction=math.atan2(gy-by,gx-bx)
        features={'layout_index_0based':i,'robot_ball_distance_m':math.hypot(rx-bx,ry-by),
            'initial_heading_error_deg':abs(math.degrees(wrapped(yaw-direction))),
            'ball_goal_distance_m':math.hypot(gx-bx,gy-by),
            'shot_line_obliqueness_deg':abs(math.degrees(wrapped(direction-gyaw)))}
        for name,bounds in PLAN['subgroup_boundaries_prespecified'].items():
            features[name+'_bin']=bin_index(features[name],bounds)
        result.append(features)
    csvout('layout_geometry.csv',result)
    return result

def subgroup_analysis(records,features):
    groups=[]
    for it in sorted({r['iteration_0based'] for r in records}):
        for name,bounds in PLAN['subgroup_boundaries_prespecified'].items():
            for group in range(3):
                ids={f['layout_index_0based'] for f in features if f[name+'_bin']==group}
                for seed in (1,2,3):
                    for method in ('B','C'):
                        item=next((r for r in records if r['iteration_0based']==it and r['training_seed']==seed and r['method']==method),None)
                        if item is None: continue
                        rows=[r for r in getrows(item['data_folder']) if r['layout_index_0based'] in ids]
                        goals=[r for r in rows if r['goal']]
                        groups.append({'iteration_0based':it,'feature':name,'bin_index':group,
                            'bin_lower_inclusive':0 if group==0 else bounds[group-1],
                            'bin_upper_exclusive':bounds[group] if group<2 else None,
                            'training_seed':seed,'method':method,'layouts':len(ids),'episodes':len(rows),
                            'goals':len(goals),'goal_rate':len(goals)/len(rows) if rows else None,
                            'mean_goal_time_s':stats.mean(r['elapsed_s'] for r in goals) if goals else None})
    csvout('subgroups_per_seed.csv',groups)
    aggregate=[]
    for it,name,group in sorted({(r['iteration_0based'],r['feature'],r['bin_index']) for r in groups}):
        row={'iteration_0based':it,'feature':name,'bin_index':group}
        for method in ('B','C'):
            selected=[r for r in groups if (r['iteration_0based'],r['feature'],r['bin_index'],r['method'])==(it,name,group,method)]
            rates=[r['goal_rate'] for r in selected if r['goal_rate'] is not None]
            row[method+'_goal_rate_mean']=stats.mean(rates) if rates else None
            row[method+'_goal_rate_sd']=stats.stdev(rates) if len(rates)>1 else None
            row['layouts_per_seed']=selected[0]['layouts']
        row['C_minus_B_pp']=100*(row['C_goal_rate_mean']-row['B_goal_rate_mean']) if row['C_goal_rate_mean'] is not None else None
        aggregate.append(row)
    csvout('subgroup_comparison.csv',aggregate)
    return aggregate

def final_analysis(features):
    records=[];final=[];failures=[]
    for job in PLAN['jobs']:
        if job['iteration_0based']!=1999:continue
        folder=Path(job['reuse'])
        rows=getrows(folder)
        summary=json.loads((folder/'summary.json').read_text())
        goals=[r for r in rows if r['goal']]
        assert math.isclose(summary['rates']['goal'],len(goals)/300,abs_tol=1e-6)
        final.append({'method':job['method'],'training_seed':job['training_seed'],'goal_count':len(goals),
            'goal_rate':len(goals)/300,'mean_goal_time_s':stats.mean(r['elapsed_s'] for r in goals),
            'median_first_kick_prep_error_m':summary['median_prep_dist_at_kick_m'],
            'median_first_kick_heading_error_deg':math.degrees(summary['median_yaw_err_at_kick_rad']),
            'summary_sha256':hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest()})
        records.append({**job,'data_folder':str(folder)})
        combinations=collections.Counter('+'.join(k for k in ('fell','robot_oob','ball_oob','timeout') if r[k]) for r in rows if not r['goal'])
        for combination,count in combinations.items():
            failures.append({'method':job['method'],'training_seed':job['training_seed'],'failure_flags_combination':combination,'count':count})
    csvout('final_results.csv',final);csvout('final_failure_combinations.csv',failures)
    aggregate={}
    for method in ('B','C'):
        selected=[r for r in final if r['method']==method]
        aggregate[method]={'goal_rate_mean':stats.mean(r['goal_rate'] for r in selected),
            'goal_rate_sample_sd':stats.stdev(r['goal_rate'] for r in selected),
            'mean_of_seed_success_time_means_s':stats.mean(r['mean_goal_time_s'] for r in selected),
            'success_time_sample_sd_s':stats.stdev(r['mean_goal_time_s'] for r in selected)}
    pairs=[]
    for i in range(100):
        row={'layout_index_0based':i,**{k:v for k,v in features[i].items() if k!='layout_index_0based'}}
        for method in ('B','C'):
            selected=[r for item in records if item['method']==method for r in getrows(item['data_folder']) if r['layout_index_0based']==i]
            row[method+'_goals_out_of_9']=sum(r['goal'] for r in selected)
        row['C_minus_B_goals_out_of_9']=row['C_goals_out_of_9']-row['B_goals_out_of_9']
        pairs.append(row)
    csvout('final_layout_pairs.csv',pairs)
    save('final_analysis.json',{'records':final,'aggregate':aggregate,'failure_combinations':failures,
        'paired_layouts':{'C_more_goals':sum(r['C_minus_B_goals_out_of_9']>0 for r in pairs),
            'B_more_goals':sum(r['C_minus_B_goals_out_of_9']<0 for r in pairs),
            'tied':sum(r['C_minus_B_goals_out_of_9']==0 for r in pairs)},
        'interpretation':'Three training seeds; repeated layouts are not independent training runs. Descriptive, no significance claim.'})
    return final,aggregate,records

def stage_analysis():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    result=[]
    for seed in (2,3):
        verify=json.loads((TRAIN/f'formal_C_seed{seed}/verification.json').read_text())
        acc=EventAccumulator(verify['actual_run_dir'],size_guidance={'scalars':0});acc.Reload()
        tags=[t for t in acc.Tags()['scalars'] if 'stage' in t.lower() and 'reset' in t.lower()]
        assert len(tags)==1,tags
        values=acc.Scalars(tags[0])
        for stage in (1,2):
            first=next(e for e in values if e.value>=stage)
            result.append({'training_seed':seed,'stage':stage,'first_logged_iteration_0based':first.step,
                'first_logged_completed_iterations':first.step+1,'tag':tags[0]})
    csvout('curriculum_stage_timing.csv',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final-only',action='store_true');args=parser.parse_args()
    features=geometry()
    final,aggregate,finalrecords=final_analysis(features)
    stages=stage_analysis()
    if args.final_only:
        groups=subgroup_analysis(finalrecords,features)
        plot_geometry(groups)
        print(json.dumps({'aggregate':aggregate,'stages':stages},indent=2));return
    status=json.loads((OUT/'status.json').read_text());assert status['state']=='finished' and status['completed']==48
    data=json.loads((OUT/'all_results.json').read_text());assert len(data)==48
    curves=[];thresholds=[];areas=[];detailed=[]
    for r in data:
        folder=Path(r['out'])
        assert hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest()==r['summary_sha256']
        assert hashlib.sha256((folder/'per_episode.pt').read_bytes()).hexdigest()==r['per_episode_sha256']
        curves.append({k:r[k] for k in ('method','training_seed','iteration_0based','completed_iterations','training_high_level_transitions','goal_count','goal_rate','mean_goal_time_s')})
        for row in getrows(folder): detailed.append({'method':r['method'],'training_seed':r['training_seed'],'iteration_0based':r['iteration_0based'],**row})
    csvout('learning_curve.csv',curves);csvout('all_per_episode.csv',detailed)
    for seed in (1,2,3):
        for method in ('B','C'):
            early=sorted([r for r in curves if r['training_seed']==seed and r['method']==method and r['iteration_0based']<=600],key=lambda r:r['iteration_0based'])
            assert len(early)==7
            for threshold in (.7,.9):
                first=next((r for r in early if r['goal_rate']>=threshold),None)
                stable=next((r for i,r in enumerate(early) if all(x['goal_rate']>=threshold for x in early[i:])),None)
                thresholds.append({'training_seed':seed,'method':method,'threshold':threshold,
                    'first_observed_iteration_0based':first['iteration_0based'] if first else None,
                    'first_observed_completed_iterations':first['completed_iterations'] if first else None,
                    'first_observed_transitions':first['training_high_level_transitions'] if first else None,
                    'first_sampled_point_without_later_sample_below_threshold_through_600':stable['iteration_0based'] if stable else None})
            area=sum((a['goal_rate']+b['goal_rate'])/2*(b['completed_iterations']-a['completed_iterations']) for a,b in zip(early,early[1:]))/600
            areas.append({'training_seed':seed,'method':method,'window_start_completed_iterations':1,
                'window_end_completed_iterations':601,'normalized_trapezoidal_goal_rate':area})
    csvout('early_thresholds.csv',thresholds);csvout('early_AUC.csv',areas)
    selected=[{**r,'data_folder':r['out']} for r in data if r['iteration_0based'] in (100,200,600,1999)]
    subgroups=subgroup_analysis(selected,features)
    paired=[]
    for it in PLAN['points_0based']:
        row={'iteration_0based':it,'completed_iterations':it+1}
        for method in ('B','C'):
            rs=[r for r in curves if r['method']==method and r['iteration_0based']==it]
            row[method+'_mean_pct']=100*stats.mean(r['goal_rate'] for r in rs)
            row[method+'_sd_pp']=100*stats.stdev(r['goal_rate'] for r in rs)
        row['C_minus_B_pp']=row['C_mean_pct']-row['B_mean_pct'];paired.append(row)
    csvout('learning_curve_aggregate.csv',paired)
    summary={'final_aggregate':aggregate,'early_thresholds':thresholds,'early_AUC':areas,
        'paired_learning_curve':paired,'curriculum_stages':stages,'subgroups':subgroups,
        'episodes_analyzed':14400,'new_scored_episodes':8400,'reused_scored_episodes':6000,
        'protocol':{k:v for k,v in PLAN.items() if k!='jobs'}}
    save('analysis_summary.json',summary)
    plot(curves,areas,subgroups,features)
    report(summary,final)
    print(json.dumps({'thresholds':thresholds,'areas':areas,'learning_curve':paired},ensure_ascii=False,indent=2))

def plot(curves,areas,subgroups,features):
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'B':'#d47b19','C':'#1977b7'}
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),sharey=True)
    for seed,ax in zip((1,2,3),axes):
        for method in ('B','C'):
            early=sorted([r for r in curves if r['training_seed']==seed and r['method']==method and r['iteration_0based']<=600],key=lambda r:r['iteration_0based'])
            ax.plot([r['completed_iterations'] for r in early],[100*r['goal_rate'] for r in early],'-o',color=colors[method],label=method)
        for threshold in (70,90):ax.axhline(threshold,color='#999',linestyle='--',linewidth=.7)
        ax.set_title(f'Training seed {seed}'+(' (C resumed after 201 updates)' if seed==1 else ''),fontsize=10)
        ax.set_xlabel('Completed PPO updates');ax.set_ylim(-2,103);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Fixed-test valid goals (%)')
    fig.suptitle('Early learning: identical 100 layouts x 3 repeats | evaluation RNG seed 0')
    fig.tight_layout();fig.savefig(OUT/'early_learning_by_seed.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for method in ('B','C'):
        xs=[];means=[];sds=[]
        for it in range(0,601,100):
            rs=[r['goal_rate']*100 for r in curves if r['method']==method and r['iteration_0based']==it]
            xs.append(it+1);means.append(stats.mean(rs));sds.append(stats.stdev(rs))
        ax.plot(xs,means,'-o',color=colors[method],label=f'{method}: mean of 3 training seeds')
        ax.fill_between(xs,np.array(means)-sds,np.array(means)+sds,alpha=.15,color=colors[method])
    ax.set(xlabel='Completed PPO updates',ylabel='Valid goals (%)',ylim=(-3,103),title='Early learning | shaded band = sample SD, not confidence interval')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'early_learning_mean_sd.png',dpi=170);plt.close(fig)
    plot_geometry(subgroups)

def plot_geometry(subgroups):
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'B':'#d47b19','C':'#1977b7'}
    names=list(PLAN['subgroup_boundaries_prespecified'])
    labels=[['<0.30 m','0.30-0.45 m','>=0.45 m'],['<30 deg','30-60 deg','>=60 deg'],['<0.70 m','0.70-0.90 m','>=0.90 m'],['<10 deg','10-20 deg','>=20 deg']]
    titles=['Initial robot-ball distance','Initial heading error','Initial ball-goal distance','Shot line obliqueness']
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
    for name,labels_,title,ax in zip(names,labels,titles,axes.flat):
        rows=sorted([r for r in subgroups if r['iteration_0based']==1999 and r['feature']==name],key=lambda r:r['bin_index'])
        for method,shift in (('B',-.18),('C',.18)):
            bars=ax.bar(np.arange(3)+shift,[100*r[method+'_goal_rate_mean'] for r in rows],width=.36,color=colors[method],label=method)
            ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
        ax.set_xticks(range(3),[f'{label}\n{r["layouts_per_seed"]} layouts' for label,r in zip(labels_,rows)])
        ax.set_title(title);ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left',ncol=2)
    for ax in axes[:,0]: ax.set_ylabel('Valid goals (%)')
    fig.suptitle('Final checkpoint: fixed geometric groups | mean of 3 training seeds')
    fig.tight_layout();fig.savefig(OUT/'final_geometry_groups.png',dpi=170);plt.close(fig)

def report(summary,final):
    at100=next(r for r in summary['paired_learning_curve'] if r['iteration_0based']==100)
    area_means={m:stats.mean(r['normalized_trapezoidal_goal_rate'] for r in summary['early_AUC'] if r['method']==m) for m in ('B','C')}
    lines=['B/C 三個 seeds：前期學習與場景分析（僅本機，未更新 GitHub）','',
        '沿用原固定 100 組場景，各 3 次，共 300 局／checkpoint；eval RNG seed 0。',
        '本次新增 28 份前期評測（8400 局），重用 20 份既有評測（6000 局）。沒有重新訓練、改題、改 PPO／課程或挑選最佳 checkpoint。','',
        '先看結論：',
        f"model_100 的三個 seeds 平均：B {at100['B_mean_pct']:.1f}%，C {at100['C_mean_pct']:.1f}%。這只代表這個前期測點，不能直接等同整體學得更快。",
        f"前 1～601 次更新的平均曲線面積：B {area_means['B']*100:.2f}%，C {area_means['C']*100:.2f}%；面積越高代表這段訓練期間的整體進球率越高，僅為稀疏測點的描述性近似。",
        f"最終進球率：B {summary['final_aggregate']['B']['goal_rate_mean']*100:.2f}%，C {summary['final_aggregate']['C']['goal_rate_mean']*100:.2f}%；C 沒有穩定提高最終平均表現。",
        '課程不能只用好／爛二分。早期開始進球、達到高進球率、最終表現、種子變動是不同面向，請對照下列三個 seeds 的數據。','',
        '角度定義：initial_heading_error_deg 是初始朝向與球到球門方向的偏差；shot_line_obliqueness_deg 是球到球門連線相對球門正向的斜角。兩者不同，也都不是實際踢出球的飛行方向。',
        '距離定義：robot_ball_distance_m 是起始機器人到球的平面距離；ball_goal_distance_m 是起始球到球門中心的平面距離。','',
        '一、三個 seeds 的前期進球率（平均 ± sample SD；不是信賴區間）']
    for row in summary['paired_learning_curve']:
        lines.append(f"model_{row['iteration_0based']}（完成 {row['completed_iterations']} 次）：B {row['B_mean_pct']:.1f}±{row['B_sd_pp']:.1f}%，C {row['C_mean_pct']:.1f}±{row['C_sd_pp']:.1f}%；C-B {row['C_minus_B_pp']:+.1f} 個百分點。")
    lines+=['','二、首次在前 600 的共同存檔觀察到達標（每 100 次取樣，不是精確達標時間）']
    for r in summary['early_thresholds']:
        first='未達標' if r['first_observed_iteration_0based'] is None else f"model_{r['first_observed_iteration_0based']}"
        lines.append(f"seed {r['training_seed']} {r['method']}：{r['threshold']*100:.0f}% 首次 {first}；前 600 後續取樣未跌回的起點 {r['first_sampled_point_without_later_sample_below_threshold_through_600']}。")
    lines+=['','三、前期曲線面積（1～601 次更新的梯形近似／600；描述性指標，並非顯著性檢定）']
    for r in summary['early_AUC']:lines.append(f"seed {r['training_seed']} {r['method']}：{r['normalized_trapezoidal_goal_rate']*100:.2f}%")
    lines+=['','四、最終模型分組（固定邊界事先寫入 plan.json；不把分組結果當成新的獨立訓練 seeds）']
    for r in summary['subgroups']:
        if r['iteration_0based']==1999:
            lines.append(f"{r['feature']} bin {r['bin_index']}，每 seed {r['layouts_per_seed']} 個場景：B {r['B_goal_rate_mean']*100:.2f}%，C {r['C_goal_rate_mean']*100:.2f}%，C-B {r['C_minus_B_pp']:+.2f} pp。")
    lines+=['','五、課程階段（日誌中首次觀察）']
    for r in summary['curriculum_stages']:lines.append(f"C seed {r['training_seed']}：stage {r['stage']} 首次記錄在 iteration {r['first_logged_iteration_0based']}（完成 {r['first_logged_completed_iterations']} 次更新）。")
    lines+=['','限制：','只有 3 個 training seeds；300 局包含重複場景，不能當 300 個獨立訓練實驗。',
        '相同 training seed 的 B/C 不保證初始網路權重與所有後續隨機事件逐項相同；課程會改變亂數使用順序。',
        'model_0 已完成一次 PPO 更新，不是未訓練模型；達標時間僅在共同存檔點觀察。',
        'C seed 1 曾從 model_200 恢復，物理 episodes 與 RNG 重初始化，原始限制保留。',
        '原評測會恢復 checkpoint clock，共用 CoM 隨機化會隨訓練步數改變；相同存檔點 B/C 相同，但不能把整條曲線當作所有物理條件完全不變。',
        '幾何分組為事後描述性分析；各維度可能相關，不代表角度或距離的獨立因果效應。',
        'C 到達 stage 2 後仍有 20% 的 reset 抽取較早階段；所以不是 100% 重置都與 B 相同。這是重置比例，不等於 20% 訓練步數。',
        '沿用舊版 seed 1 協定；未套用尚未合併 PR #4 的啟動空 episode 計數修正，結論僅適用這次實測版本。',
        '課程升級使用該階段累積成功率或 300-iteration 上限，不是最近一段 episode 的移動平均。不同 seed 因此可能有不同升級時間；尚未以消融實驗證明其因果影響。',
        '成功時間只計進球局；最終模型固定 model_1999，不用最高分中途模型替換。',
        '正式學習速度達標的穩定性僅指前 600 內的已測點；601～1999 沒有新增中途評測，不能宣稱全程不跌回。']
    (OUT/'三seed課程分析報告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')

if __name__=='__main__':main()
