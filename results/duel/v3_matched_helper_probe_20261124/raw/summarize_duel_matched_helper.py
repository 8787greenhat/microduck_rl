"""Read-only audit of the frozen-policy 2x2 shared-helper experiment."""
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/duel_matched_helper_probe_20261124'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_bytes((json.dumps(v,indent=2,ensure_ascii=False)+'\n').encode())
status=read(OUT/'status.json')
assert status['state']=='complete' and len(status['workers'])==8
assert all(w['returncode']==0 for w in status['workers'])
spec=importlib.util.spec_from_file_location('duel_protocol',ROOT/'work/microduck_duel_v3/src/mjlab_microduck/duel/protocol.py')
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
conditions=('P_H_vs_A','P_H_vs_R_H','R_H_vs_A','R_H_vs_R_H')
results={};all_rows=[];manifests=[]
for condition in conditions:
    rows=[];audit=[]
    for side in (0,1):
        folder=OUT/f'{condition}_side{side}'
        summary=read(folder/'summary.json');manifest=read(folder/'manifest.json');items=read(folder/'episodes.json')
        assert summary['complete'] and not summary['new_training'] and summary['actor_and_normalizer_unchanged']
        assert sha(folder/'per_episode.csv')==summary['per_episode_csv_sha256']
        assert protocol.summarize(items)==summary['total']
        assert len(items)==64 and len({(r['scene_id'],r['repeat']) for r in items})==64
        assert all(r['learner_side']==side and r['opponent_variant']==condition for r in items)
        assert summary['actor_used']==manifest['actor_used']==condition.startswith('P_')
        assert manifest['source']['source_commit']=='e8b9756c03ae66a693275ee84b61a36aab367b2f' and not manifest['source']['source_dirty']
        assert summary['checkpoint_sha256']==manifest['checkpoint_sha256']==status['checkpoint_sha256']
        assert manifest['wrapper_sha256']==sha(ROOT/'work/eval_duel_matched_helper.py')
        assert manifest['opponent_wrapper_sha256']==sha(ROOT/'work/duel_matched_helper.py')
        if manifest['opponent_helper_active']:assert summary['helper_opponent_guard_violations']==0
        rows+=items;manifests.append(manifest)
        audit.append(dict(side=side,guard_verified_starts_in_entire_simulation=summary['helper_opponent_guard_verified_starts'],guard_violations=summary['helper_opponent_guard_violations']))
    assert len(rows)==128 and {(r['scene_id'],r['repeat'],r['learner_side']) for r in rows}==set(itertools.product(range(32),range(2),range(2)))
    results[condition]=dict(total=protocol.summarize(rows),
        blue_kick_rounds=sum(r['learner_kick_attempts']>0 for r in rows),orange_kick_rounds=sum(r['opponent_kick_attempts']>0 for r in rows),
        by_side={str(s):protocol.summarize([r for r in rows if r['learner_side']==s]) for s in (0,1)},opponent_helper_audit=audit)
    all_rows+=rows
for manifest in manifests:
    for key in ('source','checkpoint_sha256','scene_ids','repeats','num_envs','eval_rng_seed','round_seconds','original_protocol','shared_v3_helper_config','rule_request'):
        assert manifest[key]==manifests[0][key],key
assert len(all_rows)==512
assert len({(r['opponent_variant'],r['scene_id'],r['repeat'],r['learner_side']) for r in all_rows})==512
def net(condition):return results[condition]['total']['goal_difference_per_episode']
contrasts=dict(blue_PPO_vs_rule_against_original_A_net_goal_difference_per_episode=net('P_H_vs_A')-net('R_H_vs_A'),
    blue_PPO_vs_rule_against_matched_rule_net_goal_difference_per_episode=net('P_H_vs_R_H')-net('R_H_vs_R_H'),
    caveat='Descriptive whole-policy contrasts under common helper, one trained checkpoint and a small repeated-layout subset; not a universal percentage decomposition of causal PPO contribution.')
report=dict(complete=True,official_fixed600=False,new_training=False,optimizer_updates=0,training_seed=1,
    design='2 blue high-level decisions x 2 orange helper conditions',episodes_per_condition=128,total_episodes=512,
    scene_ids=list(range(32)),repeats=[0,1],sides=[0,1],batch_per_side=64,eval_rng_seed=0,
    source_commit=manifests[0]['source']['source_commit'],checkpoint_sha256=status['checkpoint_sha256'],
    shared_helper_class='mjlab_microduck.duel.v3_env.FastStopController',shared_helper_config=manifests[0]['shared_v3_helper_config'],
    cpu_checks_passed=7,all_worker_returncodes_zero=True,all_loaded_actor_and_normalizer_unchanged=True,
    rule_request=manifests[0]['rule_request'],results=results,contrasts=contrasts,
    limitations=['The frozen PPO was trained against original A, not the newly matched-helper opponent; this also probes distribution shift.',
        'R_H is a new deterministic preparation-pose rule with a greedy request; it is not the unchanged original A.',
        'One training seed and 32 previously evaluated layouts x 2 repeats x sides; repetitions are not independent training seeds.',
        'Only compare this run with its newly rerun P_H_vs_A reference, not historical fixed600 or the earlier opponent probe.',
        'Same evaluation seed does not guarantee bitwise replay; process and contact-solver numerical sensitivity were not isolated by this experiment.',
        'Goal rates include every terminal outcome and are not shot accuracy; own goals and non-kick contacts remain allowed.',
        'All-step guard counters may include further episodes in worlds that already finished their scored first episode.',
        'No self-play or learned second policy; no claim of statistical significance or additive causal contribution percentages.'])
save(OUT/'comparison_summary.json',report)
with (OUT/'all_512_episodes.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
for name in ('duel_matched_helper.py','test_duel_matched_helper.py','eval_duel_matched_helper.py','summarize_duel_matched_helper.py'):
    shutil.copyfile(ROOT/'work'/name,OUT/name)
labels={'P_H_vs_A':'PPO＋控制器 對 原版A','P_H_vs_R_H':'PPO＋控制器 對 規則＋同控制器',
    'R_H_vs_A':'規則＋控制器 對 原版A','R_H_vs_R_H':'兩邊都規則＋同控制器'}
lines=['雙鴨：相同到位停穩控制器的2×2比較','',
    '這次沒有重新訓練、沒有修改PPO／技能／物理／原正式600局。512局均為推論，每組128局。',
    'P＝凍結的v3 seed1 PPO決策；R＝朝準備位置的幾何比例控制，持續提出出腳意圖；H＝原FastStopController。',
    'R＋H與P＋H共用同一個控制器類別與設定、位置／角度／身體／球速度guard、0.2秒ready hold及停穩流程。',
    'R＋H是新的規則基準，不是原版A不加修改的同義詞。','',
    '組別｜藍色進球／128｜橘色進球／128｜球出界｜超時｜藍／橘曾出腳局數']
for c in conditions:
    r=results[c];counts=r['total']['end_counts']
    lines.append(f"{labels[c]}｜{counts['goal_for']}（{r['total']['goal_for_rate']:.1%}）｜{counts['goal_against']}｜{counts['ball_out']}｜{counts['timeout']}｜{r['blue_kick_rounds']}／{r['orange_kick_rounds']}")
lines+=['',
    '本輪初步觀察：PPO＋H對R＋H為23局進球、1局失球；R＋H互打為11局進球、13局失球。這支持固定PPO的決策在本輪相同控制器條件下仍有優勢的跡象。',
    '另一方面，R＋H對舊A也有38局進球、6局失球，顯示只加控制流程就能對舊A取得明顯進球領先，先前優勢不能全部歸功PPO。',
    '同控制器的PPO對R＋H仍有99/128局球出界，還沒有穩定完成對戰；單一seed、小子集不能主張統計顯著或普遍優越。',
    '這輪新跑的PPO對舊A為44/128進球、3局失球；先前384局探測的36/128保留為另一輪資料，不是重新訓練改善，也不混算。',
    '讀法：第二組看相同控制器下，固定PPO對這個規則決策的對抗表現；第三組看單靠規則加控制器對舊A能做到多少；第四組作規則對規則參考。',
    '不能把這些對戰分數直接相減成「幾%來自AI、幾%來自控制器」：對手互動、初始狀態與訓練對手分布都會影響結果。',
    'PPO原本只對舊A訓練，現在換成R＋H同時是在測分布改變；目前只有一個PPO training seed。',
    '使用原固定場景0–31、各2次、兩個方向，每方向64個平行環境；RNG seed0、20秒。所有出界／跌倒／超時都留在128局分母。',
    '和原600局（每方向300環境）及先前384局分開，分數只比較本輪新跑的四組。',
    '每局進球率不是射門準確率；沿用原規則，進球不要求由某次踢球技能直接造成。','',
    '驗證：7項CPU檢查通過；8個worker returncode0；512局完整且唯一；actor/normalizer與checkpoint hash前後未變，訓練來源乾淨。',
    '同控制器橘色實際出腳時已逐次檢查原guard及零走路指令，違規次數均0；全模擬的guard計數可能包含已評完world的後續回合，不當成正式出腳局數。',
    'checkpoint SHA256：'+status['checkpoint_sha256'],
    '原始資料：all_512_episodes.csv、各方向episodes.json／summary／manifest／日誌，完整設定和限制見comparison_summary.json。',
    '本輪結果先保存本機，沒有自動上傳GitHub。']
(OUT/'公平控制器比較結果.txt').write_bytes(('\n'.join(lines)+'\n').encode())
save(OUT/'file_manifest.json',dict(files={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_manifest.json'}))
print(json.dumps({c:dict(goals=r['total']['end_counts']['goal_for'],against=r['total']['end_counts']['goal_against'],ball_out=r['total']['end_counts']['ball_out'],timeout=r['total']['end_counts']['timeout'],blue_kick_rounds=r['blue_kick_rounds'],orange_kick_rounds=r['orange_kick_rounds']) for c,r in results.items()},indent=2))
