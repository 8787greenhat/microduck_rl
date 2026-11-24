"""Audit completed robustness probe. Standard library only, no inference/training."""
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/duel_opponent_probe_20261124'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_bytes((json.dumps(v,indent=2,ensure_ascii=False)+'\n').encode())
state=read(OUT/'status.json')
assert state['state']=='complete' and len(state['workers'])==6 and all(w['returncode']==0 for w in state['workers'])
spec=importlib.util.spec_from_file_location('fixed_protocol',ROOT/'work/microduck_duel_v3/src/mjlab_microduck/duel/protocol.py')
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
variants=('original_A','short_dwell_A','direct_chase_A')
results={};all_rows=[]
expected=set(itertools.product(range(32),range(2),range(2)))
for variant in variants:
    rows=[];manifests=[]
    for side in (0,1):
        folder=OUT/f'{variant}_side{side}'
        summary=read(folder/'summary.json');manifest=read(folder/'manifest.json');items=read(folder/'episodes.json')
        assert summary['complete'] and summary['actor_and_normalizer_unchanged'] and not summary['new_training']
        assert sha(folder/'per_episode.csv')==summary['per_episode_csv_sha256']
        assert protocol.summarize(items)==summary['total']
        assert manifest['source']['source_commit']=='e8b9756c03ae66a693275ee84b61a36aab367b2f'
        assert not manifest['source']['source_dirty']
        assert manifest['checkpoint_sha256']=='f50f7fade3a12abea06f12acdfc904d2029a57238ed864aaa940e73a1df9cfc2'
        rows+=items;manifests.append(manifest)
    assert len(rows)==128 and {(r['scene_id'],r['repeat'],r['learner_side']) for r in rows}==expected
    total=protocol.summarize(rows)
    results[variant]=dict(description=manifests[0]['opponent']['description'],total=total,
        learner_kick_rounds=sum(r['learner_kick_attempts']>0 for r in rows),opponent_kick_rounds=sum(r['opponent_kick_attempts']>0 for r in rows),
        by_side={str(side):protocol.summarize([r for r in rows if r['learner_side']==side]) for side in (0,1)})
    all_rows+=rows
for side in (0,1):
    reference=read(OUT/f'original_A_side{side}/manifest.json')
    for variant in variants[1:]:
        other=read(OUT/f'{variant}_side{side}/manifest.json')
        for key in ('source','checkpoint_sha256','learner_side','num_envs','scene_ids','repeats','eval_rng_seed','round_seconds','original_protocol'):
            assert reference[key]==other[key],key
rows_by_variant={v:{(r['scene_id'],r['repeat'],r['learner_side']):r for r in all_rows if r['opponent_variant']==v} for v in variants}
comparison={}
for variant in variants[1:]:
    pairs=[(rows_by_variant['original_A'][key],rows_by_variant[variant][key]) for key in sorted(expected)]
    comparison[variant]=dict(goal_for_rate_difference_percentage_points=(results[variant]['total']['goal_for_rate']-results['original_A']['total']['goal_for_rate'])*100,
        previously_goal_now_not=sum(a['end_reason']=='goal_for' and b['end_reason']!='goal_for' for a,b in pairs),
        previously_not_goal_now_goal=sum(a['end_reason']!='goal_for' and b['end_reason']=='goal_for' for a,b in pairs),
        caution='Paired nominal initial layouts; numerical/contact sensitivity and reset-level RNG may affect outcomes. Not a significance test.')
report=dict(complete=True,official_fixed600=False,training_seed=1,new_training=False,training_method='frozen PPO v3; no curriculum/self-play',
    total_episodes=384,episodes_per_opponent=128,num_envs_per_side=64,scene_ids=list(range(32)),repeats=[0,1],sides=[0,1],eval_rng_seed=0,
    checkpoint_sha256=state['checkpoint_sha256'],cpu_wrapper_checks_passed=5,all_worker_returncodes_zero=True,
    all_actor_and_normalizer_unchanged=True,original_training_sources_clean=True,results=results,paired_nominal_layout_comparison=comparison,
    limitations=['Diagnostic subset and batch64, not the historical official600 batch300; compare to the new original_A condition only.',
        'One frozen training seed; rule variants are not proven stronger opponents, learned opponents or self-play.',
        'Learner uses the trained v3 actor plus FastStopController; rules use geometric preparation control. This compares complete control systems, not PPO alone.',
        'No generalization claim to arbitrary opponents, unseen layouts, moving-ball starts or real hardware.',
        'Repeated scene trials are not independent training seeds; no significance claim.'])
save(OUT/'comparison_summary.json',report)
with (OUT/'all_384_episodes.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
for name in ('duel_opponent_variants.py','eval_duel_opponent_variants.py','test_duel_opponent_variants.py','summarize_duel_opponent_probe.py'):
    shutil.copyfile(ROOT/'work'/name,OUT/name)
labels={'original_A':'原版規則A','short_dwell_A':'縮短停穩等待A（0.5→0.2秒）','direct_chase_A':'遠處直接追球A'}
lines=['雙鴨 v3 seed 1：更換規則對手的探索性測試','',
    '同一最終checkpoint，不訓練、不改PPO／獎勵／官方技能；對手只有固定規則變體，並非兩個學習模型互踢。',
    '原場景0–31，各2次、旋轉換邊，每個對手128局，共384局；eval RNG seed0、20秒上限。每方向64個平行環境。',
    '這是新的小規模診斷，與原正式600局分開；原167/600保留，不把新原版A分數直接與舊600局比較。','',
    '對手｜我方進球／128｜對手進球／128｜球出界｜超時｜我方／對手曾出腳局數']
for variant in variants:
    result=results[variant];counts=result['total']['end_counts']
    lines.append(f"{labels[variant]}｜{counts['goal_for']} ({result['total']['goal_for_rate']:.1%})｜{counts['goal_against']}｜{counts['ball_out']}｜{counts['timeout']}｜{result['learner_kick_rounds']}／{result['opponent_kick_rounds']}")
lines += ['',
    '本輪結論：縮短對手等待時間後，進球36→37、失球仍2，優勢大致保留；換成直接追球，進球36→22、失球2→8，對手風格確實會影響這輪觀察。',
    '直接追球條件下藍色仍進球22局、失球8局，但96/128球出界；不能把進球領先說成已學會穩定對抗或任意對手都能贏。',
    '如何解讀：看藍色對每一種規則對手的進球／失球與出界，而不是只看剪出的幾段影片。',
    '這兩種新規則尚未經獨立校準，不能先稱為更強；如果表現變差，也可能是對手風格、物理接觸與策略適應共同造成。',
    '目前只能回答對這兩種規則變體的初步表現，不能說換任何對手都強，也尚未測PPO對PPO。',
    '藍色使用PPO加上v3到位停穩控制器；橘色是幾何規則控制。這是整套控制系統比較，優勢不能全歸因PPO演算法。',
    '出界、跌倒、超時均留在128局分母；進球時間只計成功局，各條件成功場景可能不同。','',
    '驗證：5項隔離wrapper CPU檢查通過，6個評測worker returncode0，384局完整且無重複，來源程式乾淨；actor/normalizer與checkpoint SHA256前後未變。',
    '模型SHA256：'+report['checkpoint_sha256'],
    '數據：comparison_summary.json、all_384_episodes.csv，以及每個對手／方向的summary、manifest、逐局JSON與原始日誌。',
    '資料先保留本機，本次沒有推送GitHub。']
(OUT/'結果說明.txt').write_bytes(('\n'.join(lines)+'\n').encode())
save(OUT/'file_manifest.json',dict(files={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_manifest.json'}))
print(json.dumps({v:dict(goals=results[v]['total']['end_counts']['goal_for'],against=results[v]['total']['end_counts']['goal_against'],ball_out=results[v]['total']['end_counts']['ball_out'],timeout=results[v]['total']['end_counts']['timeout'],learner_kick_rounds=results[v]['learner_kick_rounds'],opponent_kick_rounds=results[v]['opponent_kick_rounds']) for v in variants},indent=2))
