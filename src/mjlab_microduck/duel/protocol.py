"""Immutable duel-v1 layout generation, role rotation and metric definitions (CPU)."""
import hashlib
import json
import math
from pathlib import Path
import random

PROTOCOL_ID='microduck-duel-fixed-v1'
GENERATOR_SEED=20261121
EVAL_RNG_SEED=0
REPEATS=3
ROUND_SECONDS=20
MODEL_HASHES={'alpha_walking.onnx':'e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c',
 'ball_kick_right.onnx':'147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af'}
END_NAMES={1:'goal_for',2:'goal_against',3:'ball_out',4:'learner_fell',5:'opponent_fell',6:'learner_oob',7:'opponent_oob',8:'timeout',9:'both_fell',10:'both_oob'}
FLAG_NAMES=('goal_for','goal_against','ball_out','learner_fell','opponent_fell','learner_oob','opponent_oob','timeout')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def wrap(angle):return (angle+math.pi)%(2*math.pi)-math.pi
def generate_layouts():
    rng=random.Random(GENERATOR_SEED);rows=[]
    for i in range(100):
        ball=[rng.uniform(-.3,.3),rng.uniform(-.25,.25)]
        poses=[]
        for side in (1,-1):
            dx,dy=side-ball[0],-ball[1];norm=math.hypot(dx,dy);ux,uy=dx/norm,dy/norm
            angle=math.atan2(uy,ux)
            # Right-foot offset is rotated, never reflected into a left-foot skill.
            px=ball[0]-(ux*.09-uy*(-.042));py=ball[1]-(uy*.09+ux*(-.042))
            distance=rng.uniform(.25,.45);lateral=rng.uniform(-.1,.1)
            poses.append([px-distance*ux-lateral*uy,py-distance*uy+lateral*ux,wrap(angle+rng.uniform(-math.pi/4,math.pi/4))])
        rows.append({'scene_id':i,'learner_xy_yaw':[round(x,10) for x in poses[0]],'opponent_xy_yaw':[round(x,10) for x in poses[1]],'ball_xy':[round(x,10) for x in ball]})
    return {'schema_version':1,'protocol_id':PROTOCOL_ID,'generator_seed':GENERATOR_SEED,'layouts':rows}

def role_layout(row,learner_index):
    if learner_index not in (0,1):raise ValueError('learner_index must be 0 or 1')
    def rotate(pose):return [-pose[0],-pose[1],wrap(pose[2]+math.pi)]
    if learner_index==0:return {'robots':[row['learner_xy_yaw'],row['opponent_xy_yaw']],'ball':row['ball_xy']}
    return {'robots':[rotate(row['opponent_xy_yaw']),rotate(row['learner_xy_yaw'])],'ball':[-x for x in row['ball_xy']]}

def write_protocol(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    layouts=folder/'layouts.json';manifest=folder/'manifest.json'
    text=json.dumps(generate_layouts(),indent=2,ensure_ascii=False)+'\n'
    if layouts.exists() and layouts.read_text(encoding='utf-8')!=text:raise ValueError('Existing fixed layouts differ; create a new protocol version')
    layouts.write_text(text,encoding='utf-8')
    value={'protocol_id':PROTOCOL_ID,'layouts_sha256':sha(layouts),'layouts':100,'repeats':REPEATS,'sides':[0,1],
        'episodes_total':600,'batch_per_side':300,'evaluation_rng_seed':EVAL_RNG_SEED,'generator_seed':GENERATOR_SEED,
        'round_seconds':ROUND_SECONDS,'actor_inference':'deterministic mean; normalization frozen',
        'nominal_physics':True,'observation_noise':False,'stationary_ball_starts':True,'auto_reset_scoring':'first episode per world only',
        'goal_rule':'whole ball crosses outward plane and fits clear opening; own goals allowed; no foot-contact gate',
        'end_priority':['goal_for','goal_against','both_fell','learner_fell','opponent_fell','both_oob','learner_oob','opponent_oob','ball_out','timeout'],
        'outcome_denominator':'all 600 episodes, including aborted/non-goal rounds','official_skill_sha256':MODEL_HASHES,
        'single_robot_BC_benchmark_changed':False}
    content=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if manifest.exists() and manifest.read_text(encoding='utf-8')!=content:raise ValueError('Existing protocol differs; version it explicitly')
    manifest.write_text(content,encoding='utf-8');return value

def load_protocol(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    data=json.loads((folder/'layouts.json').read_text(encoding='utf-8'))
    if manifest['protocol_id']!=PROTOCOL_ID or manifest['layouts_sha256']!=sha(folder/'layouts.json'):raise ValueError('Protocol/layout hash mismatch')
    if len(data['layouts'])!=100 or [r['scene_id'] for r in data['layouts']]!=list(range(100)):raise ValueError('Expected the 100 fixed scene ids')
    return manifest,data['layouts']

def summarize(rows):
    import statistics
    counts={name:sum(r['end_reason']==name for r in rows) for name in END_NAMES.values()}
    n=len(rows);assert sum(counts.values())==n
    wins=[r['elapsed_s'] for r in rows if r['end_reason']=='goal_for']
    def quantile(values,p):
        if not values:return None
        s=sorted(values);x=(len(s)-1)*p;i=int(x);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(x-i)
    return {'episodes':n,'end_counts':counts,'goal_for_rate':counts['goal_for']/n,'goal_against_rate':counts['goal_against']/n,
        'timeout_rate':counts['timeout']/n,'aborted_rate':sum(v for k,v in counts.items() if k not in ('goal_for','goal_against','timeout'))/n,
        'goal_difference_per_episode':(counts['goal_for']-counts['goal_against'])/n,
        'winning_time_s':{'mean':statistics.mean(wins) if wins else None,'median':quantile(wins,.5),'p90':quantile(wins,.9)},
        'end_time_mean_s':statistics.mean(r['elapsed_s'] for r in rows),
        'contact_attribution_unknown_goal_rounds':sum(r['last_touch']<0 and r['end_reason'] in ('goal_for','goal_against') for r in rows)}
