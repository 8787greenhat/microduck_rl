"""Independent inference rollout for first four fixed layouts; no optimization."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2')
RUN=ROOT/'outputs/duel_v3_PPO_vs_A_seed1_20261124_091746'
REPO=Path('/home/lingchentseng/projects/microduck_duel_v3_seed1_e8b9756')
PIN='e8b9756c03ae66a693275ee84b61a36aab367b2f'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    assert read(RUN/'status.json')['state']=='complete'
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    checkpoint=RUN/'training/checkpoints/model_1999.pt'
    assert sha(checkpoint)==read(RUN/'training/verification.json')['checkpoint_sha256']
    a.out.mkdir(exist_ok=False)
    from mjlab_microduck.duel.runtime import initialize
    from mjlab_microduck.duel.protocol import load_protocol,role_layout,END_NAMES,EVAL_RNG_SEED
    initialize(EVAL_RNG_SEED)
    import numpy as np
    import torch
    import mujoco
    from mjlab_microduck.duel.v3_env import DuelTrainV3
    from mjlab_microduck.duel.v3_runner import DuelV3OnPolicyRunner,MicroduckDuelV3RlCfg
    protocol,layouts=load_protocol(REPO/'benchmarks/duel/v1')
    assignments=[(row,repeat) for row in layouts for repeat in range(protocol['repeats'])]
    env=DuelTrainV3(len(assignments),EVAL_RNG_SEED,learner_index=0)
    selection=read(ROOT/'outputs/duel_followup_20261124/diagnostic_selection.json')
    selected={case['scene_id']*3+case['repeat']:case for case in selection['cases']}
    states={index:[] for index in selected}
    outcomes={};actual_rows=[];low_step=0;began=time.monotonic()
    try:
        data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg=asdict(MicroduckDuelV3RlCfg);cfg.update(logger='tensorboard',obs_groups={'actor':['actor'],'critic':['critic']})
        runner=DuelV3OnPolicyRunner(env,cfg,None,'cuda:0')
        runner.alg.actor.load_state_dict(data['actor_state_dict'],strict=True)
        actor_before={k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
        policy=runner.get_inference_policy(device='cuda:0')
        placed=[role_layout(row,0) for row,_ in assignments]
        env.env._duel_fixed_layout_override={'robots':torch.tensor([x['robots'] for x in placed],device=env.device),
                                            'ball':torch.tensor([x['ball'] for x in placed],device=env.device)}
        torch.manual_seed(EVAL_RNG_SEED);obs=env.reset()
        origins=env.env.scene.env_origins.detach().cpu().tolist()
        # Render-only setting; all simulation and controller settings stay intact.
        env.env.sim.mj_model.vis.quality.offsamples=0
        mujoco.mj_saveModel(env.env.sim.mj_model,str(a.out/'render_model.mjb'))
        def snapshot(index):
            value=env.env.sim.data
            return {'t':low_step*.02,'qpos':value.qpos[index].detach().cpu().numpy().copy(),
                'qvel':value.qvel[index].detach().cpu().numpy().copy(),
                'mocap_pos':value.mocap_pos[index].detach().cpu().numpy().copy(),
                'mocap_quat':value.mocap_quat[index].detach().cpu().numpy().copy(),
                'kicking':env.skills.mode[index].detach().cpu().numpy().copy(),
                'kick_counts':env.skills.kick_count[index].detach().cpu().numpy().copy(),
                'contact_counts':env.contact.count[index].detach().cpu().numpy().copy(),
                'last_touch':env.contact.last_touch[index].detach().cpu().numpy().copy(),
                'ball_velocity':env.env.scene['ball'].data.root_link_lin_vel_w[index].detach().cpu().numpy().copy(),
                'ball_xyz':env.env.scene['ball'].data.root_link_pos_w[index].detach().cpu().numpy().copy()}
        for index in selected:states[index].append(snapshot(index))
        original_step=env.env.step
        def traced_step(actions):
            nonlocal low_step
            result=original_step(actions);low_step+=1
            for index in selected:
                if index not in outcomes:states[index].append(snapshot(index))
            return result
        env.env.step=traced_step
        finished=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        with torch.inference_mode():
            for step in range(200):
                obs,_,done,extra=env.step(policy(obs));fresh=done.bool()&~finished;stats=extra['duel_episode']
                for index in fresh.nonzero(as_tuple=False).flatten().tolist():
                    scene,repeat=assignments[index]
                    record={'scene_id':scene['scene_id'],'repeat':repeat,'learner_side':0,
                        'end_reason':END_NAMES[int(stats['code'][index])],
                        'elapsed_s':round(float(stats['elapsed_s'][index]),5),
                        'last_touch':int(stats['last_touch'][index]),
                        'learner_kick_attempts':int(stats['kick_count'][index,0]),
                        'opponent_kick_attempts':int(stats['kick_count'][index,1])}
                    actual_rows.append(record)
                    if index in selected:
                        outcomes[index]={**record,'layout':scene,'env_index':index,'origin_xyz':origins[index]}
                finished|=fresh
                if step%20==0 or finished.all():
                    print(f'capture step {step+1}: {int(finished.sum())}/300 finished',flush=True)
                if finished.all():break
        assert finished.all() and len(outcomes)==4
        assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in actor_before.items())
        original=read(RUN/'fixed600/side_0/side_0_episodes.json')
        expected={(r['scene_id'],r['repeat']):r for r in original}
        mismatches=[]
        for record in actual_rows:
            prior=expected[record['scene_id'],record['repeat']]
            if any(record[k]!=prior[k] for k in ('end_reason','elapsed_s','last_touch','learner_kick_attempts','opponent_kick_attempts')):
                mismatches.append({'key':[record['scene_id'],record['repeat']],'actual':record,
                                   'official':{k:prior[k] for k in record if k in prior}})
        records=[]
        for index,case in selected.items():
            scene_id=case['scene_id']
            record={**outcomes[index],'selection_category':case['label'],'selection_official_row':case['official_row']}
            frames=[s for s in states[index] if s['t']<=record['elapsed_s']+1e-4]
            assert abs(frames[-1]['t']-record['elapsed_s'])<1e-4
            path=a.out/f'scene_{scene_id:03d}_states.npz'
            np.savez_compressed(path,**{k:np.stack([s[k] for s in frames]) for k in frames[0]})
            record.update(states_file=path.name,states_sha256=sha(path),low_level_samples=len(frames),
                          independent_visualization_rollout=True,matched_official_outcome=(record['end_reason']==expected[scene_id,case['repeat']]['end_reason']))
            records.append(record)
        assert sha(checkpoint)==read(RUN/'training/verification.json')['checkpoint_sha256']
        save(a.out/'capture_manifest.json',{'complete':True,'source_commit':PIN,'checkpoint_sha256':sha(checkpoint),
            'protocol_layouts_sha256':protocol['layouts_sha256'],'training':False,'optimizer_used':False,
            'actor_unchanged':True,'selection':selection['selection_rule'],'selected_cases':selection['cases'],
            'full_parallel_batch':300,'eval_rng_seed':0,'capture_hz':50,
            'independent_rollout_not_original_official_trajectory':True,
            'all300_outcome_and_time_mismatches_vs_official':mismatches,'clips':records,'wall_seconds':time.monotonic()-began})
        print(json.dumps({'capture_complete':True,'mismatches_vs_official':len(mismatches),'clips':records},indent=2),flush=True)
    finally:env.close()

if __name__=='__main__':main()
