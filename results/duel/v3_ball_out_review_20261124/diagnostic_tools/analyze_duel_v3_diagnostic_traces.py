"""Describe saved independent rollout traces. CPU only, not an official re-evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path('/mnt/c/Users/LingChengTseng/Documents/Codex/2026-10-24/mac-microduck-rl-1-00-start-2/outputs/duel_followup_20261124')
manifest=json.loads((ROOT/'capture/capture_manifest.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
records=[]
for ax,clip in zip(axes.flat,manifest['clips']):
    data=np.load(ROOT/'capture'/clip['states_file'])
    ball=data['ball_xyz']-np.array(clip['origin_xyz'])
    t=data['t'];contacts=data['contact_counts'];kicks=data['kick_counts']
    times=lambda mask:[round(float(v),4) for v in t[np.asarray(mask)]]
    starts=np.r_[False,np.diff(kicks[:,0])>0]
    record=dict(scene_id=clip['scene_id'],repeat=clip['repeat'],official_end_reason=clip['selection_official_row']['end_reason'],
        diagnostic_end_reason=clip['end_reason'],terminal_time_s=clip['elapsed_s'],learner_kick_times_s=times(starts),
        terminal_ball_xyz=ball[-1].tolist(),terminal_ball_velocity=data['ball_velocity'][-1].tolist(),
        last_observed_touch=int(data['last_touch'][-1]),
        learner_contact_times_s=times(contacts[:,0]>0),opponent_contact_times_s=times(contacts[:,1]>0),
        learner_rightfoot_contact_times_s=times(contacts[:,2]>0),opponent_rightfoot_contact_times_s=times(contacts[:,3]>0),
        exit_boundary='positive_x' if ball[-1,0]>1.035 else 'negative_x' if ball[-1,0]<-1.035 else 'positive_y' if ball[-1,1]>.635 else 'negative_y' if ball[-1,1]<-.635 else 'none',
        recording_resolution='50 Hz snapshots with within-step 200 Hz contact sample counts; not exact contact impulses',
        original_trajectory_replayed=False)
    records.append(record)
    ax.add_patch(Rectangle((-1,-.6),2,1.2,facecolor='#e9f3e7',edgecolor='#333333'))
    for x in (-1,1):ax.plot([x,x],[-.2,.2],lw=5,color='#777777')
    points=ax.scatter(ball[:,0],ball[:,1],c=t,s=9,cmap='viridis',zorder=2)
    ax.plot(ball[:,0],ball[:,1],color='#999999',lw=.6,zorder=1)
    ax.scatter(ball[0,0],ball[0,1],s=55,marker='o',color='black',label='start',zorder=3)
    ax.scatter(ball[-1,0],ball[-1,1],s=70,marker='x',color='red',label='end',zorder=3)
    ax.scatter(ball[starts,0],ball[starts,1],s=100,marker='*',color='#ed8b23',edgecolor='black',linewidth=.5,label='kick skill starts',zorder=4)
    ax.set_title(f"Scene {clip['scene_id']} repeat {clip['repeat']}: {clip['end_reason']}\nOriginal: {record['official_end_reason']}",fontsize=10)
    ax.set(xlim=(-1.18,1.18),ylim=(-.8,.8),xlabel='x (m)',ylabel='y (m)',aspect='equal')
    fig.colorbar(points,ax=ax,label='time (s)',shrink=.7)
axes.flat[0].legend(fontsize=7,loc='upper left')
fig.suptitle('v3 seed 1: independent diagnostic ball paths (not original 600 trajectories)',fontsize=12)
fig.savefig(ROOT/'diagnostic_ball_paths.png',dpi=160)
plt.close(fig)
changed_outcome=sum(r['actual']['end_reason']!=r['official']['end_reason'] for r in manifest['all300_outcome_and_time_mismatches_vs_official'])
report=dict(training=False,optimizer_used=False,checkpoint_sha256=manifest['checkpoint_sha256'],
    independent_rollout_not_original_official_trajectory=True,
    batch_size=300,records_with_any_recorded_outcome_time_touch_or_kick_difference=len(manifest['all300_outcome_and_time_mismatches_vs_official']),
    records_with_changed_end_reason=changed_outcome,clips=records,
    caveat='Instrumentation/import/renderer preparation and numerical sensitivity can affect replay. These examples cannot retrospectively classify the 402 original ball-outs; exact root of divergence not isolated.')
(ROOT/'diagnostic_trace_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(any_record_difference=report['records_with_any_recorded_outcome_time_touch_or_kick_difference'],end_reason_differences=changed_outcome,
    clips=[{k:r[k] for k in ('scene_id','repeat','official_end_reason','diagnostic_end_reason','learner_kick_times_s','terminal_ball_xyz','exit_boundary')} for r in records]),indent=2))
