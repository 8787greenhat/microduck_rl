"""Render recorded simulator states with mjlab OffscreenRenderer; no resimulation."""
import os
os.environ['MUJOCO_GL']='egl'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,default=str)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    import numpy as np
    import torch
    import mujoco
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig
    torch.set_num_threads(1)
    manifest=json.loads((a.capture/'capture_manifest.json').read_text());assert manifest['complete']
    a.out.mkdir(exist_ok=False)
    model=mujoco.MjModel.from_binary_path(str(a.capture/'render_model.mjb'))
    cfg=ViewerConfig(width=768,height=512,max_extra_envs=0,enable_shadows=False,enable_reflections=False,
        origin_type=ViewerConfig.OriginType.WORLD,lookat=(0,0,.1),distance=2.65,elevation=-52,azimuth=90)
    renderer=OffscreenRenderer(model,cfg,None);renderer.initialize()
    from OpenGL import GL
    renderer_name=GL.glGetString(GL.GL_RENDERER).decode()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    records=[];began=time.monotonic()
    try:
        for clip in manifest['clips'][:1] if a.preview_only else manifest['clips']:
            data=np.load(a.capture/clip['states_file']);assert sha(a.capture/clip['states_file'])==clip['states_sha256']
            renderer._cam.lookat[:]=np.array(clip['origin_xyz'])+np.array([0,0,.1])
            scene=clip['scene_id'];path=a.out/f'scene_{scene:03d}.mp4';latest=None;written=0
            def frame(index,terminal=False):
                tensors={k:torch.from_numpy(data[k][index:index+1].copy()) for k in ('qpos','qvel','mocap_pos','mocap_quat')}
                renderer.update(SimpleNamespace(nworld=1,**tensors))
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,768,72),fill=(13,21,32))
                draw.text((12,5),f'Duel v3 seed 1 | Scene {scene:03d} repeat {clip["repeat"]} | t={data["t"][index]:.2f}s',font=font,fill='white')
                draw.text((12,29),'BLUE: PPO v3 -> right goal     ORANGE: fixed A -> left goal',font=small,fill=(200,220,240))
                line=f'END: {clip["end_reason"]} | kick-skill starts: blue {clip["learner_kick_attempts"]}, orange {clip["opponent_kick_attempts"]}' if terminal else 'Independent diagnostic rollout | outcomes may differ from original evaluation'
                draw.text((12,50),line,font=small,fill=(255,215,120) if terminal else (210,220,230))
                return image
            if a.preview_only:
                frame(0).save(a.out/'preview_start.png')
                frame(len(data['t'])-1,True).save(a.out/'preview_end.png')
                save(a.out/'preview_info.json',{'renderer':renderer_name,'camera':asdict(cfg),'scene':clip})
                return
            indices=list(range(0,len(data['t']),2))
            if indices[-1]!=len(data['t'])-1:indices.append(len(data['t'])-1)
            with imageio.get_writer(str(path),fps=25,codec='libx264',quality=8,macro_block_size=16) as writer:
                for position,index in enumerate(indices):
                    image=frame(index,position==len(indices)-1);array=np.asarray(image)
                    writer.append_data(array);written+=1;latest=array
                    if position==0:image.save(a.out/f'scene_{scene:03d}_start.png')
                    if position==len(indices)-1:image.save(a.out/f'scene_{scene:03d}_end.png')
                    if position%50==0:
                        save(a.out/'render_progress.json',{'state':'rendering','scene_id':scene,'frames':written,'total_dynamic_frames':len(indices),'wall_seconds':time.monotonic()-began})
                        print(f'render scene {scene} frame {written}/{len(indices)}',flush=True)
                for _ in range(25):writer.append_data(latest);written+=1
            reader=imageio.get_reader(str(path));meta=reader.get_meta_data();decoded=reader.count_frames();reader.get_data(decoded-1);reader.close()
            assert decoded==written and abs(meta['duration']-written/25)<.05
            records.append({**clip,'video':path.name,'frames':written,'decoded_duration_s':meta['duration'],
                            'fps':25,'terminal_hold_s':1,'sha256':sha(path)})
            save(a.out/'clips.json',records)
        concat=a.out/'concat.txt'
        concat.write_text(''.join(f"file '{(a.out/r['video']).as_posix()}'\n" for r in records))
        combined=a.out/'duel_v3_seed1_diagnostics.mp4'
        result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart','-n',str(combined)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        reader=imageio.get_reader(str(combined));meta=reader.get_meta_data();count=reader.count_frames();reader.get_data(count-1);reader.close()
        assert count==sum(r['frames'] for r in records)
        thumb=Image.new('RGB',(768*2,512*2))
        for i,r in enumerate(records):thumb.paste(Image.open(a.out/f'scene_{r["scene_id"]:03d}_end.png'),((i%2)*768,(i//2)*512))
        thumb.save(a.out/'four_scenes_preview.png')
        save(a.out/'video_manifest.json',{'complete':True,'source_commit':manifest['source_commit'],
            'checkpoint_sha256':manifest['checkpoint_sha256'],'selection':manifest['selection'],'clips':records,
            'combined_video':combined.name,'combined_sha256':sha(combined),'decoded_frames':count,
            'decoded_duration_s':meta['duration'],'normal_speed':True,'renderer':renderer_name,
            'rendering':'Unchanged mjlab OffscreenRenderer on saved qpos/qvel/mocap states; full mj_forward; shadows/reflections/MSAA off',
            'new_simulation_or_training_during_rendering':False,'independent_rollout_not_original_official_trajectory':True,
            'outcome_mismatches_vs_official':manifest['all300_outcome_and_time_mismatches_vs_official'],'wall_seconds':time.monotonic()-began})
        print(json.dumps({'complete':True,'combined_video':str(combined),'decoded_frames':count,'duration_s':meta['duration']},indent=2))
    finally:renderer.close()

if __name__=='__main__':main()
