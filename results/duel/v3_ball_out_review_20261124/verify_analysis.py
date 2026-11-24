"""Standard-library verification of bytes and original 600-episode stratification."""
import collections
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files=read(ROOT/'file_manifest.json')['files']
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='file_manifest.json' and '__pycache__' not in p.parts}
    assert actual==set(files)
    for name,entry in files.items():
        p=ROOT/name
        assert p.resolve().is_relative_to(ROOT) and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
    source=ROOT.parent/'v3_seed1_20261124/evaluation/fixed600/per_episode.csv'
    analysis=read(ROOT/'ball_out_analysis.json')
    assert sha(source)==analysis['per_episode_csv_sha256']
    with source.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==600
    outs=[r for r in rows if r['end_reason']=='ball_out']
    touch=lambda r:'learner' if r['last_touch']==r['learner_side'] else 'opponent' if r['last_touch'] in ('0','1') else 'simultaneous' if r['last_touch']=='-2' else 'unknown'
    assert len(outs)==analysis['ball_out_as_primary_end_reason']==402
    assert collections.Counter(touch(r) for r in outs)==analysis['ball_out_last_observed_touch']
    assert sum(r['flag_ball_out']=='True' for r in rows)==analysis['raw_ball_out_flags']==576
    for group in analysis['strata']:
        assert sum((int(r['learner_kick_attempts'])>0)==group['learner_kick_skill_in_episode'] and touch(r)==group['last_observed_touch'] for r in outs)==group['episodes']
    review=read(ROOT/'diagnostic_trace_review.json')
    assert review['independent_rollout_not_original_official_trajectory']
    assert review['records_with_changed_end_reason']==65
    video=read(ROOT/'videos/video_manifest.json')
    assert video['complete'] and video['decoded_frames']==587 and abs(video['decoded_duration_s']-23.48)<.01
    assert sha(ROOT/'videos'/video['combined_video'])==video['combined_sha256']
    print(json.dumps(dict(verified=True,files=len(files)+1,original_episodes=600,ball_out=402,independent_video_seconds=23.48,new_training=False)))
if __name__=='__main__':main()
