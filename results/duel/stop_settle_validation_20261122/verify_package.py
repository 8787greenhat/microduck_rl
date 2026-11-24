"""Verify the checkpoint, source contract and saved short tests without GPU work."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest=read(ROOT/'file_manifest.json')
    for row in manifest['files']:
        path=ROOT/row['path']
        assert path.resolve().is_relative_to(ROOT),row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
    publication=read(ROOT/'publication_manifest.json')
    for name,digest in publication['candidate_source_sha256'].items():
        # Git may check out source text with CRLF on Windows. Compare canonical LF.
        actual=hashlib.sha256((REPO/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        assert actual==digest,name
    assert sha(ROOT/'inputs/model_1999.pt')==publication['checkpoint_sha256']
    verification=read(ROOT/'verification.json')
    assert verification['CPU_tests_passed']==20 and verification['PPO_optimizer_steps']==0
    for expected in verification['rows']:
        result=read(ROOT/(expected['case']+'.json'))
        assert result['complete'] and result['optimizer_steps']==0 and not result['training']
        assert result['checkpoint_sha256']==publication['checkpoint_sha256']
        assert result['actor_and_normalizer_unchanged'] and result['original_v2_readiness_gate_not_bypassed']
        episodes=result['episodes'];assert len(episodes)==64
        strict=sum(r['end_reason']=='goal_for' and r['kick_count']>0
                   and r['rightfoot_contact_samples_after_first_kick']>0
                   and r['last_touch']==result['side'] for r in episodes)
        assert strict==expected['goals_after_kick_and_rightfoot_contact']==result['goals_after_kick_and_rightfoot_contact']
        assert sum(r['end_reason']=='goal_for' for r in episodes)==expected['all_goal_rounds']
        assert result['kick_skill_rounds']==expected['kick_skill_rounds']
        for event in result.get('kick_events',[]):
            assert event['original_ready_guard_passed']
            assert event['distance_m']<.05 and event['abs_heading_rad']<.15
            assert event['base_speed_mps']<.05 and event['ball_speed_mps']<.15
    print(json.dumps({'verified':True,'short_test_batches':5,'episodes':320,
                      'checkpoint_sha256':publication['checkpoint_sha256'],'GPU_used':False,'training':False},indent=2))

if __name__=='__main__':main()
