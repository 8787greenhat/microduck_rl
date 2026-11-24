"""Verify the preserved peer packet and reviewed counts; no simulation or training."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    files=read(ROOT/'file_manifest.json')['files']
    for name,entry in files.items():
        p=ROOT/name
        assert p.resolve().is_relative_to(ROOT)
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
    checked=0
    for line in (ROOT/'received/SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest,name=line.split(maxsplit=1)
        p=ROOT/'received'/Path(*Path(name).parts[1:])
        assert p.resolve().is_relative_to(ROOT/'received') and sha(p)==digest
        checked+=1
    review=read(ROOT/'review.json')
    for case in review['tests']:
        data=read(ROOT/'received/results'/case['case']/'result.json')
        rows=data['episodes']
        assert data['complete'] and len(rows)==64 and data['optimizer_steps']==0
        assert sum(r['kick_count']>0 for r in rows)==case['peer']['kick_rounds']
        assert sum(r['end_reason']=='goal_for' for r in rows)==case['peer']['goals']
        assert sum(r['end_reason']=='goal_for' and r['kick_count']>0 and r['rightfoot_contact_samples_after_first_kick']>0 and r['last_touch']==data['side'] for r in rows)==case['peer']['kick_then_rightfoot_contact_goals']
    print(json.dumps(dict(verified=True,received_files=checked,short_test_episodes=192,new_training=False)))
if __name__=='__main__':main()
