"""Verify that the study files in study/ are the exact files whose SHA-256 hashes were recorded
before each run. Files the hashes refer to but that are not redistributed (raw measurements that
contain landmark positions) are listed as not included.
Run: python check_hashes.py
"""
import hashlib, json, sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent / 'study'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ok = bad = missing = 0
for rec in sorted(STUDY.rglob('*.json')):
    name = rec.name.lower()
    if not ('freeze' in name or 'seal' in name):
        continue
    d = json.loads(rec.read_text())
    items = d['sha256'] if isinstance(d['sha256'], dict) else {d['file']: d['sha256']}
    when = d.get('frozen_utc') or d.get('sealed_utc')
    for rel, h in items.items():
        cands = [rec.parent / rel, STUDY / rel]
        path = next((c for c in cands if c.exists()), None)
        if path is None:
            missing += 1; status = 'not included (raw measurements)'
        elif digest(path) == h:
            ok += 1; status = 'ok'
        else:
            bad += 1; status = 'MISMATCH'
        print(f'{status:<34} {rec.relative_to(STUDY)} ({when}) -> {rel}')
print(f'\n{ok} hashes match, {bad} mismatch, {missing} refer to files not included.')
sys.exit(1 if bad else 0)
