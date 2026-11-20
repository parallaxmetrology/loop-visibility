"""Truth-blind measurement: all 45 phase-pair edges per eligible POPI landmark.

Imports the pilot's frozen instrument read-only. Never reads target landmarks
(only the phase-10 source points, which place the query centre).
"""
import sys, json, hashlib, time
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'dvc-holonomy-pilot'
sys.path.insert(0, str(PILOT))
sys.dont_write_bytecode = True
from correspondence.common import volume, landmarks  # noqa: E402
from interior.selection import envelope, interior, texture  # noqa: E402
from dvc_pilot.acceptance_run import normalized_patch  # noqa: E402
from dvc_pilot.registration import register  # noqa: E402
from dvc_pilot.refinement import refine_translation  # noqa: E402
from dvc_pilot.registration import registered_residual  # noqa: E402

PHASES = [f'{i:02}' for i in range(0, 100, 10)]
EDGES = list(combinations(range(10), 2))


def work(job):
    rec, charts = job
    edges = []
    for i, j in EDGES:
        r = register(charts[i], charts[j])
        e = dict(i=i, j=j, valid=r['valid'])
        if r['valid']:
            e.update(t=r['t'], residual=r['residual'], gradient_residual=r['gradient_residual'],
                     ambiguity=r['ambiguity'], valid_fraction=r['valid_fraction'])
            f = refine_translation(charts[i], charts[j], r['t'])
            e['refine_success'] = f['success']
            e['t_refined'] = f['t'] if f['success'] else r['t']
            e['residual_refined'] = registered_residual(charts[i], charts[j], e['t_refined'])[0]
        edges.append(e)
    rec['edges'] = edges
    return rec


def main():
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    dest = out / 'measurements.jsonl'
    if dest.exists():
        raise SystemExit('measurements exist; refusing to overwrite')
    vols, meta, masks, spacing = {}, {}, {}, None
    for p in PHASES:
        v, s, _ = volume(p)
        spacing = s if spacing is None else spacing
        assert np.array_equal(s, spacing)
        vols[p] = v
        masks[p], meta[p] = envelope(v)
    source = landmarks('10', spacing)
    jobs, excluded = [], []
    for k, pt in enumerate(source):
        c = np.rint(pt).astype(int)
        rec = dict(landmark=k + 1, center=c.tolist())
        bad = [p for p in PHASES if not interior(masks[p], c)]
        if bad:
            excluded.append(dict(**rec, reason='not_interior', phases=bad)); continue
        charts = [normalized_patch(vols[p], c, 32, meta[p]['levels']) for p in PHASES]
        t = texture(charts[1])
        if not t['accepted']:
            excluded.append(dict(**rec, reason='texture')); continue
        jobs.append((rec, charts))
    print('eligible', len(jobs), 'excluded', len(excluded), flush=True)
    with dest.open('x') as f, ProcessPoolExecutor(max_workers=6) as pool:
        for n, r in enumerate(pool.map(work, jobs)):
            f.write(json.dumps(r) + '\n'); f.flush()
            print('measured', n + 1, '/', len(jobs), flush=True)
    seal = dict(sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), records=len(jobs),
                spacing_zyx_mm=spacing.tolist(), excluded=excluded,
                levels={p: meta[p]['levels'] for p in PHASES})
    (out / 'measure_seal.json').write_text(json.dumps(seal, indent=1))


if __name__ == '__main__':
    main()
