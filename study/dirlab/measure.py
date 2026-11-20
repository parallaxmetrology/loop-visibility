"""Truth-blind DIR-Lab measurement (PREREG.json + AMENDMENT_1.json).

Reads only T00 landmark files. Instrument A: pilot FFT window + refinement.
Instrument B: pilot chapter-15 Diffeomorphic Demons, T00 (fixed) -> T50 (moving).
"""
import sys, json, time, hashlib
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import numpy as np

D = Path(__file__).resolve().parent
PILOT = D.parents[1] / 'dvc-holonomy-pilot'
sys.path.insert(0, str(PILOT)); sys.path.insert(0, str(D))
sys.dont_write_bytecode = True
from common import CASES, load, landmarks_zyx  # noqa: E402
from interior.selection import envelope, interior, texture  # noqa: E402
from dvc_pilot.acceptance_run import normalized_patch  # noqa: E402
from dvc_pilot.registration import register, registered_residual  # noqa: E402
from dvc_pilot.refinement import refine_translation  # noqa: E402

OUT = D / 'results'
SIX = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50']


def edge(a, b):
    r = register(a, b)
    if not r['valid']:
        return dict(valid=False)
    f = refine_translation(a, b, r['t'])
    tr = f['t'] if f['success'] else r['t']
    return dict(valid=True, t=r['t'], residual=r['residual'], ambiguity=r['ambiguity'], refine_success=f['success'],
                t_refined=tr, residual_refined=registered_residual(a, b, tr)[0])


def fits(shape, c):
    return bool(np.all(c - 16 >= 0) and np.all(c + 16 <= np.array(shape)))


def window_case(c):
    vols = {p: load(c, p, False)[0] for p in SIX}
    env = {p: envelope(vols[p]) for p in SIX}
    lev = {p: env[p][1]['levels'] for p in SIX}
    rows = []
    for k, pt in enumerate(landmarks_zyx(c, 'T00', '300')):
        ctr = np.rint(pt).astype(int)
        r = dict(case=c, landmark=k, center=ctr.tolist(), fits=fits(vols['T00'].shape, ctr))
        if not r['fits']:
            rows.append(r); continue
        a = normalized_patch(vols['T00'], ctr, 32, lev['T00']); b = normalized_patch(vols['T50'], ctr, 32, lev['T50'])
        r['interior'] = bool(interior(env['T00'][0], ctr) and interior(env['T50'][0], ctr))
        r['texture'] = bool(texture(a)['accepted'])
        r['edge'] = edge(a, b)
        rows.append(r)
    six = []
    for k, pt in enumerate(landmarks_zyx(c, 'T00', '4D-75')):
        ctr = np.rint(pt).astype(int)
        r = dict(case=c, landmark=k, center=ctr.tolist(), fits=fits(vols['T00'].shape, ctr))
        if r['fits']:
            ch = [normalized_patch(vols[p], ctr, 32, lev[p]) for p in SIX]
            r['texture'] = bool(texture(ch[0])['accepted'])
            r['interior_all'] = bool(all(interior(env[p][0], ctr) for p in SIX))
            r['edges'] = [dict(i=i, j=j, **edge(ch[i], ch[j])) for i, j in combinations(range(6), 2)]
        six.append(r)
    return c, rows, six


def demons_case(c, plan):
    from popi_deformation.instrument import image_from_array, fit, field_arrays, map_points, geometry
    (nx, ny, nz), sp = CASES[c]
    imgs = {}
    for p in ['T00', 'T50']:
        v, _ = load(c, p, False)
        hu = np.asarray(v, np.float32) - 1024.0
        imgs[p] = image_from_array((np.clip(hu, -1000, 400) + 1000) / 1400, sp)
    t0 = time.monotonic()
    field, levels = fit(imgs['T00'], imgs['T50'], plan)
    arr, _ = field_arrays(field)
    pts_xyz = landmarks_zyx(c, 'T00', '300')[:, ::-1] * np.array(sp)
    mapped, covered = map_points(arr, field, pts_xyz)
    g = geometry(field)
    return dict(case=c, seconds=time.monotonic() - t0, displacement_zyx_mm=(mapped - pts_xyz)[:, ::-1].tolist(),
                covered=covered.tolist(), geometry=g,
                levels=[{k: v for k, v in L.items() if k != 'trace'} for L in levels])


def main():
    OUT.mkdir(exist_ok=True)
    for n in ['window.jsonl', 'six_phase.jsonl', 'demons.jsonl']:
        if (OUT / n).exists():
            raise SystemExit('results exist; refusing to overwrite')
    with ProcessPoolExecutor(max_workers=5) as pool, (OUT / 'window.jsonl').open('x') as fw, (OUT / 'six_phase.jsonl').open('x') as fs:
        for c, rows, six in pool.map(window_case, range(1, 11)):
            for r in rows: fw.write(json.dumps(r) + '\n')
            for r in six: fs.write(json.dumps(r) + '\n')
            print('window case', c, flush=True)
    plan = json.loads((PILOT / 'popi_deformation/PLAN.json').read_text())['method']
    with (OUT / 'demons.jsonl').open('x') as fd:
        for c in range(1, 11):
            r = demons_case(c, plan)
            fd.write(json.dumps(r) + '\n'); fd.flush()
            print('demons case', c, round(r['seconds']), 's', r['geometry']['valid'], flush=True)
    seal = dict(sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), targets_read=False,
                sha256={n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in ['window.jsonl', 'six_phase.jsonl', 'demons.jsonl']})
    (OUT / 'SEAL.json').write_text(json.dumps(seal, indent=1))
    print('sealed')


if __name__ == '__main__':
    main()
