"""Truth-blind Demons on all 15 DIR-Lab phase pairs per case (ADDENDUM_2.json)."""
import sys, json, time, hashlib
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import numpy as np

D = Path(__file__).resolve().parent
PILOT = D.parents[1] / 'dvc-holonomy-pilot'
sys.path[:0] = [str(PILOT), str(D)]
sys.dont_write_bytecode = True
from common import CASES, load, landmarks_zyx  # noqa: E402

SIX = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50']
OUT = D / 'results' / 'demons_k6.jsonl'


def run_case(c):
    from popi_deformation.instrument import image_from_array, fit, field_arrays, sample, geometry
    plan = json.loads((PILOT / 'popi_deformation/PLAN.json').read_text())['method']
    plan = dict(plan, threads=3)
    sp = np.array(CASES[c][1])  # xyz
    imgs = {}
    for p in SIX:
        v, _ = load(c, p, False)
        hu = np.asarray(v, np.float32) - 1024.0
        imgs[p] = image_from_array((np.clip(hu, -1000, 400) + 1000) / 1400, sp)
    x0 = landmarks_zyx(c, 'T00', '4D-75')[:, ::-1] * sp  # physical xyz, T00 only
    fields, geo, secs = {}, {}, 0.0
    for i, j in combinations(range(6), 2):
        t0 = time.monotonic()
        f, _ = fit(imgs[SIX[i]], imgs[SIX[j]], plan)
        fields[(i, j)] = (field_arrays(f)[0], f)
        geo[f'{i}{j}'] = geometry(f)['valid']
        secs += time.monotonic() - t0
    xs = {0: x0}
    cov = {0: np.ones(len(x0), bool)}
    for i in range(1, 6):
        arr, f = fields[(0, i)]
        u, ok = sample(arr, f, x0)
        xs[i] = x0 + u; cov[i] = ok
    edges = []
    for i, j in combinations(range(6), 2):
        arr, f = fields[(i, j)]
        u, ok = sample(arr, f, xs[i])
        edges.append(dict(i=i, j=j, d_zyx_mm=u[:, ::-1].tolist(), covered=(ok & cov[i]).tolist()))
    return dict(case=c, seconds=secs, geometry_valid=geo, edges=edges)


def main():
    if OUT.exists():
        raise SystemExit('exists')
    with ProcessPoolExecutor(max_workers=3) as pool, OUT.open('x') as f:
        for r in pool.map(run_case, range(1, 11)):
            f.write(json.dumps(r) + '\n'); f.flush()
            print('demons k6 case', r['case'], round(r['seconds']), 's valid', sum(r['geometry_valid'].values()), '/ 15', flush=True)
    (D / 'results' / 'demons_k6_SEAL.json').write_text(json.dumps(dict(
        sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), targets_read=False,
        sha256=hashlib.sha256(OUT.read_bytes()).hexdigest()), indent=1))
    print('sealed', flush=True)


if __name__ == '__main__':
    main()
