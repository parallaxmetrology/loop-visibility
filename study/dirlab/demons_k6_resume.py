"""Memory-lean resume of demons_k6.py for cases 6-10 (ADDENDUM_2b.json). Same computation."""
import sys, json, time, hashlib, gc
from pathlib import Path
from itertools import combinations
import numpy as np

D = Path(__file__).resolve().parent
PILOT = D.parents[1] / 'dvc-holonomy-pilot'
sys.path[:0] = [str(PILOT), str(D)]
sys.dont_write_bytecode = True
from common import CASES, load, landmarks_zyx  # noqa: E402
from popi_deformation.instrument import image_from_array, fit, field_arrays, sample, geometry  # noqa: E402

SIX = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50']
PART = D / 'results' / 'demons_k6_part2.jsonl'


def run_case(c, plan):
    sp = np.array(CASES[c][1])
    imgs = {}
    for p in SIX:
        v, _ = load(c, p, False)
        hu = np.asarray(v, np.float32) - 1024.0
        imgs[p] = image_from_array((np.clip(hu, -1000, 400) + 1000) / 1400, sp)
        del v, hu
    x0 = landmarks_zyx(c, 'T00', '4D-75')[:, ::-1] * sp
    xs, cov, geo, secs, out = {0: x0}, {0: np.ones(len(x0), bool)}, {}, 0.0, {}
    order = [(0, i) for i in range(1, 6)] + [(i, j) for i, j in combinations(range(6), 2) if i > 0]
    for i, j in order:
        t0 = time.monotonic()
        f, _ = fit(imgs[SIX[i]], imgs[SIX[j]], plan)
        arr = field_arrays(f)[0]
        geo[f'{i}{j}'] = geometry(f)['valid']
        if i == 0:
            u0, ok0 = sample(arr, f, x0)
            xs[j] = x0 + u0; cov[j] = ok0
        u, ok = sample(arr, f, xs[i])
        out[(i, j)] = dict(i=i, j=j, d_zyx_mm=u[:, ::-1].tolist(), covered=(ok & cov[i]).tolist())
        del f, arr; gc.collect()
        secs += time.monotonic() - t0
        print(f'  case {c} edge {i}{j} done', flush=True)
    edges = [out[(i, j)] for i, j in combinations(range(6), 2)]
    return dict(case=c, seconds=secs, geometry_valid={k: geo[k] for k in sorted(geo)}, edges=edges)


def main():
    plan = dict(json.loads((PILOT / 'popi_deformation/PLAN.json').read_text())['method'], threads=6)
    done = {json.loads(l)['case'] for l in PART.read_text().splitlines()} if PART.exists() else set()
    with PART.open('a') as fh:
        for c in range(6, 11):
            if c in done:
                continue
            r = run_case(c, plan)
            fh.write(json.dumps(r) + '\n'); fh.flush()
            print('demons k6 case', c, round(r['seconds']), 's valid', sum(r['geometry_valid'].values()), '/ 15', flush=True)
            gc.collect()
    first = [l for l in (D / 'results/demons_k6.jsonl').read_text().splitlines() if json.loads(l)['case'] <= 5]
    second = PART.read_text().splitlines()
    cases = sorted(json.loads(l)['case'] for l in first + second)
    assert cases == list(range(1, 11)), cases
    allp = D / 'results' / 'demons_k6_all.jsonl'
    allp.write_text('\n'.join(sorted(first + second, key=lambda l: json.loads(l)['case'])) + '\n')
    (D / 'results' / 'demons_k6_SEAL.json').write_text(json.dumps(dict(
        sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), targets_read=False, file='demons_k6_all.jsonl',
        sha256=hashlib.sha256(allp.read_bytes()).hexdigest()), indent=1))
    print('sealed', flush=True)


if __name__ == '__main__':
    main()
