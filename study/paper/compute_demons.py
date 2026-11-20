"""Visibility for Demons on DIR-Lab K6 (dirlab/ADDENDUM_2.json), from the sealed run."""
import sys, json, hashlib
from pathlib import Path
from itertools import combinations
import numpy as np

P = Path(__file__).resolve().parent; R = P.parent
sys.path[:0] = [str(R.parent / 'dvc-holonomy-pilot'), str(R / 'dirlab')]
sys.dont_write_bytecode = True
from dh.complexes import Complex2  # noqa: E402
from dh.hodge import decompose  # noqa: E402
from common import CASES, landmarks_zyx, PUBLISHED_OBSERVER_MM  # noqa: E402

seal = json.loads((R / 'dirlab/results/demons_k6_SEAL.json').read_text())
src = R / 'dirlab/results' / seal['file']  # demons_k6_all.jsonl (cases 1-5 original run, 6-10 resume; ADDENDUM_2b)
assert hashlib.sha256(src.read_bytes()).hexdigest() == seal['sha256']
E6 = list(combinations(range(6), 2)); CX = Complex2(6, E6, list(combinations(range(6), 3)))
PH = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50']
rng = np.random.default_rng(20261120)
win = set()
for l in (R / 'dirlab/results/six_phase.jsonl').read_text().splitlines():
    r = json.loads(l)
    if r['fits'] and r['texture'] and all(e['valid'] for e in r['edges']):
        win.add((r['case'], r['landmark']))
rows = []
geo_valid = 0
for l in src.read_text().splitlines():
    r = json.loads(l); c = r['case']; spz = np.array(CASES[c][1][::-1])
    geo_valid += sum(r['geometry_valid'].values())
    T = np.array([landmarks_zyx(c, p, '4D-75') for p in PH]) * spz  # 6 x 75 x 3
    D = np.array([e['d_zyx_mm'] for e in r['edges']])  # 15 x 75 x 3
    cov = np.array([e['covered'] for e in r['edges']])  # 15 x 75
    for k in range(D.shape[1]):
        if not cov[:, k].all():
            continue
        err = D[:, k] - np.array([T[j, k] - T[i, k] for i, j in E6])
        g = cc = 0.0
        for a in range(3):
            h = decompose(CX, err[:, a]); g += h.E_grad; cc += h.E_curl + h.E_harm
        rows.append(dict(case=c, lm=k, g=g, c=cc, N=2.356 * PUBLISHED_OBSERVER_MM[c] ** 2 * 15,
                         tre=float(np.linalg.norm(err, axis=1).mean()), matched=(c, k) in win))


def summarise(sel):
    G = np.array([r['g'] for r in sel]); C = np.array([r['c'] for r in sel]); Nn = np.array([r['N'] for r in sel])
    cases = np.array([r['case'] for r in sel]); uc = sorted(set(cases))
    idx = {u: np.where(cases == u)[0] for u in uc}
    def boot(Gx):
        out = []
        for _ in range(4000):
            rr = np.concatenate([idx[u] for u in rng.choice(uc, len(uc))])
            out.append(Gx[rr].sum() / (Gx[rr].sum() + C[rr].sum()))
        return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
    per = G / (G + C)
    return dict(landmarks=len(sel), integrable_share=float(G.sum() / (G.sum() + C.sum())), ci95_case_cluster=boot(G),
                truth_noise_energy_fraction=float(Nn.sum() / (G.sum() + C.sum())),
                truth_noise_adjusted_share=float((G.sum() - Nn.sum()) / (G.sum() + C.sum() - Nn.sum())),
                adjusted_ci95_case_cluster=boot(G - Nn), per_landmark_median=float(np.median(per)),
                mean_edge_tre_mm=float(np.mean([r['tre'] for r in sel])),
                by_case={int(u): dict(n=int(len(idx[u])), share=float(G[idx[u]].sum() / (G[idx[u]].sum() + C[idx[u]].sum())),
                                      mean_edge_tre_mm=float(np.mean([sel[i]['tre'] for i in idx[u]]))) for u in uc})


wt = []
for l in (R / 'dirlab/results/six_phase.jsonl').read_text().splitlines():
    r = json.loads(l)
    if not (r['fits'] and r['texture'] and all(e['valid'] for e in r['edges'])):
        continue
    c = r['case']; spz = np.array(CASES[c][1][::-1])
    T = np.array([landmarks_zyx(c, p, '4D-75')[r['landmark']] for p in PH]) * spz
    Y = np.array([e['t_refined'] for e in r['edges']]) * spz
    wt.append(float(np.linalg.norm(Y - np.array([T[j] - T[i] for i, j in E6]), axis=1).mean()))
out = dict(fits_geometry_valid=geo_valid, fits=150, all=summarise(rows), matched=summarise([r for r in rows if r['matched']]),
           window_refined_mean_edge_tre_mm=float(np.mean(wt)))
(P / 'numbers_demons.json').write_text(json.dumps(out, indent=1))
for k in ['all', 'matched']:
    print(k, json.dumps({a: b for a, b in out[k].items() if a != 'by_case'}))
print({c: round(v['share'], 3) for c, v in out['all']['by_case'].items()})
print('window matched-set mean edge TRE for comparison computed in note')
