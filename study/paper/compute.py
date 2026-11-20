"""Numbers for the visibility-ceiling note, from sealed measurements only.

V (visibility) = cyclic share of error energy = 1 - integrable share.
Truth (node positions) is exactly integrable, so the cyclic part of a measurement
equals the cyclic part of its error. Annotation noise is node noise: integrable.
"""
import sys, json
from pathlib import Path
from itertools import combinations
import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parent
PILOT = ROOT.parent / 'dvc-holonomy-pilot'
sys.path[:0] = [str(PILOT), str(ROOT), str(ROOT / 'dirlab')]
sys.dont_write_bytecode = True
from dh.complexes import Complex2  # noqa: E402
from dh.hodge import decompose  # noqa: E402
import evaluate as ev  # noqa: E402  (POPI)
from common import CASES, landmarks_zyx, PUBLISHED_OBSERVER_MM  # noqa: E402  (DIR-Lab)

rng = np.random.default_rng(20261120)
K = lambda n: Complex2(n, list(combinations(range(n), 2)), list(combinations(range(n), 3)))


def split(cx, err):
    g = c = 0.0
    for a in range(3):
        h = decompose(cx, err[:, a]); g += h.E_grad; c += h.E_curl + h.E_harm
    return g, c


def boot_share(G, C, groups, B=4000):
    G, C = np.asarray(G), np.asarray(C)
    ug = sorted(set(groups)); idx = {u: np.where(np.array(groups) == u)[0] for u in ug}
    sh = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug))
        rows = np.concatenate([idx[u] for u in pick])
        sh.append(G[rows].sum() / (G[rows].sum() + C[rows].sum()))
    return [float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))]


out = {}
# chance baseline: iid isotropic edge errors
for n in [3, 6, 10]:
    cx = K(n); m = cx.n_edges
    s = [split(cx, rng.standard_normal((m, 3))) for _ in range(2000)]
    out[f'chance_K{n}'] = dict(simulated=float(np.mean([g / (g + c) for g, c in s])), formula_2_over_n=2 / n, edges=m)

# POPI K10
recs = [json.loads(l) for l in (ROOT / 'results/measurements.jsonl').read_text().splitlines()]
sp = np.array(json.loads((ROOT / 'results/measure_seal.json').read_text())['spacing_zyx_mm'])
P10 = ev.truth_mm(); cx10 = K(10)
for key in ['t', 't_refined']:
    G, C, per, msq = [], [], [], []
    for r in recs:
        k = r['landmark'] - 1
        Y = np.array([e[key] for e in r['edges']]) * sp
        T = np.array([P10[p][k] for p in ev.PHASES])
        err = Y - np.array([T[j] - T[i] for i, j in ev.EDGES])
        g, c = split(cx10, err); G.append(g); C.append(c); per.append(g / (g + c)); msq.append((err ** 2).sum(1).mean())
    tot = sum(G) + sum(C); Gs = sum(G)
    adj = {}
    for d in [0.5, 0.85, 1.13, 2.0]:
        N = 2.356 * d * d * 45 * len(recs)  # conservative: d read as per-point mean error
        adj[str(d)] = (Gs - N) / (tot - N) if tot > N else None
    out[f'popi_{key}'] = dict(landmarks=len(recs), edges_per=45, integrable_share=Gs / tot, ci95=boot_share(G, C, list(range(len(G)))),
                              per_landmark_median=float(np.median(per)), per_landmark_q10_q90=np.percentile(per, [10, 90]).tolist(),
                              per_landmark_min=float(np.min(per)), mean_sq_edge_error_mm2=float(np.mean(msq)),
                              truth_noise_adjusted_share=adj)

# DIR-Lab K6
six = [json.loads(l) for l in (ROOT / 'dirlab/results/six_phase.jsonl').read_text().splitlines()]
E6 = list(combinations(range(6), 2)); cx6 = K(6)
PH6 = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50']
for key in ['t', 't_refined']:
    G, C, per, cases, msq, Nlist = [], [], [], [], [], []
    for r in six:
        if not r['fits'] or not r['texture'] or not all(e['valid'] for e in r['edges']):
            continue
        c = r['case']; spz = np.array(CASES[c][1][::-1])
        T = np.array([landmarks_zyx(c, p, '4D-75')[r['landmark']] for p in PH6]) * spz
        Y = np.array([e[key] for e in r['edges']]) * spz
        err = Y - np.array([T[j] - T[i] for i, j in E6])
        g, cc = split(cx6, err); G.append(g); C.append(cc); per.append(g / (g + cc)); cases.append(c)
        msq.append((err ** 2).sum(1).mean()); Nlist.append(2.356 * PUBLISHED_OBSERVER_MM[c] ** 2 * 15)
    G, C, Nlist = np.array(G), np.array(C), np.array(Nlist)
    tot = G.sum() + C.sum(); N = Nlist.sum()
    by_case = {}
    for c in sorted(set(cases)):
        m = np.array(cases) == c
        by_case[c] = dict(n=int(m.sum()), share=float(G[m].sum() / (G[m].sum() + C[m].sum())),
                          adjusted=float((G[m].sum() - Nlist[m].sum()) / (G[m].sum() + C[m].sum() - Nlist[m].sum())),
                          mean_sq_edge_error_mm2=float(np.mean(np.array(msq)[m])))
    out[f'dirlab_{key}'] = dict(landmarks=int(len(G)), cases=len(set(cases)), edges_per=15, integrable_share=float(G.sum() / tot),
                                ci95_case_cluster=boot_share(G, C, cases), per_landmark_median=float(np.median(per)),
                                per_landmark_q10_q90=np.percentile(per, [10, 90]).tolist(), mean_sq_edge_error_mm2=float(np.mean(msq)),
                                truth_noise_energy_fraction=float(N / tot),
                                truth_noise_adjusted_share=float((G.sum() - N) / (tot - N)),
                                adjusted_ci95_case_cluster=boot_share(G - Nlist, C, cases),
                                by_case=by_case)
(P / 'numbers.json').write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, json.dumps({a: b for a, b in v.items() if a != 'by_case'}))
for c, v in out['dirlab_t_refined']['by_case'].items():
    print(' case', c, v)
