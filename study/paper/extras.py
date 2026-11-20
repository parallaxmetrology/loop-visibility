"""Two supporting numbers for the note (visible-energy concentration; Figure 4 motion)."""
import sys, json
from pathlib import Path
from itertools import combinations
import numpy as np
P = Path(__file__).resolve().parent; R = P.parent
sys.path[:0] = [str(R.parent / 'dvc-holonomy-pilot'), str(R / 'dirlab')]; sys.dont_write_bytecode = True
from dh.complexes import Complex2
from dh.hodge import decompose
from common import CASES, landmarks_zyx
E6 = list(combinations(range(6), 2)); CX = Complex2(6, E6, list(combinations(range(6), 3))); PH = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50']
rules = json.loads((P / 'figs.json').read_text())['rules']['typical']
vis = []; out = {}
for l in (R / 'dirlab/results/six_phase.jsonl').read_text().splitlines():
    r = json.loads(l)
    if not (r['fits'] and r['texture'] and all(e['valid'] for e in r['edges'])): continue
    c = r['case']; spz = np.array(CASES[c][1][::-1])
    T = np.array([landmarks_zyx(c, p, '4D-75')[r['landmark']] for p in PH]) * spz
    Y = np.array([e['t_refined'] for e in r['edges']]) * spz
    err = Y - np.array([T[j] - T[i] for i, j in E6])
    vis.append(sum(((err[:, a] - decompose(CX, err[:, a]).grad) ** 2).sum() for a in range(3)))
    if c == rules['case'] and r['landmark'] == rules['landmark']:
        out['typical_true_T00_T50_mm'] = float(np.linalg.norm(T[5] - T[0]))
        out['typical_est_T00_T50_mm'] = float(np.linalg.norm(Y[E6.index((0, 5))]))
v = np.sort(vis)[::-1]; out['window_visible_energy_top10_share'] = float(v[:len(v) // 10].sum() / v.sum())
(P / 'extras.json').write_text(json.dumps(out, indent=1)); print(out)
