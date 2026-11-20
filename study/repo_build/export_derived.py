"""Export per-landmark derived results for the public repo.

Writes only energies, mean errors and indicator values per landmark. No coordinates, no
displacement or error vectors, no image data. Reads the sealed study outputs plus the
landmark files (DIR-Lab, POPI), which are not redistributed.
"""
import sys, json, csv
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import spearmanr

H = Path(__file__).resolve().parents[1]
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else H.parent.parent / 'parallax-repos' / 'loop-visibility'
sys.path[:0] = [str(REPO), str(H / 'dirlab'), str(H)]
sys.dont_write_bytecode = True
from instrument.hodge_split import energies, split  # noqa: E402
from common import CASES, landmarks_zyx, PUBLISHED_OBSERVER_MM  # noqa: E402

PH = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50']; E6 = list(combinations(range(6), 2))
NOISE = lambda d, m: 2.356 * d * d * m  # pessimistic allowance, mm^2 per landmark graph with m edges
D = REPO / 'derived'; D.mkdir(exist_ok=True)


def wcsv(name, header, rows):
    with (D / name).open('w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


# DIR-Lab window registrar (six phases, 15 edges per landmark)
rows, loc = [], []
for l in (H / 'dirlab/results/six_phase.jsonl').read_text().splitlines():
    r = json.loads(l)
    if not (r['fits'] and r['texture'] and all(e['valid'] for e in r['edges'])):
        continue
    c, k = r['case'], r['landmark']; spz = np.array(CASES[c][1][::-1])
    T = np.array([landmarks_zyx(c, p, '4D-75')[k] for p in PH]) * spz
    Tt = np.array([T[j] - T[i] for i, j in E6])
    for reg, key, rk in [('window_refined', 't_refined', 'residual_refined'), ('window_fft', 't', 'residual')]:
        Y = np.array([e[key] for e in r['edges']]) * spz
        err = Y - Tt
        g, cy = energies(err, 6)
        te = np.linalg.norm(err, axis=1)
        rows.append([c, k, reg, f'{g:.10g}', f'{cy:.10g}', f'{te.mean():.10g}', f'{NOISE(PUBLISHED_OBSERVER_MM[c], 15):.10g}'])
        _, C = split(Y, 6)  # cyclic part of the measurement = cyclic part of the error
        pred = dict(cyclic=np.linalg.norm(C, axis=1), ncc=np.array([e[rk] for e in r['edges']]), shift=np.linalg.norm(Y, axis=1))
        worst = int(np.argmax(te))
        loc.append([c, k, reg] + [int(np.argmax(pred[p]) == worst) for p in ['cyclic', 'ncc', 'shift']]
                   + [f'{np.nan_to_num(spearmanr(pred[p], te)[0]):.10g}' for p in ['cyclic', 'ncc', 'shift']])
wcsv('dirlab_window_per_landmark.csv', ['case', 'landmark_index', 'registrar', 'E_integrable_mm2', 'E_cyclic_mm2', 'mean_edge_error_mm', 'noise_allowance_mm2'], rows)
wcsv('dirlab_window_worst_edge.csv', ['case', 'landmark_index', 'registrar', 'worst_named_by_cyclic', 'worst_named_by_ncc', 'worst_named_by_shift',
                                      'spearman_cyclic', 'spearman_ncc', 'spearman_shift'], loc)
win_set = {(r[0], r[1]) for r in rows}

# DIR-Lab Demons (all 15 pairs, 75 landmarks per case)
rows = []
for l in (H / 'dirlab/results/demons_k6_all.jsonl').read_text().splitlines():
    r = json.loads(l); c = r['case']; spz = np.array(CASES[c][1][::-1])
    T = np.array([landmarks_zyx(c, p, '4D-75') for p in PH]) * spz
    Dd = np.array([e['d_zyx_mm'] for e in r['edges']]); cov = np.array([e['covered'] for e in r['edges']])
    for k in range(Dd.shape[1]):
        if not cov[:, k].all():
            continue
        err = Dd[:, k] - np.array([T[j, k] - T[i, k] for i, j in E6])
        g, cy = energies(err, 6)
        rows.append([c, k, 'demons', f'{g:.10g}', f'{cy:.10g}', f'{np.linalg.norm(err, axis=1).mean():.10g}',
                     f'{NOISE(PUBLISHED_OBSERVER_MM[c], 15):.10g}', int((c, k) in win_set)])
wcsv('dirlab_demons_per_landmark.csv', ['case', 'landmark_index', 'registrar', 'E_integrable_mm2', 'E_cyclic_mm2', 'mean_edge_error_mm',
                                        'noise_allowance_mm2', 'in_window_set'], rows)

# POPI window registrar (ten phases, 45 edges per landmark)
import importlib.util as _iu  # POPI scorer from the study (truth loader, phase order), loaded by path
_spec = _iu.spec_from_file_location('popi_evaluate', H / 'evaluate.py'); ev = _iu.module_from_spec(_spec); _spec.loader.exec_module(ev)
P = ev.truth_mm(); sp = np.array(json.loads((H / 'results/measure_seal.json').read_text())['spacing_zyx_mm'])
rows = []
for l in (H / 'results/measurements.jsonl').read_text().splitlines():
    r = json.loads(l); k = r['landmark'] - 1
    T = np.array([P[p][k] for p in ev.PHASES])
    for reg, key in [('window_refined', 't_refined'), ('window_fft', 't')]:
        err = np.array([e[key] for e in r['edges']]) * sp - np.array([T[j] - T[i] for i, j in ev.EDGES])
        g, cy = energies(err, 10)
        rows.append([r['landmark'], reg, f'{g:.10g}', f'{cy:.10g}', f'{(err ** 2).sum(1).mean():.10g}'])
wcsv('popi_window_per_landmark.csv', ['landmark', 'registrar', 'E_integrable_mm2', 'E_cyclic_mm2', 'mean_sq_edge_error_mm2'], rows)
print('exported', sorted(p.name for p in D.iterdir()))
