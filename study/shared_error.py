"""Post-hoc: is the remaining TRE shared between two unrelated instruments?

Instrument 1: this study's K10 measurements (FFT and Powell-refined 32-cube
translation, phase-10 star edges). Instrument 2: the pilot's chapter-15
whole-volume Diffeomorphic Demons, final 2 mm level, read from
../dvc-holonomy-pilot/popi_deformation/results/scored.json (not refitted).
Truth: POPI expert landmarks.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.dont_write_bytecode = True
import evaluate as ev  # noqa: E402

P = ev.truth_mm()
sc = json.loads((ev.PILOT / 'popi_deformation/results/scored.json').read_text())
dem = {}
for r in sc:
    t = ev.TRIPLETS[r['triplet']]
    for b, e in zip([1, 2], r['levels']['2']['edges'][:2]):  # edges A->B, A->C; A is phase 10
        if e['valid']:
            dem[(r['landmark'], t[b])] = (np.array(e['predicted_xyz_mm']) - np.array(e['source_xyz_mm']))[::-1]
recs = [json.loads(l) for l in (HERE / 'results/measurements.jsonl').read_text().splitlines()]
sp = np.array(json.loads((HERE / 'results/measure_seal.json').read_text())['spacing_zyx_mm'])
targets = [p for p in ev.PHASES if p != '10']
out = dict(instrument_2='chapter-15 Diffeomorphic Demons 2 mm level (stored predictions)')
for key, name in [('t', 'fft'), ('t_refined', 'refined')]:
    E1, E2, RES, MOT = [], [], [], []
    for rec in recs:
        lm = rec['landmark']; k = lm - 1
        Y = np.array([e[key] for e in rec['edges']]) * sp
        rk = 'residual' if key == 't' else 'residual_refined'
        row1, row2 = [], []
        for p in targets:
            tru = P[p][k] - P['10'][k]
            row1.append(ev.edge_lookup(Y, '10', p) - tru); row2.append(dem[(lm, p)] - tru)
            i, j = sorted([1, ev.PHASES.index(p)])
            RES.append(rec['edges'][ev.EIDX[(i, j)]][rk]); MOT.append(np.linalg.norm(tru))
        E1.append(row1); E2.append(row2)
    E1, E2 = np.array(E1), np.array(E2)
    f1, f2 = E1.reshape(-1, 3), E2.reshape(-1, 3)
    cov = np.array([np.cov(f1[:, a], f2[:, a])[0, 1] for a in range(3)])
    S = (E1 + E2) / 2; Sd = S - S.mean(1, keepdims=True)
    D = (E1 - E2) / 2; Dd = D - D.mean(1, keepdims=True)
    ph = [int(p) for p in targets]

    def lag(X, d):
        xs, ys = [], []
        for a in range(9):
            for b in range(a + 1, 9):
                if min(abs(ph[a] - ph[b]), 100 - abs(ph[a] - ph[b])) // 10 == d:
                    xs.append(X[:, a].ravel()); ys.append(X[:, b].ravel())
        return float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])
    e1n, e2n = np.linalg.norm(f1, axis=1), np.linalg.norm(f2, axis=1)
    avg = np.linalg.norm((f1 + f2) / 2, axis=1)
    out[name] = dict(
        pairs=int(len(f1)), landmarks=int(len(E1)),
        tre_mean=dict(instrument_1=float(e1n.mean()), demons=float(e2n.mean()), average_of_two=float(avg.mean())),
        error_corr_zyx=[float(np.corrcoef(f1[:, a], f2[:, a])[0, 1]) for a in range(3)],
        error_cosine_mean=float(np.mean(np.sum(f1 * f2, 1) / (e1n * e2n))),
        shared_sd_zyx=np.sqrt(cov).tolist(), shared_rms_3d=float(np.sqrt(cov.sum())),
        own_sd_zyx_instrument_1=np.sqrt(np.maximum(f1.var(0) - cov, 0)).tolist(),
        own_sd_zyx_demons=np.sqrt(np.maximum(f2.var(0) - cov, 0)).tolist(),
        shared_phase_lag_corr={d: lag(Sd, d) for d in [1, 2, 3, 4, 5]},
        difference_phase_lag_corr={d: lag(Dd, d) for d in [1, 2, 3, 4, 5]},
        iid_expectation_after_demeaning=-1 / 8,
        shared_abs_vs_ncc_residual_spearman=float(spearmanr(np.linalg.norm(S, axis=2).ravel(), RES)[0]),
        shared_abs_vs_true_motion_spearman=float(spearmanr(np.linalg.norm(S, axis=2).ravel(), MOT)[0]))
z = np.concatenate([P[p][:, 0] for p in ev.PHASES])
out['annotation_grid'] = dict(z_points=int(z.size), z_on_2mm_slice_grid=int(np.sum(np.isclose(z % 2, 0))),
                              z_slice_snap_pair_sd_mm=float(np.sqrt(2) * 2 / np.sqrt(12)),
                              note='Uniform +-1 mm snapping per point gives sd 2/sqrt(12) per point, sqrt(2) times that per pair.')
(HERE / 'results/shared_error.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
