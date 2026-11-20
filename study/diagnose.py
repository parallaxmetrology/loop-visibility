"""Post-hoc diagnosis (declared after evaluate.py results were seen).

1. Visibility ceiling: Hodge-split the true edge ERROR on K10. Truth is a node
   field (exactly integrable), so the cyclic part of the measurement equals the
   cyclic part of the error. The integrable share of error energy is what no loop
   can ever see. Chance share for iid edge noise on K10 is 9/45 = 20%.
2. Estimator gain, truth-free: integer-voxel crop offsets of the same phase volume
   give exact known shifts with no interpolation. Fit measured vs applied shift.
3. Gain shapes: is the integrable error a per-axis gain on the true node
   trajectory (regression of node error on node displacement)?
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'dvc-holonomy-pilot'
sys.path.insert(0, str(PILOT)); sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from evaluate import truth_mm, PHASES, EDGES, CX, sync  # noqa: E402
from dh.hodge import decompose  # noqa: E402

sp = np.array(json.loads((HERE / 'results/measure_seal.json').read_text())['spacing_zyx_mm'])
recs = [json.loads(l) for l in (HERE / 'results/measurements.jsonl').read_text().splitlines()]
P = truth_mm()
out = {}

# 1. visibility ceiling
for key in ['t', 't_refined']:
    Eg = Ec = 0.0; per = []
    node_err, node_true = [], []
    for r in recs:
        k = r['landmark'] - 1
        Y = np.array([e[key] for e in r['edges']]) * sp
        T = np.array([P[p][k] for p in PHASES])
        Tt = np.array([T[j] - T[i] for i, j in EDGES])
        err = Y - Tt
        g = c = 0.0; zs = []
        for a in range(3):
            h = decompose(CX, err[:, a])
            g += h.E_grad; c += h.E_curl + h.E_harm; zs.append(h.z)
        per.append(g / (g + c)); Eg += g; Ec += c
        Z = np.array(zs).T  # node error potentials (zero mean)
        node_err.append(Z - Z[1]); node_true.append(T - T[1])
    node_err = np.concatenate(node_err); node_true = np.concatenate(node_true)
    gains = [float(node_err[:, a] @ node_true[:, a] / (node_true[:, a] @ node_true[:, a])) for a in range(3)]
    fit_res = node_err - node_true * np.array(gains)
    out[key] = dict(integrable_share_pooled=Eg / (Eg + Ec), integrable_share_median=float(np.median(per)),
                    integrable_share_min=float(np.min(per)), chance_share=9 / 45,
                    node_error_gain_zyx=[1 + g for g in gains],
                    node_error_energy_explained_by_gain=float(1 - (fit_res ** 2).sum() / (node_err ** 2).sum()))

# 2. truth-free estimator gain on real POPI texture
from correspondence.common import volume, landmarks  # noqa: E402
from interior.selection import envelope  # noqa: E402
from dvc_pilot.acceptance_run import normalized_patch  # noqa: E402
from dvc_pilot.registration import register  # noqa: E402
from dvc_pilot.refinement import refine_translation  # noqa: E402

v, s, _ = volume('10'); _, meta = envelope(v)
src = landmarks('10', s)
shifts = [np.array(d) for d in [(1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 2, 0), (0, 4, 0), (0, 6, 0),
                                (0, 0, 2), (0, 0, 4), (0, 0, 6), (2, 3, 3), (3, 6, 6)]]
rows = []
for r in recs:
    c = np.array(r['center'])
    a = normalized_patch(v, c, 32, meta['levels'])
    for d in shifts:
        b = normalized_patch(v, c + d, 32, meta['levels'])  # content at x+d appears at x: true t = -d? check sign below
        f = register(a, b)
        rf = refine_translation(a, b, f['t'])
        rows.append(dict(landmark=r['landmark'], d=d.tolist(), t=f['t'], t_ref=rf['t'] if rf['success'] else f['t']))
D = np.array([r['d'] for r in rows], float); Tm = np.array([r['t'] for r in rows]); Tr = np.array([r['t_ref'] for r in rows])
# sign: patch b at centre c+d shows a's point x+d at local x, so b(x) = a(x+d); a-coords map to b-coords by x -> x-d.
truth = -D
def gain(M):
    return [float(M[:, a] @ truth[:, a] / (truth[:, a] @ truth[:, a])) for a in range(3)]
out['known_shift'] = dict(n=len(rows), gain_fft_zyx=gain(Tm), gain_refined_zyx=gain(Tr),
                          mean_err_fft_vox=float(np.linalg.norm(Tm - truth, axis=1).mean()),
                          mean_err_refined_vox=float(np.linalg.norm(Tr - truth, axis=1).mean()),
                          by_shift={str(d.tolist()): dict(fft=np.mean(Tm[(D == d).all(1)], 0).round(3).tolist(),
                                                          refined=np.mean(Tr[(D == d).all(1)], 0).round(3).tolist())
                                    for d in shifts})
(HERE / 'results/diagnosis.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
