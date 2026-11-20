"""Check the public repo's standalone instrument against the study's sealed outputs and solver."""
import sys, json
from pathlib import Path
from itertools import combinations
import numpy as np

H = Path(__file__).resolve().parents[1]
REPO = H.parent.parent / 'parallax-repos' / 'loop-visibility'
sys.path[:0] = [str(REPO), str(H.parent / 'dvc-holonomy-pilot'), str(H / 'dirlab'), str(H.parent / 'dvc-holonomy-pilot/popi_diagnosis/vendor')]
sys.dont_write_bytecode = True
from instrument import window as W, demons as DM, hodge_split as HS  # noqa: E402
from dh.complexes import Complex2  # noqa: E402
from dh.hodge import decompose  # noqa: E402
from common import CASES, load, landmarks_zyx  # noqa: E402

out = {}
# 1. window registrar on sealed DIR-Lab measurements
win = [json.loads(l) for l in (H / 'dirlab/results/window.jsonl').read_text().splitlines()]
for c, limit in [(1, None), (10, 60)]:
    v0, _ = load(c, 'T00', False); v5, _ = load(c, 'T50', False)
    m0, i0 = W.envelope(v0); m5, i5 = W.envelope(v5)
    rows = [r for r in win if r['case'] == c and r['fits']][:limit]
    dt = dr = dres = 0.0; flags = 0
    for r in rows:
        ctr = np.array(r['center'])
        a = W.normalized_patch(v0, ctr, 32, i0['levels']); b = W.normalized_patch(v5, ctr, 32, i5['levels'])
        flags += int(W.texture(a)['accepted'] != r['texture']) + int((W.interior(m0, ctr) and W.interior(m5, ctr)) != r['interior'])
        e = W.register(a, b); f = W.refine_translation(a, b, e['t']); tr = f['t'] if f['success'] else e['t']
        dt = max(dt, float(np.max(np.abs(np.array(e['t']) - r['edge']['t']))))
        dr = max(dr, float(np.max(np.abs(np.array(tr) - r['edge']['t_refined']))))
        dres = max(dres, abs(W.registered_residual(a, b, tr)[0] - r['edge']['residual_refined']))
    out[f'window_case{c}'] = dict(landmarks=len(rows), max_abs_diff_fft_vox=dt, max_abs_diff_refined_vox=dr,
                                  max_abs_diff_refined_residual=dres, guard_flag_mismatches=flags)
    print(out[f'window_case{c}'], flush=True)

# 2. Hodge split vs the study's solver on every graph used
E6 = list(combinations(range(6), 2)); CX6 = Complex2(6, E6, list(combinations(range(6), 3)))
mx = 0.0; n = 0
for l in (H / 'dirlab/results/six_phase.jsonl').read_text().splitlines():
    r = json.loads(l)
    if not (r['fits'] and r['texture'] and all(e['valid'] for e in r['edges'])):
        continue
    Y = np.array([e['t_refined'] for e in r['edges']])
    g_dh = sum(decompose(CX6, Y[:, a]).E_grad for a in range(3))
    g_hs, c_hs = HS.energies(Y, 6)
    mx = max(mx, abs(g_dh - g_hs) / max(g_dh + c_hs, 1e-12)); n += 1
E10 = list(combinations(range(10), 2)); CX10 = Complex2(10, E10, list(combinations(range(10), 3)))
for l in (H / 'results/measurements.jsonl').read_text().splitlines():
    r = json.loads(l); Y = np.array([e['t'] for e in r['edges']])
    g_dh = sum(decompose(CX10, Y[:, a]).E_grad for a in range(3)); g_hs, c_hs = HS.energies(Y, 10)
    mx = max(mx, abs(g_dh - g_hs) / max(g_dh + c_hs, 1e-12)); n += 1
out['hodge'] = dict(graphs=n, max_relative_energy_diff=mx)
print(out['hodge'], flush=True)

# 3. Demons: one pair re-fitted with the repo wrapper (case 1, T00 -> T10)
plan = json.loads((H.parent / 'dvc-holonomy-pilot/popi_deformation/PLAN.json').read_text())['method']
same = {k: plan[k] == DM.SETTINGS[k] for k in DM.SETTINGS if k != 'threads'}
sp = np.array(CASES[1][1])
f, _ = DM.fit(DM.ct_image(load(1, 'T00', False)[0], sp), DM.ct_image(load(1, 'T10', False)[0], sp), dict(DM.SETTINGS, threads=3))
arr, _ = DM.field_arrays(f)
x0 = landmarks_zyx(1, 'T00', '4D-75')[:, ::-1] * sp
u, _ = DM.sample(arr, f, x0)
sealed = [json.loads(l) for l in (H / 'dirlab/results/demons_k6_all.jsonl').read_text().splitlines()][0]
e01 = np.array(next(e for e in sealed['edges'] if e['i'] == 0 and e['j'] == 1)['d_zyx_mm'])
out['demons_case1_T00_T10'] = dict(settings_match_plan=same, landmarks=len(x0), max_abs_diff_mm=float(np.max(np.abs(u[:, ::-1] - e01))))
print(out['demons_case1_T00_T10'], flush=True)
(REPO / 'verification/instrument_equivalence.json').write_text(json.dumps(out, indent=1))
