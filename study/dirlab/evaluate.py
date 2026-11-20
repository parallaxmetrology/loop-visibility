"""Score sealed DIR-Lab measurements against PREREG.json / AMENDMENT_1.json.

Instrument A primary = refined window (FFT + Powell); FFT-only reported secondarily.
Covariance needs n >= 3 landmarks; fewer means not estimable (affects case 1 only
under the frozen guard, n = 1).
"""
import sys, json
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import spearmanr

D = Path(__file__).resolve().parent
PILOT = D.parents[1] / 'dvc-holonomy-pilot'
sys.path.insert(0, str(PILOT)); sys.path.insert(0, str(D)); sys.dont_write_bytecode = True
from common import CASES, PUBLISHED_OBSERVER_MM, landmarks_zyx  # noqa: E402
from dh.complexes import Complex2  # noqa: E402
from dh.hodge import decompose  # noqa: E402

R = D / 'results'
BAND = (1.41 / 1.5, 1.41 * 1.5)
DISPLACEMENT_MM = {1: 4.01, 2: 4.65, 3: 6.73, 4: 9.42, 5: 7.10, 6: 11.10, 7: 11.59, 8: 15.16, 9: 7.82, 10: 7.63}
rng = np.random.default_rng(20261119)
Z = rng.standard_normal((100000, 3))


def mean_norm(var):
    return float(np.linalg.norm(Z * np.sqrt(np.maximum(var, 0)), axis=1).mean())


def load_jsonl(n):
    return [json.loads(l) for l in (R / n).read_text().splitlines()]


def main():
    seal = json.loads((R / 'SEAL.json').read_text())
    import hashlib
    for n, h in seal['sha256'].items():
        assert hashlib.sha256((R / n).read_bytes()).hexdigest() == h, n
    win = load_jsonl('window.jsonl'); dem = {r['case']: r for r in load_jsonl('demons.jsonl')}
    out = dict(band=BAND, cases={})
    for setname in ['primary', 'S1']:
        for akey in ['t_refined', 't']:
            tag = f'{setname}_{"refined" if akey == "t_refined" else "fft"}'
            res = {}
            for c in range(1, 11):
                spz = np.array(CASES[c][1][::-1])
                L0 = landmarks_zyx(c, 'T00', '300'); L5 = landmarks_zyx(c, 'T50', '300')
                truth = (L5 - L0) * spz
                Bd = np.array(dem[c]['displacement_zyx_mm']); cov_ok = np.array(dem[c]['covered'])
                rows = [r for r in win if r['case'] == c and r['fits'] and r['edge']['valid'] and r['texture']
                        and (setname == 'S1' or r['interior']) and cov_ok[r['landmark']]]
                for drop in [False, True]:
                    k = np.array([r['landmark'] for r in rows], int)
                    if len(k) == 0:
                        res[(c, drop)] = dict(n=0, estimable=False); continue
                    A = np.array([r['edge'][akey] for r in rows]) * spz
                    eA = A - truth[k]; eB = Bd[k] - truth[k]
                    if drop:
                        keep = np.linalg.norm(A - Bd[k], axis=1) <= 3.0
                        eA, eB = eA[keep], eB[keep]
                    n = len(eA)
                    if n < 3:
                        res[(c, drop)] = dict(n=n, estimable=False); continue
                    cv = np.array([np.cov(eA[:, a], eB[:, a])[0, 1] for a in range(3)])
                    est = mean_norm(cv)
                    ratio = PUBLISHED_OBSERVER_MM[c] / est if est > 0 else float('inf')
                    res[(c, drop)] = dict(
                        n=n, estimable=True, shared_cov_zyx=cv.tolist(), negative_axes=int((cv < 0).sum()),
                        estimate_mean_norm_mm=est, published_mm=PUBLISHED_OBSERVER_MM[c], ratio=ratio,
                        in_band=bool(BAND[0] <= ratio <= BAND[1]),
                        tre_A=float(np.linalg.norm(eA, axis=1).mean()), tre_B=float(np.linalg.norm(eB, axis=1).mean()),
                        corr_zyx=[float(np.corrcoef(eA[:, a], eB[:, a])[0, 1]) for a in range(3)],
                        own_sd_A=np.sqrt(np.maximum(eA.var(0, ddof=1) - cv, 0)).tolist(),
                        own_sd_B=np.sqrt(np.maximum(eB.var(0, ddof=1) - cv, 0)).tolist(),
                        shared_sd_zyx=np.sqrt(np.maximum(cv, 0)).tolist(),
                        rounding_only_sd_zyx=(spz / np.sqrt(6)).tolist())
            for drop in [False, True]:
                sub = {c: res[(c, drop)] for c in range(1, 11)}
                est = [c for c in sub if sub[c]['estimable']]
                inb = sum(sub[c]['in_band'] for c in est)
                pooled = sum(PUBLISHED_OBSERVER_MM[c] for c in est) / sum(sub[c]['estimate_mean_norm_mm'] for c in est)
                key = tag + ('_drop3mm' if drop else '')
                out['cases'][key] = dict(
                    per_case=sub, estimable=len(est), in_band=inb, pooled_ratio=pooled,
                    pooled_in_band=bool(BAND[0] <= pooled <= BAND[1]),
                    verdict='PASS' if (inb >= 7 and BAND[0] <= pooled <= BAND[1]) else 'FAIL',
                    spearman_vs_published=float(spearmanr([sub[c]['estimate_mean_norm_mm'] for c in est], [PUBLISHED_OBSERVER_MM[c] for c in est])[0]),
                    spearman_vs_displacement=float(spearmanr([sub[c]['estimate_mean_norm_mm'] for c in est], [DISPLACEMENT_MM[c] for c in est])[0]))
    out['primary_verdict'] = out['cases']['primary_refined']['verdict']
    # visibility ceiling on the six-phase graphs (window instrument, refined), chance 5/15
    six = load_jsonl('six_phase.jsonl')
    E6 = list(combinations(range(6), 2)); CX = Complex2(6, E6, list(combinations(range(6), 3)))
    vis = {}
    for akey in ['t_refined', 't']:
        g = cyc = 0.0; n = 0
        for r in six:
            if not r['fits'] or not r['texture'] or not all(e['valid'] for e in r['edges']):
                continue
            c = r['case']; spz = np.array(CASES[c][1][::-1])
            T = np.array([landmarks_zyx(c, p, '4D-75')[r['landmark']] for p in ['T00', 'T10', 'T20', 'T30', 'T40', 'T50']]) * spz
            Y = np.array([e[akey] for e in r['edges']]) * spz
            err = Y - np.array([T[j] - T[i] for i, j in E6])
            for a in range(3):
                h = decompose(CX, err[:, a]); g += h.E_grad; cyc += h.E_curl + h.E_harm
            n += 1
        vis[akey] = dict(landmarks=n, integrable_share=g / (g + cyc), chance=5 / 15)
    out['visibility_ceiling_six_phase'] = vis
    out['phase_independence'] = 'Not run: Demons was fitted T00->T50 only (as pre-registered), so no second instrument exists on T10-T40; and those coordinates are model-propagated (AMENDMENT_1).'
    (R / 'summary.json').write_text(json.dumps(out, indent=1, default=str))
    for key, v in out['cases'].items():
        print(key, v['verdict'], 'in band', v['in_band'], '/', v['estimable'], 'pooled', round(v['pooled_ratio'], 3),
              'rho pub', round(v['spearman_vs_published'], 2), 'rho disp', round(v['spearman_vs_displacement'], 2))
    print(json.dumps(vis, indent=1))


if __name__ == '__main__':
    main()
