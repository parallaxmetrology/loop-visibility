"""Score the 10-phase graph against POPI landmarks, as declared in PLAN.json."""
import sys, json, csv
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'dvc-holonomy-pilot'
sys.path.insert(0, str(PILOT))
sys.dont_write_bytecode = True
from dh.complexes import Complex2  # noqa: E402
from dh.hodge import decompose  # noqa: E402

PHASES = [f'{i:02}' for i in range(0, 100, 10)]
EDGES = list(combinations(range(10), 2))
EIDX = {e: k for k, e in enumerate(EDGES)}
CX = Complex2(10, EDGES, list(combinations(range(10), 3)))
REF = 1  # phase '10'
TRIPLETS = [['10', '00', '20'], ['10', '30', '40'], ['10', '50', '60'], ['10', '70', '80'], ['10', '90', '00']]
B, SEED = 4000, 20261119


def truth_mm():
    P = {}
    for p in PHASES:
        a = np.loadtxt(PILOT / 'correspondence/data' / f'{p}-Landmarks.pts')
        assert a.shape == (41, 3)
        P[p] = a[:40, ::-1]  # XYZ mm -> ZYX mm
    return P


def sync(Y, w):
    """Hodge gradient per axis. Returns node potentials z (10x3) and cyclic remainder (45x3)."""
    Z, G, ortho = [], [], 0.0
    for k in range(3):
        h = decompose(CX, Y[:, k], w)
        Z.append(h.z); G.append(h.grad); ortho = max(ortho, h.ortho_err)
    Z, G = np.array(Z).T, np.array(G).T
    return Z, Y - G, ortho


def irls(Y, prior, scale=1.0, iters=10):
    w = prior.copy()
    for _ in range(iters):
        Z, R, _ = sync(Y, w)
        r = np.linalg.norm(R, axis=1)
        w = prior / (1 + (r / scale) ** 2)
    Z, R, _ = sync(Y, w)
    return Z, w


def pair_est(Z):
    return np.array([Z[j] - Z[i] for i, j in EDGES])


def edge_lookup(Yp, a, b):
    """Estimated displacement from phase a to phase b from a 45x3 pair array."""
    i, j = PHASES.index(a), PHASES.index(b)
    return Yp[EIDX[(i, j)]] if i < j else -Yp[EIDX[(j, i)]]


def auroc(score, label):
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return float('nan')
    r = rankdata(score)
    n1 = label.sum(); n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    spacing = np.array(json.loads((HERE / 'results/measure_seal.json').read_text())['spacing_zyx_mm'])
    recs = [json.loads(l) for l in (HERE / 'results/measurements.jsonl').read_text().splitlines()]
    P = truth_mm()
    estimators = ['direct', 'ls_sync', 'holonomy_robust', 'residual_weighted', 'combined']
    per = []  # one dict per landmark
    for rec in recs:
        lm = rec['landmark']; k = lm - 1
        assert all(e['valid'] for e in rec['edges']), lm
        T = np.array([[P[p][k][a] for a in range(3)] for p in PHASES])  # node truth, mm
        Ttrue = np.array([T[j] - T[i] for i, j in EDGES])
        row = dict(landmark=lm)
        for variant, key, rkey in [('fft', 't', 'residual'), ('refined', 't_refined', 'residual_refined')]:
            Y = np.array([e[key] for e in rec['edges']]) * spacing
            res = np.array([e[rkey] for e in rec['edges']])
            ones = np.ones(45)
            Z_ls, R_ls, ortho = sync(Y, ones)
            prior_res = 1 / (res + 0.02)
            Z_rob, w_rob = irls(Y, ones)
            Z_res, _, _ = sync(Y, prior_res)
            Z_cmb, w_cmb = irls(Y, prior_res)
            pairs = dict(direct=Y, ls_sync=pair_est(Z_ls), holonomy_robust=pair_est(Z_rob),
                         residual_weighted=pair_est(Z_res), combined=pair_est(Z_cmb))
            v = row.setdefault(variant, {})
            v['ortho_max'] = ortho
            for name, Yp in pairs.items():
                err = np.linalg.norm(Yp - Ttrue, axis=1)
                out = [EIDX[(min(REF, j), max(REF, j))] for j in range(10) if j != REF]
                v[name] = dict(
                    out10=float(err[out].mean()),
                    out10_per_phase={PHASES[j]: float(err[EIDX[(min(REF, j), max(REF, j))]]) for j in range(10) if j != REF},
                    all45=float(err.mean()),
                    gross45=int((err > 2).sum()),
                    triplet_max=[float(max(np.linalg.norm(edge_lookup(Yp, a, b) - (P[b][k] - P[a][k]))
                                           for a, b in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]))
                                 for t in TRIPLETS])
            # localisation, on the raw measured edges
            err_direct = np.linalg.norm(Y - Ttrue, axis=1)
            v['edge'] = dict(true_err=err_direct.tolist(), cyclic=np.linalg.norm(R_ls, axis=1).tolist(),
                             ncc=res.tolist(), shift=np.linalg.norm(Y, axis=1).tolist(),
                             true_motion=np.linalg.norm(Ttrue, axis=1).tolist(),
                             w_robust=w_rob.tolist())
        per.append(row)

    rng = np.random.default_rng(SEED)
    L = len(per)
    boots = rng.integers(0, L, size=(B, L))

    def ci(vals):
        vals = np.asarray(vals, float)
        m = vals[boots].mean(axis=1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    summary = dict(landmarks=L, edges_per_landmark=45, independent_cycles_per_axis=36,
                   identity_out10=float(np.mean([np.mean([np.linalg.norm(P[p][r['landmark'] - 1] - P['10'][r['landmark'] - 1])
                                                          for p in PHASES if p != '10']) for r in per])),
                   identity_all45=float(np.mean([np.mean([np.linalg.norm(P[PHASES[j]][r['landmark'] - 1] - P[PHASES[i]][r['landmark'] - 1])
                                                          for i, j in EDGES]) for r in per])))
    for variant in ['fft', 'refined']:
        S = summary[variant] = {}
        S['hodge_orthogonality_max'] = max(r[variant]['ortho_max'] for r in per)
        base = np.array([r[variant]['direct']['out10'] for r in per])
        for name in estimators:
            x = np.array([r[variant][name]['out10'] for r in per])
            a = np.array([r[variant][name]['all45'] for r in per])
            tm = np.array([np.mean(r[variant][name]['triplet_max']) for r in per])
            S[name] = dict(out10_mean=float(x.mean()), out10_gain_vs_direct=float(1 - x.mean() / base.mean()),
                           out10_diff_ci=ci(base - x), all45_mean=float(a.mean()),
                           all45_diff_ci=ci(np.array([r[variant]['direct']['all45'] for r in per]) - a),
                           gross45=int(sum(r[variant][name]['gross45'] for r in per)),
                           triplet_max_mean=float(tm.mean()),
                           out10_by_phase={p: float(np.mean([r[variant][name]['out10_per_phase'][p] for r in per]))
                                           for p in PHASES if p != '10'})
        rw = np.array([r[variant]['residual_weighted']['out10'] for r in per])
        for name in ['holonomy_robust', 'combined', 'ls_sync']:
            x = np.array([r[variant][name]['out10'] for r in per])
            S[name]['vs_residual_weighted_diff'] = float((rw - x).mean())
            S[name]['vs_residual_weighted_ci'] = ci(rw - x)
        # localisation
        loc = {}
        for pred in ['cyclic', 'ncc', 'shift']:
            rhos = np.array([spearmanr(r[variant]['edge'][pred], r[variant]['edge']['true_err'])[0] for r in per])
            top1 = top3 = 0
            for r in per:
                te = np.array(r[variant]['edge']['true_err']); s = np.array(r[variant]['edge'][pred])
                worst = set(np.argsort(-te)[:3].tolist())
                top1 += int(np.argmax(s) == np.argmax(te)); top3 += int(np.argmax(s) in worst)
            allp = np.concatenate([r[variant]['edge'][pred] for r in per])
            allt = np.concatenate([r[variant]['edge']['true_err'] for r in per])
            loc[pred] = dict(within_spearman_mean=float(rhos.mean()), within_spearman_ci=ci(rhos),
                             auroc_gt2=auroc(allp, allt > 2), auroc_gt1=auroc(allp, allt > 1),
                             top1=top1, top3_contains_argmax=top3, rhos=rhos.tolist())
        cyc = np.array(loc['cyclic']['rhos'])
        for other in ['ncc', 'shift']:
            d = cyc - np.array(loc[other]['rhos'])
            loc['cyclic'][f'minus_{other}'] = float(d.mean()); loc['cyclic'][f'minus_{other}_ci'] = ci(d)
        allt = np.concatenate([r[variant]['edge']['true_err'] for r in per])
        loc['edges_gt2'] = int((allt > 2).sum()); loc['edges_gt1'] = int((allt > 1).sum()); loc['edges'] = int(allt.size)
        loc['chance_top1'] = L / 45; loc['chance_top3'] = 3 * L / 45
        # within pair identity: same phase pair, across landmarks (controls for which phases)
        wp = {}
        for pred in ['cyclic', 'ncc', 'shift']:
            rh = []
            for e in range(45):
                te = [r[variant]['edge']['true_err'][e] for r in per]; s = [r[variant]['edge'][pred][e] for r in per]
                rh.append(spearmanr(s, te)[0])
            wp[pred] = float(np.nanmean(rh))
        loc['within_pair_identity_spearman'] = wp
        S['localisation'] = loc
    # continuity with chapter 13 (triplet max-of-3, FFT, direct edges)
    old = {(int(r['landmark']), r['phases']): float(r['fft_max_tre_mm'])
           for r in csv.DictReader((PILOT / 'correspondence/results/cases.csv').open())}
    mine, theirs = [], []
    for r in per:
        for t, v in zip(TRIPLETS, r['fft']['direct']['triplet_max']):
            key = (r['landmark'], '/'.join(t))
            if key in old:
                mine.append(v); theirs.append(old[key])
    summary['continuity'] = dict(records=len(mine), chapter13_fft_mean=float(np.mean(theirs)),
                                 this_direct_mean=float(np.mean(mine)),
                                 max_abs_diff=float(np.max(np.abs(np.array(mine) - theirs))),
                                 note='Edge directions differ for pairs where the K10 index order reverses the triplet order; FFT is not exactly antisymmetric.')
    # annotation noise floor from trajectory smoothness
    ph = np.arange(10) * 2 * np.pi / 10
    X = np.stack([np.ones(10), np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)], 1)
    rms = []
    for r in per:
        k = r['landmark'] - 1
        T = np.array([P[p][k] for p in PHASES])
        resid = T - X @ np.linalg.lstsq(X, T, rcond=None)[0]
        rms.append(np.sqrt((resid ** 2).sum(axis=1).mean() * 10 / 5))
    summary['noise_floor'] = dict(point_rms_mm_median=float(np.median(rms)), point_rms_mm_mean=float(np.mean(rms)),
                                  implied_pair_floor_mm=float(np.sqrt(2) * np.median(rms)),
                                  note='3-D RMS per point; upper bound if real motion has more than 2 harmonics.')
    (HERE / 'results/per_landmark.json').write_text(json.dumps(per))
    (HERE / 'results/summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items()}, indent=1)[:6000])


if __name__ == '__main__':
    main()
