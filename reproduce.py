"""Recompute every headline number in the note from derived/ and compare with results/.

Needs only numpy and the standard library; no images or landmark files. Point estimates must
match to 1e-6 (absolute, in share units); bootstrap intervals are recomputed with this script's
own resampling and must lie within 1.5 percentage points of the reported ones.
Run: python reproduce.py
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from instrument.hodge_split import split, chance_invisible_share  # noqa: E402

RES = ROOT / 'results' / 'paper'
N = json.loads((RES / 'numbers.json').read_text())
ND = json.loads((RES / 'numbers_demons.json').read_text())
LOC = json.loads((RES / 'dirlab_localisation.json').read_text())
EX = json.loads((RES / 'extras.json').read_text())
checks = []


def check(name, got, want, tol=1e-6):
    ok = want is None and got is None or (got is not None and want is not None and abs(got - want) <= tol)
    checks.append((name, ok, got, want))


def rows(name, **filt):
    with (ROOT / 'derived' / name).open() as f:
        out = [r for r in csv.DictReader(f) if all(r[k] == v for k, v in filt.items())]
    return out


def pooled(rs, adjust=False):
    G = np.array([float(r['E_integrable_mm2']) for r in rs]); C = np.array([float(r['E_cyclic_mm2']) for r in rs])
    if adjust:
        Nn = np.array([float(r['noise_allowance_mm2']) for r in rs]); return (G.sum() - Nn.sum()) / (G.sum() + C.sum() - Nn.sum())
    return G.sum() / (G.sum() + C.sum())


def case_boot(rs, adjust=False, B=4000, seed=7):
    by = defaultdict(list)
    for r in rs:
        by[r['case']].append(r)
    keys = sorted(by); rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys))
        vals.append(pooled([r for i in pick for r in by[keys[i]]], adjust))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# 1. independent-error baseline 2/n (simulation, 2000 graphs each)
rng = np.random.default_rng(11)
for n in [3, 6, 10]:
    m = n * (n - 1) // 2; s = []
    for _ in range(2000):
        g, c = split(rng.standard_normal((m, 3)), n); s.append((g ** 2).sum() / ((g ** 2).sum() + (c ** 2).sum()))
    check(f'baseline K{n} simulated vs 2/n', float(np.mean(s)), chance_invisible_share(n), tol=0.01)

# 2. DIR-Lab window registrar
for reg, key in [('window_refined', 'dirlab_t_refined'), ('window_fft', 'dirlab_t')]:
    rs = rows('dirlab_window_per_landmark.csv', registrar=reg); ref = N[key]
    check(f'{reg} landmarks', len(rs), ref['landmarks'], 0)
    check(f'{reg} invisible share', pooled(rs), ref['integrable_share'])
    check(f'{reg} after noise allowance', pooled(rs, True), ref['truth_noise_adjusted_share'])
    G = np.array([float(r['E_integrable_mm2']) for r in rs]); C = np.array([float(r['E_cyclic_mm2']) for r in rs])
    check(f'{reg} per-landmark median', float(np.median(G / (G + C))), ref['per_landmark_median'])
    for c, v in ref['by_case'].items():
        check(f'{reg} case {c}', pooled([r for r in rs if r['case'] == c]), v['share'])
    for (lo, hi), (rlo, rhi) in [(case_boot(rs), ref['ci95_case_cluster'])]:
        check(f'{reg} CI low', lo, rlo, 0.015); check(f'{reg} CI high', hi, rhi, 0.015)
    if reg == 'window_refined':
        v = np.sort(C)[::-1]; check('window visible energy, top 10% share', float(v[:len(v) // 10].sum() / v.sum()), EX['window_visible_energy_top10_share'])

# 3. DIR-Lab Demons
for sub, filt in [('all', {}), ('matched', {'in_window_set': '1'})]:
    rs = rows('dirlab_demons_per_landmark.csv', **filt); ref = ND[sub]
    check(f'demons {sub} landmarks', len(rs), ref['landmarks'], 0)
    check(f'demons {sub} invisible share', pooled(rs), ref['integrable_share'])
    check(f'demons {sub} after noise allowance', pooled(rs, True), ref['truth_noise_adjusted_share'])
    check(f'demons {sub} mean edge error', float(np.mean([float(r['mean_edge_error_mm']) for r in rs])), ref['mean_edge_tre_mm'])
    lo, hi = case_boot(rs); check(f'demons {sub} CI low', lo, ref['ci95_case_cluster'][0], 0.015); check(f'demons {sub} CI high', hi, ref['ci95_case_cluster'][1], 0.015)
for c, v in ND['all']['by_case'].items():
    check(f'demons case {c}', pooled(rows('dirlab_demons_per_landmark.csv', case=c)), v['share'])
check('window mean edge error', float(np.mean([float(r['mean_edge_error_mm']) for r in rows('dirlab_window_per_landmark.csv', registrar='window_refined')])),
      ND['window_refined_mean_edge_tre_mm'])

# 4. POPI window registrar (allowance needs an assumed d; POPI publishes none)
for reg, key in [('window_refined', 'popi_t_refined'), ('window_fft', 'popi_t')]:
    rs = rows('popi_window_per_landmark.csv', registrar=reg); ref = N[key]
    G = sum(float(r['E_integrable_mm2']) for r in rs); C = sum(float(r['E_cyclic_mm2']) for r in rs)
    check(f'POPI {reg} invisible share', G / (G + C), ref['integrable_share'])
    for d, want in ref['truth_noise_adjusted_share'].items():
        Nn = 2.356 * float(d) ** 2 * 45 * len(rs)
        check(f'POPI {reg} after allowance d={d}', (G - Nn) / (G + C - Nn) if G + C > Nn else None, want)

# 5. worst-edge localisation, DIR-Lab window
for reg, key in [('window_refined', 't_refined'), ('window_fft', 't')]:
    rs = rows('dirlab_window_worst_edge.csv', registrar=reg)
    for p in ['cyclic', 'ncc', 'shift']:
        check(f'{reg} worst edge named by {p}', float(np.mean([int(r[f'worst_named_by_{p}']) for r in rs])), LOC[key][p]['top1_rate'])
        # Rank correlations get 1e-3: FFT estimates on a 1/20-voxel grid often close loops exactly, so several
        # cyclic remainders are ~1e-15 and their order depends on solver rounding (this repo's solver vs the study's).
        check(f'{reg} within-landmark Spearman {p}', float(np.mean([float(r[f'spearman_{p}']) for r in rs])), LOC[key][p]['within_spearman'], 1e-3)

w = max(len(c[0]) for c in checks)
for name, ok, got, want in checks:
    print(f"{'ok  ' if ok else 'FAIL'} {name:<{w}}  {got}  (reported {want})")
bad = [c for c in checks if not c[1]]
print(f"\n{len(checks) - len(bad)} of {len(checks)} checks match." + (' ALL MATCH' if not bad else ''))
sys.exit(1 if bad else 0)
