"""Build LABBOOK.html from the saved result files (no recomputation)."""
import json, html
from pathlib import Path

H = Path(__file__).resolve().parent
J = lambda p: json.loads((H / p).read_text())
S = J('results/summary.json'); LG = J('lagrangian/results/summary.json')
DG = J('results/diagnosis.json'); PL = J('results/posthoc_localisation.json'); SE = J('results/shared_error.json')
FZ = J('FREEZE_measure.json'); FE = J('FREEZE_evaluate.json'); FL = J('lagrangian/FREEZE.json')
f3 = lambda x: f'{x:.3f}'
pct = lambda x: f'{100 * x:.1f}%'
ci = lambda c: f'[{c[0]:.3f}, {c[1]:.3f}]'
NAMES = dict(direct='Direct phase-10 edges', ls_sync='Least-squares sync (all loops)', holonomy_robust='Holonomy-robust sync',
             residual_weighted='Residual-weighted sync (conventional)', combined='Combined')


def acc_table(summ, variant):
    rows = []
    for n in ['direct', 'ls_sync', 'holonomy_robust', 'residual_weighted', 'combined']:
        x = summ[variant][n]
        rows.append(f"<tr><td>{NAMES[n]}</td><td>{f3(x['out10_mean'])}</td><td>{pct(x['out10_gain_vs_direct'])}</td>"
                    f"<td>{ci(x['out10_diff_ci']) if n != 'direct' else '-'}</td><td>{f3(x['all45_mean'])}</td>"
                    f"<td>{ci(x['all45_diff_ci']) if n != 'direct' else '-'}</td><td>{f3(x['triplet_max_mean'])}</td><td>{x['gross45']}</td></tr>")
    return ('<table><thead><tr><th>Estimator</th><th>Phase-10 TRE, mm</th><th>vs direct</th><th>Paired gain CI, mm</th>'
            '<th>All-45 TRE, mm</th><th>All-45 gain CI</th><th>Triplet max-of-3, mm</th><th>Edges &gt;2 mm (of 1260)</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def loc_table(summ, variant, ph):
    L = summ[variant]['localisation']; rows = []
    for p, lab in [('cyclic', 'Loop (cyclic) remainder'), ('ncc', 'NCC image residual'), ('shift', 'Measured shift size')]:
        x = L[p]
        rows.append(f"<tr><td>{lab}</td><td>{f3(x['within_spearman_mean'])} {ci(x['within_spearman_ci'])}</td>"
                    f"<td>{f3(L['within_pair_identity_spearman'][p])}</td><td>{f3(x['auroc_gt2'])}</td><td>{f3(x['auroc_gt1'])}</td>"
                    f"<td>{ph['top1'][p]} / {summ['landmarks']}</td></tr>")
    return ('<table><thead><tr><th>Predictor of true edge error</th><th>Within-landmark Spearman [CI]</th><th>Same pair, across landmarks</th>'
            '<th>AUROC &gt;2 mm</th><th>AUROC &gt;1 mm</th><th>Finds worst edge (top-1)</th></tr></thead><tbody>' + ''.join(rows)
            + f'</tbody></table><p class="note">Chance top-1 is {L["chance_top1"]:.2f} of {summ["landmarks"]}. Loop minus NCC top-1, landmark bootstrap CI: {ph["top1_cyc_minus_ncc_ci"]}.</p>')


def bar_svg():
    items = [('Chance (iid edge noise on K10)', 0.2, True)] + [
        (f'{run}, {v}', val, False) for run, v, val in [
            ('Eulerian', 'FFT', DG['t']['integrable_share_pooled']), ('Eulerian', 'refined', DG['t_refined']['integrable_share_pooled'])]]
    W, rowh, left = 640, 34, 230
    out = [f'<svg viewBox="0 0 {W} {rowh * len(items) + 40}" role="img" aria-label="Integrable share of error energy">']
    for k, (lab, v, ref) in enumerate(items):
        y = 10 + k * rowh; w = (W - left - 60) * v
        cls = 'bar ref' if ref else 'bar'
        out.append(f'<text x="{left - 8}" y="{y + 17}" text-anchor="end" class="lab">{html.escape(lab)}</text>'
                   f'<rect class="{cls}" x="{left}" y="{y + 4}" width="{w:.1f}" height="20" rx="4"><title>{html.escape(lab)}: {pct(v)}</title></rect>'
                   f'<text x="{left + w + 6:.1f}" y="{y + 19}" class="val">{pct(v)}</text>')
    y0 = 10 + len(items) * rowh
    for t in [0, .25, .5, .75, 1]:
        x = left + (W - left - 60) * t
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="8" y2="{y0}" class="grid"/><text x="{x:.1f}" y="{y0 + 16}" text-anchor="middle" class="tick">{int(t * 100)}%</text>')
    out.append('</svg>')
    return ''.join(out)


def lag_svg():
    sh = SE['refined']['shared_phase_lag_corr']; df = SE['refined']['difference_phase_lag_corr']
    W, Hh, l, r, t, b = 640, 260, 60, 150, 16, 40
    X = lambda d: l + (int(d) - 1) / 4 * (W - l - r)
    Y = lambda v: t + (0.5 - v) / 1.0 * (Hh - t - b)
    o = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Phase-lag correlation">']
    for v in [-0.5, -0.25, 0, 0.25, 0.5]:
        o.append(f'<line x1="{l}" x2="{W - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid{" zero" if v == 0 else ""}"/><text x="{l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="tick">{v:+.2f}</text>')
    for d in range(1, 6):
        o.append(f'<text x="{X(d):.1f}" y="{Hh - b + 18}" text-anchor="middle" class="tick">{d}</text>')
    o.append(f'<text x="{(l + W - r) / 2:.0f}" y="{Hh - 4}" text-anchor="middle" class="tick">phase separation (steps of 10% of the breathing cycle)</text>')
    o.append(f'<line x1="{l}" x2="{W - r}" y1="{Y(-1 / 8):.1f}" y2="{Y(-1 / 8):.1f}" class="iid"/><line x1="{W - r + 10}" x2="{W - r + 28}" y1="{Y(-0.42):.1f}" y2="{Y(-0.42):.1f}" class="iid"/><text x="{W - r + 32}" y="{Y(-0.42) + 4:.1f}" class="lab small">independent</text><text x="{W - r + 32}" y="{Y(-0.42) + 18:.1f}" class="lab small">jitter, -0.125</text>')
    for series, cls, lab in [(sh, 's1', 'shared error'), (df, 's2', 'instrument difference')]:
        pts = ' '.join(f'{X(d):.1f},{Y(v):.1f}' for d, v in series.items())
        o.append(f'<polyline points="{pts}" class="line {cls}"/>')
        for d, v in series.items():
            o.append(f'<circle cx="{X(d):.1f}" cy="{Y(v):.1f}" r="5" class="dot {cls}"><title>{lab}, separation {d}: {v:+.3f}</title></circle>')
        last = list(series.items())[-1]
        o.append(f'<text x="{X(last[0]) + 10:.1f}" y="{Y(last[1]) + 4:.1f}" class="lab small">{lab}</text>')
    o.append('</svg>')
    return ''.join(o)


s = S['fft']; r = S['refined']; sef = SE['refined']; ag = SE['annotation_grid']
DL = J('dirlab/results/summary.json'); ND = J('paper/numbers_demons.json'); DLA = J('dirlab/AMENDMENT_1.json'); DLF = J('dirlab/PREREG.freeze.json'); DLS = J('dirlab/results/SEAL.json')


def dl_table(key):
    v = DL['cases'][key]; rows = []
    for c in range(1, 11):
        x = v['per_case'][str(c)]
        if not x['estimable']:
            rows.append(f'<tr><td>{c}</td><td>{x["n"]}</td><td colspan="6">not estimable</td></tr>'); continue
        rows.append(f"<tr><td>{c}</td><td>{x['n']}</td><td>{x['estimate_mean_norm_mm']:.2f}</td><td>{x['published_mm']:.2f}</td>"
                    f"<td>{x['ratio']:.2f}</td><td>{'yes' if x['in_band'] else 'no'}</td><td>{x['tre_A']:.2f}</td><td>{x['tre_B']:.2f}</td></tr>")
    return ('<table><thead><tr><th>Case</th><th>n</th><th>Estimate, mm</th><th>Published observer, mm</th><th>Ratio</th><th>In band</th>'
            '<th>Window TRE, mm</th><th>Demons TRE, mm</th></tr></thead><tbody>' + ''.join(rows) +
            f"</tbody></table><p class='note'>{v['in_band']} of {v['estimable']} estimable cases in band; pooled ratio {v['pooled_ratio']:.3f}; "
            f"Spearman of estimate with published observer error {v['spearman_vs_published']:.2f}, with case displacement {v['spearman_vs_displacement']:.2f}. Verdict: {v['verdict']}.</p>")


def dl_variants():
    rows = []
    for k, lab in [('primary_refined', 'Primary (frozen guard), refined window'), ('primary_fft', 'Frozen guard, FFT window'),
                   ('primary_refined_drop3mm', 'Frozen guard, drop |A-B| > 3 mm'), ('S1_refined', 'S1 (no envelope margin), refined'),
                   ('S1_refined_drop3mm', 'S1, drop |A-B| > 3 mm')]:
        v = DL['cases'][k]
        rows.append(f"<tr><td>{lab}</td><td>{v['in_band']} / {v['estimable']}</td><td>{v['pooled_ratio']:.3f}</td><td>{v['spearman_vs_displacement']:.2f}</td><td>{v['verdict']}</td></tr>")
    return ('<table><thead><tr><th>Variant</th><th>Cases in band</th><th>Pooled ratio</th><th>Spearman with displacement</th><th>Verdict</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-cycle holonomy lab book</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--s1:#2a78d6;--s2:#eb6834;--card:#f4f3f0}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835;--s1:#3987e5;--s2:#d95926;--card:#232321}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835;--s1:#3987e5;--s2:#d95926;--card:#232321}}
body{{background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,system-ui,sans-serif;margin:0;padding:0 16px}}
main{{max-width:980px;margin:32px auto 80px}} h1{{font-size:28px;margin:0 0 4px}} h2{{margin-top:44px;border-top:1px solid var(--rule);padding-top:20px}}
.sub{{color:var(--ink2)}} .note{{color:var(--ink2);font-size:14px}} code{{font-size:14px}}
.box{{background:var(--card);border-radius:8px;padding:14px 18px;margin:16px 0}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:14px;margin:12px 0;min-width:640px}} th,td{{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--ink2);font-weight:600}}
svg{{width:100%;height:auto;max-width:720px;display:block;margin:8px 0}} .lab{{fill:var(--ink);font-size:13px}} .small{{font-size:12px}} .val,.tick{{fill:var(--ink2);font-size:12px}}
.grid{{stroke:var(--rule);stroke-width:1}} .zero{{stroke:var(--ink2)}} .bar{{fill:var(--s1)}} .bar.ref{{fill:var(--ink2);opacity:.45}}
.line{{fill:none;stroke-width:2}} .line.s1{{stroke:var(--s1)}} .line.s2{{stroke:var(--s2)}} .dot{{stroke:var(--bg);stroke-width:2}} .dot.s1{{fill:var(--s1)}} .dot.s2{{fill:var(--s2)}}
.iid{{stroke:var(--ink2);stroke-dasharray:4 4;stroke-width:1.5}}
</style></head><body><main>
<h1>Multi-cycle holonomy on POPI</h1>
<p class="sub">Lab book, 19-20 November 2026. Follow-up to <code>../dvc-holonomy-pilot</code> (that folder is imported read-only and was not modified). One exposed patient, 28 landmarks, ten breathing phases.</p>

<div class="box"><b>Findings in order of weight</b>
<ol>
<li><b>Visibility ceiling.</b> {pct(DG['t']['integrable_share_pooled'])} of the registration error energy is integrable on the 10-phase graph (median landmark {pct(DG['t']['integrable_share_median'])}); independent edge noise would be 20%. Loops cannot see integrable error, so in this setting holonomy can see at most about 7% of the error energy.</li>
<li><b>Most of the remaining error is shared by two unrelated instruments.</b> The 32-voxel translation window and whole-volume Diffeomorphic Demons have error vectors with per-axis correlation {', '.join(f'{v:.2f}' for v in sef['error_corr_zyx'])} (z, y, x). Shared part: {f3(sef['shared_rms_3d'])} mm RMS. Each instrument's own part: {f3((sum(v*v for v in sef['own_sd_zyx_instrument_1']))**.5)} mm (refined window), {f3((sum(v*v for v in sef['own_sd_zyx_demons']))**.5)} mm (Demons).</li>
<li><b>The shared part behaves like annotation jitter.</b> Its phase-to-phase correlation is about -0.13 at every separation, which is what independent per-point noise gives after demeaning (-0.125). {ag['z_on_2mm_slice_grid']} of {ag['z_points']} landmark z values sit on the 2 mm slice grid; snapping alone predicts {f3(ag['z_slice_snap_pair_sd_mm'])} mm pair SD in z, and the shared z SD is {f3(sef['shared_sd_zyx'][0])} mm.</li>
<li><b>Inside its window, the loop residual localises.</b> It names the truly worst edge in {PL['eulerian_fft']['top1']['cyclic']} to {PL['lagrangian_refined']['top1']['cyclic']} of 28 landmarks across the four runs, versus 0 to 1 for the image residual and 0.6 by chance. It also ranks landmarks for a given phase pair better than the image residual (Spearman {f3(s['localisation']['within_pair_identity_spearman']['cyclic'])} vs {f3(s['localisation']['within_pair_identity_spearman']['ncc'])}). The declared localisation criterion (within-landmark Spearman above both comparators) was not met: NCC {f3(s['localisation']['ncc']['within_spearman_mean'])}, loop {f3(s['localisation']['cyclic']['within_spearman_mean'])}.</li>
<li><b>DIR-Lab known-answer test: FAIL (pre-registered).</b> Reading registrar covariance as annotation noise did not reproduce DIR-Lab's published repeat-observer error: estimates ran 1.3 to 5.2 times the published value, where the prediction was 0.71 times (pooled ratio {DL['cases']['primary_refined']['pooled_ratio']:.2f} against a band of 0.94 to 2.12; {DL['cases']['primary_refined']['in_band']} of {DL['cases']['primary_refined']['estimable']} cases in band) and tracked motion size. On large-motion extreme phases both registrars share a tracking error, so the shared term is not truth noise there. The POPI attribution in items 2 and 3 therefore rests on its own fingerprints, not on a validated estimator.</li>
<li><b>Visibility ceiling replicates on DIR-Lab:</b> {pct(DL['visibility_ceiling_six_phase']['t_refined']['integrable_share'])} of window-instrument error is loop-invisible on 477 six-phase graphs across 10 patients (chance 33%).</li>
<li><b>Synchronisation does not improve the phase-10 estimates</b> ({pct(s['holonomy_robust']['out10_gain_vs_direct'])} for holonomy-robust sync, CI {ci(s['holonomy_robust']['out10_diff_ci'])} mm). It improves all-pairs TRE by 2 to 3% and triplet max-of-3 by about 5%. Holonomy-robust sync beats residual-weighted sync by {f3(s['holonomy_robust']['vs_residual_weighted_diff'])} mm, CI {ci(s['holonomy_robust']['vs_residual_weighted_ci'])}.</li>
</ol></div>

<h2>1. Question</h2>
<p>Every chapter of the pilot used three observations per query: one independent loop. One loop can flag an inconsistency but cannot say which edge is wrong. POPI has ten phases. The complete graph on ten phases has 45 edges and 36 independent cycles per axis, so the Hodge solver can both synchronise (replace the 45 edges with their integrable part) and localise (each edge's cyclic remainder). This was untested.</p>

<h2>2. Method</h2>
<p>Instrument unchanged and imported from the pilot: tapered FFT translation on the canonical 32-voxel normalised window, Powell NCC refinement as the secondary variant, the interior/texture guards, and the pilot's <code>dh.hodge.decompose</code>. Same integer centre (the phase-10 landmark) in all ten phases, as in chapter 13. All 45 directed edges i&lt;j measured for each of the 28 eligible landmarks (the same 28 as chapter 13; of the other 12, 9 fail the frozen interior guard in at least one phase and 3 fail the texture check). Measurement never reads target landmarks.</p>
<p>Order of record: <code>PLAN.json</code> and <code>measure.py</code> hashed at {FZ['frozen_utc']}; <code>evaluate.py</code> and the measurements hashed at {FE['frozen_utc']}, before its first run; the Lagrangian addendum hashed at {FL['frozen_utc']}, before its run. <code>diagnose.py</code> and <code>shared_error.py</code> are post-hoc, written after the declared results were seen. POPI was already exposed by the pilot, so everything here is exploratory.</p>
<p class="note">Continuity check: the direct edges reproduce chapter 13's FFT triplet figure ({f3(S['continuity']['chapter13_fft_mean'])} mm) on all {S['continuity']['records']} records, max difference {S['continuity']['max_abs_diff']:.1e} mm. Hodge orthogonality error at most {S['fft']['hodge_orthogonality_max']:.1e}.</p>

<h2>3. Declared results</h2>
<h3>Accuracy, FFT edges</h3><div class="tw">{acc_table(S, 'fft')}</div>
<h3>Accuracy, refined edges</h3><div class="tw">{acc_table(S, 'refined')}</div>
<p class="note">No-registration baseline on the same 28 landmarks: {f3(S['identity_out10'])} mm phase-10 TRE, {f3(S['identity_all45'])} mm all-45. Confidence intervals: 4000 landmark-cluster resamples, paired, conditional on one patient.</p>
<h3>Localisation, FFT edges</h3><div class="tw">{loc_table(S, 'fft', PL['eulerian_fft'])}</div>
<h3>Localisation, refined edges</h3><div class="tw">{loc_table(S, 'refined', PL['eulerian_refined'])}</div>

<h2>4. Why the loops see so little (post-hoc)</h2>
<p>Truth is a set of node positions, so it is exactly integrable. The cyclic part of a measurement is therefore exactly the cyclic part of its error. Hodge-splitting the true error gives the share no loop can see.</p>
{bar_svg()}
<p>Per edge, the median gross edge (&gt;2 mm) has {pct(PL['eulerian_fft']['gross_edge_cyclic_share_median'])} of its error in the cyclic part; {PL['eulerian_fft']['gross_edges_mostly_cyclic']} of {PL['eulerian_fft']['gross_edges']} gross edges are mostly cyclic. The worst edge per landmark averages {f3(PL['eulerian_fft']['worst_edge_err_mean'])} mm with a median cyclic share of {pct(PL['eulerian_fft']['worst_edge_cyclic_share_median'])}. The top-1 hits in section 3 come from those rare isolated failures.</p>
<p><b>Estimator shrinkage, truth-free.</b> Integer-voxel crop offsets of the phase-10 volume give exact known shifts ({DG['known_shift']['n']} tests on the 28 landmark windows). Measured/applied gain z, y, x: FFT {', '.join(f'{v:.3f}' for v in DG['known_shift']['gain_fft_zyx'])}; refined {', '.join(f'{v:.3f}' for v in DG['known_shift']['gain_refined_zyx'])}. Refinement recovers small shifts exactly and falls back to FFT beyond its 4-voxel support. Against landmarks, the node-error gain is lower (FFT {', '.join(f'{v:.2f}' for v in DG['t']['node_error_gain_zyx'])}), and a per-axis gain explains only {pct(DG['t']['node_error_energy_explained_by_gain'])} of node-error energy. Estimator shrinkage is a small part of the integrable error.</p>

<h2>5. Lagrangian re-centring (declared before its run)</h2>
<p>Hypothesis: in the Eulerian design an edge such as 50 to 60 is measured on whatever tissue sits at the phase-10 position, not on the landmark. Each edge (i, j) was re-measured with both windows centred at the estimated landmark position in phase i (from the direct FFT star). All 1260 edges were re-centred.</p>
<div class="tw">{acc_table(LG, 'fft')}</div>
<p>All-45 TRE moves from {f3(S['fft']['direct']['all45_mean'])} to {f3(LG['fft']['direct']['all45_mean'])} mm; the integrable share stays at about 93%. The wrong-tissue effect is minor. The declared alternative reading holds: the integrable error is mostly window-versus-landmark tracking difference plus annotation noise.</p>

<h2>6. Shared error between two unrelated instruments (post-hoc)</h2>
<p>Instrument 2 is the pilot's chapter-15 whole-volume Diffeomorphic Demons (stored 2 mm-level predictions, not refitted). Comparison on the 252 phase-10 star edges.</p>
<div class="tw"><table><thead><tr><th>Quantity</th><th>z</th><th>y</th><th>x</th></tr></thead><tbody>
<tr><td>Error correlation, refined window vs Demons</td>{''.join(f'<td>{v:.3f}</td>' for v in sef['error_corr_zyx'])}</tr>
<tr><td>Shared SD, mm</td>{''.join(f'<td>{v:.3f}</td>' for v in sef['shared_sd_zyx'])}</tr>
<tr><td>Refined window own SD, mm</td>{''.join(f'<td>{v:.3f}</td>' for v in sef['own_sd_zyx_instrument_1'])}</tr>
<tr><td>Demons own SD, mm</td>{''.join(f'<td>{v:.3f}</td>' for v in sef['own_sd_zyx_demons'])}</tr>
</tbody></table></div>
<p>Mean TRE: refined window {f3(sef['tre_mean']['instrument_1'])}, Demons {f3(sef['tre_mean']['demons'])}, average of the two {f3(sef['tre_mean']['average_of_two'])} mm. Averaging two instruments whose error vectors have mean cosine {f3(sef['error_cosine_mean'])} recovers almost nothing.</p>
<p><b>Structure across the breathing cycle.</b> After removing each landmark's mean over phases (which also removes the phase-10 annotation term), the shared error's correlation between phases sits near -0.125 at every separation, the value independent per-point noise gives. The difference between the instruments is smooth and periodic, as a tracking difference would be.</p>
{lag_svg()}
<div class="tw"><table><thead><tr><th>Phase separation</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th></tr></thead><tbody>
<tr><td>Shared error</td>{''.join(f'<td>{v:+.3f}</td>' for v in sef['shared_phase_lag_corr'].values())}</tr>
<tr><td>Instrument difference</td>{''.join(f'<td>{v:+.3f}</td>' for v in sef['difference_phase_lag_corr'].values())}</tr></tbody></table></div>
<p>The shared error's size correlates with the image residual (Spearman {f3(sef['shared_abs_vs_ncc_residual_spearman'])}) and with true motion ({f3(sef['shared_abs_vs_true_motion_spearman'])}), so part of it is image ambiguity that both registrars inherit. In z, its size matches slice snapping of the annotations: {ag['z_on_2mm_slice_grid']} of {ag['z_points']} z coordinates are on the 2 mm grid, predicting {f3(ag['z_slice_snap_pair_sd_mm'])} mm pair SD; observed shared z SD {f3(sef['shared_sd_zyx'][0])} mm.</p>

<h2>6b. DIR-Lab known-answer test (pre-registered)</h2>
<p>Question: does the covariance of two registrars' errors estimate a benchmark's annotation noise? DIR-Lab 4DCT publishes, per case, the repeat-registration error of three observers, which is an independent known answer. <code>dirlab/PREREG.json</code> was frozen at {DLF['frozen_utc']} with no DIR-Lab data on the machine. Predicted ratio published/estimate = 1.41; PASS if at least 7 cases and the pooled ratio fall in [0.94, 2.12]. <code>AMENDMENT_1.json</code> was recorded after unpacking and before measurement: conventions (1-based integer voxel landmarks, no flip, stored = HU + 1024), the frozen guard leaving case 1 with one landmark (not estimable), and a declared sensitivity set S1 without the body-envelope margin. Measurements were sealed at {DLS['sealed_utc']} before any T50 landmark was read; the scorer was frozen before the seal.</p>
<h3>Primary</h3><div class="tw">{dl_table('primary_refined')}</div>
<h3>All variants</h3><div class="tw">{dl_variants()}</div>
<p>Reading: the estimate exceeds the published observer error everywhere except case 4 in the two frozen-guard drop-3 mm variants, and in S1 it follows case displacement (Spearman {DL['cases']['S1_refined']['spearman_vs_displacement']:.2f}). Both registrars run well above state-of-the-art accuracy on these extreme-phase pairs (window TRE up to about 8 mm, Demons up to about 10 mm), and the shared part sits mainly in z. That matches the first and third falsifiers named in advance: shared image-driven error and motion-driven registration failure. Dropping landmarks where the instruments disagree by more than 3 mm brings 9 of 10 cases (S1) to 0.9 to 1.9 mm and leaves case 7 at 5.3 mm; still outside the band overall.</p>
<p>Operating window this maps: registrar covariance reads as truth noise only where both registrars' own errors are small and their biases differ. These two registrars on DIR-Lab extreme phases are outside that window. The phase-independence secondary was not run (Demons was fitted T00 to T50 only, and the T10 to T40 points are model-propagated).</p>
<p><b>Demons on all 15 pairs (ADDENDUM_2, 2b):</b> whole-volume Diffeomorphic Demons fitted independently on every phase pair of all 10 cases (150 fits, all with positive interior Jacobians; resumed after an out-of-memory crash with a declared memory-lean script). Invisible share {pct(ND['all']['integrable_share'])} over {ND['all']['landmarks']} landmarks (case-clustered 95% {ND['all']['ci95_case_cluster'][0]*100:.1f} to {ND['all']['ci95_case_cluster'][1]*100:.1f}%), {pct(ND['all']['truth_noise_adjusted_share'])} after the pessimistic annotation-noise allowance; mean edge error {ND['all']['mean_edge_tre_mm']:.2f} mm. Write-up: <code>paper/visibility_ceiling.html</code>.</p>
<p><b>Visibility ceiling, replicated:</b> on the 75-landmark six-phase graphs (K6, 15 edges, chance integrable share 33%), {pct(DL['visibility_ceiling_six_phase']['t_refined']['integrable_share'])} of the refined window's error energy is integrable ({pct(DL['visibility_ceiling_six_phase']['t']['integrable_share'])} for FFT), over {DL['visibility_ceiling_six_phase']['t_refined']['landmarks']} landmarks in 10 patients.</p>

<h2>7. Reading the pilot chapters against these measurements</h2>
<ul>
<li>Chapters before 13 (DVC foam scans, acceptance and interior studies, known-motion controls): one loop per query, and the pilot's own calibration found consistent-but-wrong taper bias. That is integrable error, invisible to loops.</li>
<li>Chapter 13 (POPI transfer): direct FFT phase-10 TRE {f3(S['fft']['direct']['out10_mean'])} mm; about 1.05 mm RMS of the error per edge is shared with an unrelated instrument.</li>
<li>Chapter 15 (Demons, "18.32% oracle headroom"): the oracle chose between candidates whose star-edge errors correlate at 0.77 to 0.90 per axis, scored against points with slice-snapped z. The selection gain is measured against that truth.</li>
</ul>

<h2>8. Operating window</h2>
<p>The loop residual measures edge-specific inconsistency. Measured here: it finds isolated single-edge failures (top-1 worst edge several times above image residual and chance) and ranks landmarks within a phase pair. It carries no information about error shared by all edges of a node: window-versus-landmark tracking difference, estimator gain, or truth noise. On POPI that shared class is about 93% of the error energy, and about 1 mm RMS of it is common to two unrelated registrars.</p>

<h2>9. Limits</h2>
<ul><li>One patient, 28 landmarks, already exposed. All CIs are conditional on this patient.</li>
<li>The POPI shared-error attribution to annotation is by fingerprint (phase independence, z grid), not by repeat annotation. Image ambiguity common to both registrars is a co-explanation for part of it. The general estimator (covariance = truth noise) failed its pre-registered known-answer test on DIR-Lab (section 6b).</li>
<li>Demons predictions are the pilot's stored values; Demons was run on a finite iteration budget.</li>
<li>The Lagrangian centres come from FFT star estimates and inherit their error.</li></ul>

<h2>10. Files</h2>
<p><code>PLAN.json</code>, <code>measure.py</code>, <code>evaluate.py</code>, <code>FREEZE_*.json</code>; <code>diagnose.py</code>, <code>shared_error.py</code> (post-hoc); <code>lagrangian/</code> (addendum, measurement, summary); <code>results/</code> (measurements, summary, per-landmark, diagnosis, shared error, localisation). Rebuild this page with <code>python3 build_labbook.py</code>.</p>
</main></body></html>"""
(H / 'LABBOOK.html').write_text(page)
print('wrote', len(page))
