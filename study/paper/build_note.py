"""Build visibility_ceiling.html from saved numbers and figures (no recomputation)."""
import json, html
from pathlib import Path

P = Path(__file__).resolve().parent; R = P.parent
J = lambda p: json.loads((R / p).read_text())
N = J('paper/numbers.json'); LOCD = J('paper/dirlab_localisation.json'); S = J('results/summary.json')
import os
NO_CT = os.environ.get('LV_NO_CT') == '1'
F = J('paper/figs_noct.json' if NO_CT else 'paper/figs.json'); ND = J('paper/numbers_demons.json'); da, dm = ND['all'], ND['matched']
PL = J('results/posthoc_localisation.json'); DG = J('results/diagnosis.json'); SE = J('results/shared_error.json')
DL = J('dirlab/results/summary.json'); LG = J('lagrangian/results/summary.json')
pc = lambda x: f'{100 * x:.1f}%'
rng = lambda c: f'{100 * c[0]:.1f} to {100 * c[1]:.1f}%'
dr, df = N['dirlab_t_refined'], N['dirlab_t']
pf, prf = N['popi_t'], N['popi_t_refined']
ld = LOCD['t_refined']; TM = F['trajectory_meta']; RU = F['rules']
EX = J('paper/extras.json'); VIS_TOP10, TRUE5, EST5 = EX['window_visible_energy_top10_share'], EX['typical_true_T00_T50_mm'], EX['typical_est_T00_T50_mm']


def case_figure():
    rows = []
    for c in range(1, 11):
        rows.append((f'Case {c}', dr['by_case'][str(c)]['share'], da['by_case'][str(c)]['share'], 1 / 3,
                     dr['by_case'][str(c)]['n'], da['by_case'][str(c)]['n']))
    rows.append(('DIR-Lab pooled', dr['integrable_share'], da['integrable_share'], 1 / 3, dr['landmarks'], da['landmarks']))
    rows.append(('POPI pooled', prf['integrable_share'], None, 0.2, prf['landmarks'], None))
    W, rh, left, right = 680, 34, 120, 70
    plot = W - left - right; X = lambda v: left + plot * v
    h = 30 + rh * len(rows) + 40
    o = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Loop-invisible share of error by case and registrar">',
         '<rect x="20" y="6" width="12" height="12" rx="2" class="bar"/><text x="38" y="16" class="lab">32-voxel translation window</text>'
         '<rect x="245" y="6" width="12" height="12" rx="2" class="barD"/><text x="263" y="16" class="lab">whole-volume Demons</text>'
         '<line x1="430" x2="430" y1="4" y2="20" class="chance"/><text x="438" y="16" class="lab">independent-error baseline</text>']
    y0 = 30
    for t in [0, .25, .5, .75, 1]:
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{y0}" y2="{h - 34}" class="grid"/><text x="{X(t):.1f}" y="{h - 20}" class="tick" text-anchor="middle">{int(t * 100)}%</text>')
    o.append(f'<text x="{left + plot / 2:.0f}" y="{h - 4}" class="tick" text-anchor="middle">share of error energy invisible to every loop</text>')
    for k, (lab, vw, vd, ch, nw, nd) in enumerate(rows):
        y = y0 + k * rh; strong = 'pooled' in lab
        o.append(f'<text x="{left - 8}" y="{y + 18}" text-anchor="end" class="lab{" strong" if strong else ""}">{html.escape(lab)}</text>')
        o.append(f'<rect x="{left}" y="{y + 3}" width="{plot * vw:.1f}" height="12" rx="3" class="bar"><title>{html.escape(lab)}, window (n={nw}): {pc(vw)}</title></rect>'
                 f'<text x="{X(vw) + 5:.1f}" y="{y + 13}" class="val">{pc(vw)}</text>')
        if vd is not None:
            o.append(f'<rect x="{left}" y="{y + 17}" width="{plot * vd:.1f}" height="12" rx="3" class="barD"><title>{html.escape(lab)}, Demons (n={nd}): {pc(vd)}</title></rect>'
                     f'<text x="{X(vd) + 5:.1f}" y="{y + 27}" class="val">{pc(vd)}</text>')
        o.append(f'<line x1="{X(ch):.1f}" x2="{X(ch):.1f}" y1="{y + 1}" y2="{y + 31}" class="chance"/>')
    o.append('</svg>')
    return ''.join(o)


cases_tbl = ''.join(f"<tr><td>{c}</td><td>{v['n']}</td><td>{pc(v['share'])}</td><td>{pc(v['adjusted'])}</td><td>{da['by_case'][c]['n']}</td><td>{pc(da['by_case'][c]['share'])}</td></tr>"
                    for c, v in dr['by_case'].items())
est_ratio = [1 / x['ratio'] for x in DL['cases']['primary_refined']['per_case'].values() if x.get('estimable')]

REPRO = '<p>Code and per-landmark results: <a href="https://github.com/parallaxmetrology/loop-visibility">github.com/parallaxmetrology/loop-visibility</a>. <code>python reproduce.py</code> recomputes every number in this note from the per-landmark files (81 checks; no images or landmark files needed). The repository also holds the analysis plans with the SHA-256 hashes recorded before each run, the exact study scripts, and standalone copies of both registrars and of the Hodge split, verified to reproduce the study\'s outputs exactly on DIR-Lab data. The Demons run was interrupted by an out-of-memory crash after five cases; cases 6 to 10 were completed by a memory-lean script with identical parameters and query rule, declared and hashed before it ran. DIR-Lab and POPI images, landmark coordinates and raw measurements are not redistributed; DIR-Lab data are available from its authors on request. Analysis code was written with AI coding assistants (OpenAI Codex for the preceding pilot, Anthropic Claude for this study).</p>' if NO_CT else '<p>Code, analysis plans with SHA-256 hashes recorded before each run, sealed measurements and the lab book are in <code>dvc-holonomy-multicycle/</code>: <code>measure.py</code>, <code>evaluate.py</code> (POPI); <code>dirlab/</code> (pre-registration, amendments, window and Demons measurement, scoring); <code>paper/compute.py</code>, <code>compute_demons.py</code>, <code>figures.py</code> (every number and figure in this note). The Hodge solver and Demons wrapper are reused unchanged from the preceding pilot. The Demons run was interrupted by an out-of-memory crash after five cases; cases 6 to 10 were completed by a memory-lean script with identical parameters and query rule, declared and hashed before it ran (<code>dirlab/ADDENDUM_2b.json</code>). DIR-Lab data are not redistributed. Analysis code was written with AI coding assistants (OpenAI Codex for the preceding pilot, Anthropic Claude for this study).</p>'
TOPLINKS = ' &middot; <a href="https://github.com/parallaxmetrology/loop-visibility">Code and data</a> &middot; <a href="labbook.html">Lab book</a>' if NO_CT else ''
HEADMETA = '<link rel="canonical" href="https://parallaxmetrology.com/loop-visibility/"><meta name="description" content="How much registration error can consistency and transitivity checks see? On DIR-Lab lung 4DCT, 93% of Demons error and 79% of a translation registrar&#39;s error is invisible to every loop, against 33% if errors were independent.">' if NO_CT else ''
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visibility Ceiling of Loops</title>{HEADMETA}<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--s1:#2a78d6;--s1b:#104281;--s2:#eb6834;--card:#f4f3f0}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835;--s1:#3987e5;--s1b:#b7d3f6;--s2:#d95926;--card:#232321}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835;--s1:#3987e5;--s1b:#b7d3f6;--s2:#d95926;--card:#232321}}
body{{background:var(--bg);color:var(--ink);font:17px/1.6 Georgia,'Times New Roman',serif;margin:0;padding:0 16px}}
main{{max-width:760px;margin:40px auto 90px}} h1{{font-size:30px;line-height:1.25;margin:0 0 8px}} h2{{font-size:20px;margin-top:40px}}
.meta{{color:var(--ink2);font:14px/1.5 -apple-system,system-ui,sans-serif}} .abs{{background:var(--card);border-radius:8px;padding:14px 18px;margin:22px 0}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font:14px/1.4 -apple-system,system-ui,sans-serif;margin:12px 0;min-width:520px}} th,td{{padding:5px 10px;border-bottom:1px solid var(--rule);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--ink2)}}
.cap{{color:var(--ink2);font:14px/1.5 -apple-system,system-ui,sans-serif;margin-top:4px}} code{{font-size:14px}} .eq{{text-align:center;margin:14px 0;font-style:italic}}
figure{{margin:22px 0}} svg{{width:100%;height:auto;display:block}}
.lab{{fill:var(--ink);font:13px -apple-system,system-ui,sans-serif}} .strong{{font-weight:600}} .val,.tick{{fill:var(--ink2);font:12px -apple-system,system-ui,sans-serif}}
.grid{{stroke:var(--rule)}} .bar{{fill:var(--s1)}} .bar2{{fill:var(--s1b)}} .barD{{fill:var(--s2)}} .chance{{stroke:var(--ink);stroke-width:2}}
.edge{{stroke:var(--ink2);stroke-width:2;fill:none}} .edge.bad{{stroke:var(--s2)}} .mk-edge{{fill:var(--ink2)}} .node{{fill:var(--bg);stroke:var(--ink);stroke-width:1.5}}
.nodelab{{fill:var(--ink);font:12px -apple-system,system-ui,sans-serif}} .ghost{{fill:none;stroke:var(--s2);stroke-width:1.5;stroke-dasharray:3 2}} .gap{{stroke:var(--s2);stroke-width:2;stroke-dasharray:2 2}}
.trl{{fill:none !important;stroke-width:2.5}} .trT{{stroke:#f4f3f0;fill:#f4f3f0}} .trE{{stroke:#eb6834;fill:#eb6834}} .trp{{stroke:#111;stroke-width:1.5}}
.ctlab{{font:12px -apple-system,system-ui,sans-serif;paint-order:stroke;stroke:#111;stroke-width:3px}} .trTl{{fill:#f4f3f0}} .trEl{{fill:#f7a07a}}
.scale{{stroke:#fff;stroke-width:3}} .scalet{{fill:#fff;font:12px -apple-system,system-ui,sans-serif}}
ol.refs{{font-size:15px}}
</style></head><body><main>
<h1>Loops cannot see most registration error: measuring the visibility ceiling of cycle consistency</h1>
<p class="meta">Russell Parrish, Parallax Metrology. Draft methods note, 20 November 2026. Not peer reviewed.{TOPLINKS}</p>

<div class="abs"><b>Abstract.</b> Consistency and transitivity checks (register A to B, B to C and A to C; see whether the loop closes) are used as label-free quality checks on deformable image registration. True displacements are differences of positions, so the part of a registration set that fails to close is exactly the part of its error that fails to close. A Hodge decomposition of the error, computed wherever truth exists, therefore gives the share of error that any closure-based check can see. If errors were independent per registration, the invisible share on a complete graph of n images would be 2/n. On DIR-Lab thoracic 4DCT (10 patients, all 15 pairs among six phases per landmark) the invisible share is {pc(da['integrable_share'])} for whole-volume Diffeomorphic Demons ({da['landmarks']} landmarks; case-clustered 95% interval {rng(da['ci95_case_cluster'])}) and {pc(dr['integrable_share'])} for a local translation registrar ({dr['landmarks']} landmarks; {rng(dr['ci95_case_cluster'])}), against 33.3% expected. Both stay above 70% after a pessimistic allowance for annotation noise, which is itself invisible to loops. On POPI (1 patient, 10 phases) the translation registrar gives {pc(prf['integrable_share'])} against 20%, though there annotation noise cannot be excluded as the main contributor. The visible part still names a landmark's worst registration above chance, but no better than an image residual. The invisible share bounds what any closure-only check can report, and it can be measured on a truth-bearing subset before such a check is relied on.</div>

<h2>1. Why measure this</h2>
<p>Registration quality is hard to check without ground truth, so self-consistency is attractive. Clinical registration QA in radiotherapy is framed by AAPM TG-132 [Brock et al. 2017]. The MIRSIG position paper [Lowther et al. 2022] accepts testing consistency (independence from registration direction) and, where needed, transitivity (independence from the route through three or more images), and notes that passing a consistency tolerance does not by itself establish spatial accuracy. Cycle consistency is also built into recent registration methods [van Harten et al. 2024]. The caveat is widely stated; its size is rarely measured. This note measures it.</p>

<h2>2. The identity</h2>
<figure>{F['identity']}
<figcaption class="cap">Figure 1. Three images A, B, C of one point. Left: true displacements are differences of positions, so every loop closes. Middle: if the registrar tracks the point in B to a slightly wrong place, both edges touching B are wrong, yet the loop still closes; this error is invisible to any loop. Right: if one registration fails on its own, the loop misses by exactly that error; this error is visible.</figcaption></figure>
<p>Take n images and independent pairwise registrations among them, forming a graph whose nodes are images and whose edges carry measured displacements y<sub>ij</sub> of a material point. The truth is t<sub>ij</sub> = p<sub>j</sub> &minus; p<sub>i</sub>, a gradient on the graph. Write the error e = y &minus; t. The Hodge decomposition splits any edge field orthogonally into an integrable (gradient) part and a cyclic part [Jiang et al. 2011]. Because t is purely integrable,</p>
<p class="eq">cyclic(y) = cyclic(e).</p>
<p>Every closure statistic built from the measured edges, from one triangle's transitivity residual to cycle residuals on a larger graph, is a function of cyclic(y), so it sees only cyclic(e). We report the <b>invisible share</b>, the fraction of error energy in the integrable part, pooled over landmarks. (This note measures displacements of tracked points; for nonlinear maps the argument applies to the sampled displacements.)</p>
<p><b>Independent-error baseline.</b> For independent, isotropic edge errors on the complete graph K<sub>n</sub>, the integrable subspace has dimension n &minus; 1 per axis out of n(n &minus; 1)/2 edges, so the expected invisible share is 2/n: 66.7% for three images, 33.3% for six, 20% for ten. Simulation reproduces these values ({pc(N['chance_K3']['simulated'])}, {pc(N['chance_K6']['simulated'])}, {pc(N['chance_K10']['simulated'])}). A share far above 2/n means errors are organised per image (a node offset, as in Figure 1, middle) rather than per registration.</p>
<p><b>Annotation noise is invisible too.</b> A landmark placed slightly wrong in one image is a node offset. We bound this with a pessimistic allowance: the published mean repeat-annotation distance d is read as the mean 3-D error of a single annotation, giving expected noise energy 2&middot;3&middot;(d/1.596)<sup>2</sup> per edge (1.596 = 2&radic;(2/&pi;)), which is removed from the integrable part.</p>

<h2>3. Data and registrars</h2>
<p><b>DIR-Lab 4DCT</b> [Castillo et al. 2009, 2010]: 10 patients, 0.97 to 1.16 mm in-plane, 2.5 mm slices; 75 landmarks per case on the six expiratory phases T00 to T50 (T10 to T40 coordinates are propagated); published per-case repeat-observer errors 0.70 to 1.13 mm. Obtained through DIR-Lab's access form. <b>POPI</b> [Vandemeulebroucke et al. 2007]: one patient, ten phases, 40 expert landmarks, 0.98 mm in-plane, 2 mm slices.</p>
<p><b>Whole-volume Demons.</b> SimpleITK Diffeomorphic Demons, three-level pyramid (8, 4, 2 mm grids; 80, 60, 40 iterations; HU clipped to [&minus;1000, 400]), fitted independently for all 15 phase pairs of each DIR-Lab case (150 fits; {ND['fits_geometry_valid']} with positive Jacobians throughout the interior). Query points follow the landmark: its T00 position is mapped to each phase with the T00-to-phase field, and each edge is read from its own field at that point. Parameters were fixed in advance, not tuned to these data. Mean edge error {da['mean_edge_tre_mm']:.2f} mm.</p>
<p><b>Local translation window.</b> Tapered FFT cross-correlation on a 32-voxel window at the T00 landmark, refined by spatial NCC; the same centre in every phase. Kept if the window fit in the volume and passed a fixed texture check: {dr['landmarks']} of 750 DIR-Lab landmarks, 28 of 40 POPI landmarks (POPI also used a body-interior guard). Mean edge error on DIR-Lab {ND['window_refined_mean_edge_tre_mm']:.2f} mm.</p>
<p>Every registration was independent, so no loop closes by construction. Only T00 landmarks were read during measurement; measurements were sealed (hashed) before targets were read for scoring, and analysis plans were hashed before each run.</p>

<h2>4. Results</h2>
<figure>{case_figure()}
<figcaption class="cap">Figure 2. Share of registration error energy invisible to every loop, by DIR-Lab case and registrar, with the independent-error baseline (vertical tick: 33.3% for six phases, 20% for ten). Hover a bar for its landmark count.</figcaption></figure>
<div class="tw"><table><thead><tr><th>Registrar, dataset</th><th>Landmarks</th><th>Invisible share</th><th>95% interval</th><th>After noise allowance</th><th>Baseline</th></tr></thead><tbody>
<tr><td>Demons, DIR-Lab</td><td>{da['landmarks']}</td><td>{pc(da['integrable_share'])}</td><td>{rng(da['ci95_case_cluster'])}</td><td>{pc(da['truth_noise_adjusted_share'])} ({rng(da['adjusted_ci95_case_cluster'])})</td><td>33.3%</td></tr>
<tr><td>Demons, DIR-Lab, window's landmarks</td><td>{dm['landmarks']}</td><td>{pc(dm['integrable_share'])}</td><td>{rng(dm['ci95_case_cluster'])}</td><td>{pc(dm['truth_noise_adjusted_share'])}</td><td>33.3%</td></tr>
<tr><td>Window (refined), DIR-Lab</td><td>{dr['landmarks']}</td><td>{pc(dr['integrable_share'])}</td><td>{rng(dr['ci95_case_cluster'])}</td><td>{pc(dr['truth_noise_adjusted_share'])} ({rng(dr['adjusted_ci95_case_cluster'])})</td><td>33.3%</td></tr>
<tr><td>Window (FFT only), DIR-Lab</td><td>{df['landmarks']}</td><td>{pc(df['integrable_share'])}</td><td>{rng(df['ci95_case_cluster'])}</td><td>{pc(df['truth_noise_adjusted_share'])}</td><td>33.3%</td></tr>
<tr><td>Window (refined), POPI</td><td>{prf['landmarks']}</td><td>{pc(prf['integrable_share'])}</td><td>{rng(prf['ci95'])}</td><td>not identifiable</td><td>20.0%</td></tr>
</tbody></table></div>
<p class="cap">Intervals: 4000 bootstrap resamples of patients (DIR-Lab) or landmarks (POPI). Shares pool error energy, so landmarks with larger errors weigh more.</p>
<p><b>DIR-Lab.</b> Both registrars sit far above the baseline in every case. For Demons, annotation noise under the pessimistic allowance is at most {pc(da['truth_noise_energy_fraction'])} of error energy; for the window, {pc(dr['truth_noise_energy_fraction'])}. Per landmark the split is extreme: the median landmark's error is {pc(da['per_landmark_median'])} invisible for Demons and {pc(dr['per_landmark_median'])} for the window. The pooled share sits below the median because visible energy is concentrated: for the window, the 10% of landmarks with the most visible energy carry {pc(VIS_TOP10)} of it; for the window the two cases with the largest errors (6 and 8) have the lowest shares.</p>
<div class="tw"><table><thead><tr><th>Case</th><th>Window landmarks</th><th>Window invisible</th><th>Window after allowance</th><th>Demons landmarks</th><th>Demons invisible</th></tr></thead><tbody>{cases_tbl}</tbody></table></div>
<p><b>POPI.</b> The window's invisible share is {pc(prf['integrable_share'])}, but POPI errors are small (mean squared edge error {prf['mean_sq_edge_error_mm2']:.1f} mm&sup2;). Under the allowance with d = 0.5 mm the adjusted share is {pc(prf['truth_noise_adjusted_share']['0.5'])}; with d = 0.85 mm, {pc(prf['truth_noise_adjusted_share']['0.85'])}; with d of 1.13 mm or more the allowance exceeds the total error energy. POPI publishes no repeat-annotation figure, so on POPI annotation noise cannot be excluded as the main invisible component. POPI is consistent with DIR-Lab but does not establish the result alone.</p>

<figure>{F['edges']}
<figcaption class="cap">Figure 3. Error on each of the 15 phase pairs for two DIR-Lab landmarks (window registrar), with the loop-visible part of each edge's error inside. Left: the landmark whose invisible share is closest to the median ({pc(RU['median_share'])}), within the case whose pooled share is closest to the DIR-Lab pooled value (case {RU['case_pick']}). Right: the landmark with the largest loop-visible error energy in the dataset. Both chosen by these rules, not by eye.</figcaption></figure>
<figure>{F['trajectory']}
<figcaption class="cap">Figure 4. The left-hand landmark of Figure 3 {'in the ' + TM['plane'] + ' plane through the landmark (5 mm grid; the CT image is omitted in this public edition)' if NO_CT else 'on its T00 CT (' + TM['plane'] + ' slice through the landmark'}; plane chosen as the one showing the larger in-plane offset at T50{'' if NO_CT else ')'}. White: annotated positions T00 to T50. Orange: the window registrar's estimates, starting from the same T00 point. The landmark moves {TRUE5:.1f} mm from T00 to T50 while the estimate moves {EST5:.1f} mm, an offset of up to {max(TM['offset_mm_by_phase']):.1f} mm, yet its 15 registrations close around every loop to within a few percent of the error energy.</figcaption></figure>

<h2>5. What the visible part does</h2>
<div class="tw"><table><thead><tr><th>Worst registration of a landmark named by</th><th>DIR-Lab, window (477)</th><th>POPI, window (28)</th></tr></thead><tbody>
<tr><td>Loop residual (cyclic part of the edge)</td><td>{pc(ld['cyclic']['top1_rate'])}</td><td>{PL['eulerian_refined']['top1']['cyclic']} of 28</td></tr>
<tr><td>NCC image residual</td><td>{pc(ld['ncc']['top1_rate'])}</td><td>{PL['eulerian_refined']['top1']['ncc']} of 28</td></tr>
<tr><td>Measured shift size</td><td>{pc(ld['shift']['top1_rate'])}</td><td>{PL['eulerian_refined']['top1']['shift']} of 28</td></tr>
<tr><td>Chance</td><td>{pc(ld['chance_top1_rate'])}</td><td>{28 / 45:.1f} of 28</td></tr></tbody></table></div>
<p>Loop residuals name a landmark's worst registration well above chance. On POPI they did so more often than the image residual; that did not replicate on DIR-Lab, where the two are indistinguishable (difference {ld['cyc_minus_ncc_top1_rate_ci'][0]*100:+.1f} to {ld['cyc_minus_ncc_top1_rate_ci'][1]*100:+.1f} points, case-clustered 95%) and the image residual ranks all edges better (mean within-landmark Spearman {ld['ncc']['within_spearman']:.2f} versus {ld['cyclic']['within_spearman']:.2f}). Using loops to correct rather than detect fares no better: least-squares synchronisation of all 45 POPI registrations increased phase-10 error by {pc(-S['refined']['ls_sync']['out10_gain_vs_direct'])} and reduced all-pairs error by {pc(1 - S['refined']['ls_sync']['all45_mean'] / S['refined']['direct']['all45_mean'])}.</p>

<h2>6. Where the invisible error comes from (exploratory)</h2>
<p>For the window registrar three checks narrow it. Estimator shrinkage is small: exact integer-voxel shifts of real POPI windows are recovered with gains of {', '.join(f'{v:.2f}' for v in DG['known_shift']['gain_refined_zyx'])} (z, y, x, refined). Re-centring each window on the estimated landmark position in its source phase left POPI all-pairs error essentially unchanged ({S['fft']['direct']['all45_mean']:.3f} to {LG['fft']['direct']['all45_mean']:.3f} mm). And an unrelated registrar shares the error: on POPI, Demons and the window have error correlations of {', '.join(f'{v:.2f}' for v in SE['refined']['error_corr_zyx'])} per axis. With annotation noise bounded on DIR-Lab, the main invisible component is a per-image tracking offset: the registrar follows a trajectory close to, but not the same as, the landmark's, consistently across every pairing (Figure 4).</p>
<p>A pre-registered attempt to read the error shared between the two registrars as annotation noise (triple collocation [Stoffelen 1998]) failed its known-answer test against DIR-Lab's published observer errors: on T00 to T50, estimates ran {min(est_ratio):.1f} to {max(est_ratio):.1f} times the published values and tracked motion amplitude. The two registrars share tracking error, which violates that method's independence assumption.</p>

<h2>7. Relation to prior work</h2>
<p>That consistent errors escape consistency checks is understood qualitatively. Inverse consistency has long been imposed as a desirable property of registration [Christensen and Johnson 2001], yet a consistent registration can still be wrong, as the MIRSIG position paper states for clinical QA [Lowther et al. 2022]. Loop constraints are used to detect erroneous pairwise relations in structure from motion [Zach et al. 2010], which works only where an error breaks closure. Image-similarity surrogates are likewise unreliable proxies for accuracy [Rohlfing 2012]. Synchronisation methods exploit closure on graphs [Singer 2011]. The contribution here is a measurement: the Hodge split of the error on a truth-bearing subset, with an independent-error baseline of 2/n and an annotation-noise allowance, turns the qualitative caveat into a number per registrar and data type.</p>

<h2>8. Limits</h2>
<ul>
<li>Two registrars, both run with fixed parameters and neither tuned to these data; their DIR-Lab errors ({ND['window_refined_mean_edge_tre_mm']:.1f} and {da['mean_edge_tre_mm']:.1f} mm mean edge error over all 15 phase pairs) are above the roughly 1 mm that the best published methods reach on DIR-Lab's extreme phases. Registrars near the annotation floor, including learned ones, may split differently; the measurement applies per registrar.</li>
<li>Lung 4DCT only, 11 patients. POPI alone does not separate registration error from annotation noise, and its targets had been examined in a preceding pilot.</li>
<li>Energy pooling weights large errors heavily; per-landmark medians are reported beside pooled shares.</li>
<li>DIR-Lab intermediate-phase landmarks are propagated rather than independently annotated; their error is per-image and therefore inside the invisible share and the allowance.</li>
<li>Demons query points come from the registrar's own T00-to-phase estimates, as they would in use.</li>
<li>Everything except the pre-registered covariance test in section 6 (which failed) is exploratory.</li>
</ul>

<h2>9. Reproducibility</h2>
{REPRO}

<h2>References</h2>
<ol class="refs">
<li>Brock KK, Mutic S, McNutt TR, Li H, Kessler ML. Use of image registration and fusion algorithms and techniques in radiotherapy: Report of the AAPM Radiation Therapy Committee Task Group No. 132. Med Phys 44(7):e43 (2017). doi:10.1002/mp.12256</li>
<li>Castillo R, Castillo E, Guerra R, et al. A framework for evaluation of deformable image registration spatial accuracy using large landmark point sets. Phys Med Biol 54:1849 (2009). doi:10.1088/0031-9155/54/7/001</li>
<li>Castillo E, Castillo R, Martinez J, Shenoy M, Guerrero T. Four-dimensional deformable image registration using trajectory modeling. Phys Med Biol 55:305 (2010).</li>
<li>Christensen GE, Johnson HJ. Consistent image registration. IEEE Trans Med Imaging 20(7):568 (2001). doi:10.1109/42.932742</li>
<li>Jiang X, Lim L-H, Yao Y, Ye Y. Statistical ranking and combinatorial Hodge theory. Math Program 127:203 (2011). doi:10.1007/s10107-010-0419-x</li>
<li>Lowther N, Louwe R, Yuen J, Hardcastle N, Yeo A, Jameson M. MIRSIG position paper: the use of image registration and fusion algorithms in radiotherapy. Phys Eng Sci Med 45:421 (2022). doi:10.1007/s13246-022-01125-3</li>
<li>Rohlfing T. Image similarity and tissue overlaps as surrogates for image registration accuracy: widely used but unreliable. IEEE Trans Med Imaging 31(2):153 (2012). doi:10.1109/TMI.2011.2163944</li>
<li>Singer A. Angular synchronization by eigenvectors and semidefinite programming. Appl Comput Harmon Anal 30(1):20 (2011). doi:10.1016/j.acha.2010.02.001</li>
<li>Stoffelen A. Toward the true near-surface wind speed: error modeling and calibration using triple collocation. J Geophys Res Oceans 103:7755 (1998). doi:10.1029/97JC03180</li>
<li>van Harten LD, Stoker J, Išgum I. Robust deformable image registration using cycle-consistent implicit representations. IEEE Trans Med Imaging 43(2):784 (2024). doi:10.1109/TMI.2023.3321425</li>
<li>Vandemeulebroucke J, Sarrut D, Clarysse P. The POPI-model, a point-validated pixel-based breathing thorax model. Proc. ICCR (2007).</li>
<li>Zach C, Klopschitz M, Pollefeys M. Disambiguating visual relations using loop constraints. Proc. IEEE CVPR (2010). doi:10.1109/CVPR.2010.5539801</li>
</ol>
</main></body></html>"""
(P / ('visibility_ceiling_public.html' if NO_CT else 'visibility_ceiling.html')).write_text(page)
print('wrote', len(page))
