"""Figures for the note, as SVG strings in paper/figs.json. Examples chosen by stated rules."""
import sys, json, base64, io
from pathlib import Path
from itertools import combinations
import numpy as np

P = Path(__file__).resolve().parent; R = P.parent
sys.path[:0] = [str(R.parent / 'dvc-holonomy-pilot'), str(R / 'dirlab')]
sys.dont_write_bytecode = True
from dh.complexes import Complex2  # noqa: E402
from dh.hodge import decompose  # noqa: E402
from common import CASES, load, landmarks_zyx  # noqa: E402

E6 = list(combinations(range(6), 2)); CX = Complex2(6, E6, list(combinations(range(6), 3)))
PH = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50']
N = json.loads((P / 'numbers.json').read_text())
figs = {}

# ---------- Figure 1: the identity (schematic) ----------
def tri(ox, title, sub, node_shift=None, edge_bad=None):
    A, B, C = (ox + 40, 170), (ox + 190, 170), (ox + 115, 50)
    pts = dict(A=A, B=B, C=C)
    o = [f'<text x="{ox + 115}" y="18" text-anchor="middle" class="lab strong">{title}</text>',
         f'<text x="{ox + 115}" y="218" text-anchor="middle" class="tick">{sub}</text>']
    def arrow(p, q, cls, dx=0, dy=0):
        (x1, y1), (x2, y2) = p, q
        x2 += dx; y2 += dy
        vx, vy = x2 - x1, y2 - y1; L = (vx * vx + vy * vy) ** .5
        sx, sy = x1 + vx * 14 / L, y1 + vy * 14 / L; ex, ey = x2 - vx * 16 / L, y2 - vy * 16 / L
        return f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" class="{cls}" marker-end="url(#ah-{cls.split()[0]})"/>'
    for (u, v) in [('A', 'B'), ('B', 'C'), ('A', 'C')]:
        dx = dy = 0
        cls = 'edge'
        if node_shift and v == node_shift:
            dx, dy = 18, -10; cls = 'edge bad'
        if node_shift and u == node_shift:
            cls = 'edge bad'
        if edge_bad == (u, v):
            dx, dy = 20, 6; cls = 'edge bad'
        if node_shift and u == node_shift:
            (x1, y1) = pts[u]; pts_u = (x1 + 18, y1 - 10)
            o.append(arrow(pts_u, pts[v], cls))
        else:
            o.append(arrow(pts[u], pts[v], cls, dx, dy))
    for k, (x, y) in pts.items():
        o.append(f'<circle cx="{x}" cy="{y}" r="11" class="node"/><text x="{x}" y="{y + 4}" text-anchor="middle" class="nodelab">{k}</text>')
    if node_shift:
        x, y = pts[node_shift]
        o.append(f'<circle cx="{x + 18}" cy="{y - 10}" r="11" class="ghost"/><text x="{x + 34}" y="{y - 22}" class="tick">estimated B</text>')
    if edge_bad:
        x, y = pts[edge_bad[1]]
        o.append(f'<line x1="{x + 6}" y1="{y + 2}" x2="{x + 18}" y2="{y + 6}" class="gap"/><text x="{x + 26}" y="{y + 10}" class="tick">gap = visible error</text>')
    return ''.join(o)

defs = ''.join(f'<marker id="ah-{c}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="mk-{c}"/></marker>' for c in ['edge'])
fig1 = (f'<svg viewBox="0 0 770 230" role="img" aria-label="Why loops see only part of the error"><defs>{defs}</defs>'
        + tri(0, 'Truth', 'positions differ; every loop closes')
        + tri(245, 'Per-image offset', 'B tracked wrongly; loop still closes', node_shift='B')
        + tri(490, 'One bad registration', 'A to C wrong; loop fails to close', edge_bad=('A', 'C'))
        + '</svg>')
figs['identity'] = fig1

# ---------- examples from the window registrar (refined) ----------
six = [json.loads(l) for l in (R / 'dirlab/results/six_phase.jsonl').read_text().splitlines()]
rows = []
for r in six:
    if not r['fits'] or not r['texture'] or not all(e['valid'] for e in r['edges']):
        continue
    c = r['case']; spz = np.array(CASES[c][1][::-1])
    T = np.array([landmarks_zyx(c, p, '4D-75')[r['landmark']] for p in PH]) * spz
    Y = np.array([e['t_refined'] for e in r['edges']]) * spz
    err = Y - np.array([T[j] - T[i] for i, j in E6])
    cyc = np.stack([err[:, a] - decompose(CX, err[:, a]).grad for a in range(3)], 1)
    Eg = (err ** 2).sum() - (cyc ** 2).sum()
    rows.append(dict(case=c, lm=r['landmark'], center=r['center'], T=T, Y=Y, err=err, cyc=cyc,
                     share=float(1 - (cyc ** 2).sum() / (err ** 2).sum()), visible_energy=float((cyc ** 2).sum())))
pooled = N['dirlab_t_refined']['integrable_share']
bc = N['dirlab_t_refined']['by_case']
case_pick = min(bc, key=lambda c: abs(bc[c]['share'] - pooled))
med = float(np.median([r['share'] for r in rows]))
typical = min([r for r in rows if r['case'] == int(case_pick)], key=lambda r: abs(r['share'] - med))
gross = max(rows, key=lambda r: r['visible_energy'])
figs['rules'] = dict(case_pick=int(case_pick), median_share=med,
                     typical=dict(case=typical['case'], landmark=typical['lm'], share=typical['share']),
                     gross=dict(case=gross['case'], landmark=gross['lm'], share=gross['share']))


def edge_panel(ex, ox, title, W=330, H=210):
    e = np.linalg.norm(ex['err'], axis=1); v = np.linalg.norm(ex['cyc'], axis=1)
    top = max(e.max(), 0.5); left, bot = ox + 44, H - 30
    sy = (H - 76) / top; bw = (W - 50) / 15
    o = [f'<text x="{ox + W / 2}" y="16" text-anchor="middle" class="lab strong">{title}</text>',
         f'<text x="{ox + W / 2}" y="32" text-anchor="middle" class="tick">{100 * (1 - ex["share"]):.1f}% of this landmark\'s error energy is loop-visible</text>']
    for t in np.linspace(0, top, 4):
        y = bot - t * sy
        o.append(f'<line x1="{left}" x2="{ox + W - 8}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{left - 4}" y="{y + 4:.1f}" text-anchor="end" class="tick">{t:.1f}</text>')
    for k, (i, j) in enumerate(E6):
        x = left + k * bw + 1
        tip = f'{PH[i]} to {PH[j]}: error {e[k]:.2f} mm, loop-visible part {v[k]:.2f} mm'
        o.append(f'<g><title>{tip}</title><rect x="{x:.1f}" y="{bot - e[k] * sy:.1f}" width="{bw - 3:.1f}" height="{e[k] * sy:.1f}" rx="2" class="bar"/>'
                 f'<rect x="{x + 3:.1f}" y="{bot - v[k] * sy:.1f}" width="{max(bw - 9, 2):.1f}" height="{v[k] * sy:.1f}" rx="2" class="bar2"/></g>')
    o.append(f'<text x="{left + 7.5 * bw}" y="{H - 10}" text-anchor="middle" class="tick">15 phase pairs (T00 to T10 ... T40 to T50)</text>')
    o.append(f'<text x="{ox + 2}" y="{bot - (H - 76) / 2:.0f}" class="tick" text-anchor="middle" transform="rotate(-90 {ox + 8} {bot - (H - 76) / 2:.0f})">mm</text>')
    return ''.join(o)

legend = ('<rect x="200" y="226" width="12" height="12" rx="2" class="bar"/><text x="218" y="236" class="lab">registration error, mm</text>'
          '<rect x="390" y="226" width="12" height="12" rx="2" class="bar2"/><text x="408" y="236" class="lab">its loop-visible (cyclic) part</text>')
figs['edges'] = (f'<svg viewBox="0 0 700 246" role="img" aria-label="Error per phase pair and its loop-visible part">' + legend
                 + edge_panel(typical, 0, f'Typical landmark (case {typical["case"]})')
                 + edge_panel(gross, 360, f'Largest visible error (case {gross["case"]})') + '</svg>')

# ---------- Figure: CT crop with trajectories (typical landmark) ----------
from scipy import ndimage
c = typical['case']; spz = np.array(CASES[c][1][::-1])
v0, _ = load(c, 'T00', False)
ctr = np.array(typical['center'])  # voxel zyx
T_ = typical['T']; est5 = T_[0] + typical['Y'][E6.index((0, 5))]
off = est5 - T_[5]
H_AX = 2 if abs(off[2]) >= abs(off[1]) else 1  # rule: plane (z with x or y) showing the larger in-plane T50 offset
PLANE = 'coronal' if H_AX == 2 else 'sagittal'
half_mm = 12.0
hz, hx = int(half_mm / spz[0]), int(half_mm / spz[H_AX])
z0, z1 = max(ctr[0] - hz, 0), min(ctr[0] + hz, v0.shape[0]); x0, x1 = max(ctr[H_AX] - hx, 0), min(ctr[H_AX] + hx, v0.shape[H_AX])
sl = (v0[z0:z1, ctr[1], x0:x1] if H_AX == 2 else v0[z0:z1, x0:x1, ctr[2]]).astype(np.float32) - 1024
img = ndimage.zoom(sl, (spz[0] / 0.25, spz[H_AX] / 0.25), order=3)  # 0.25 mm square pixels
img = np.clip((img + 1000) / 1100, 0, 1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
buf = io.BytesIO(); plt.imsave(buf, img, cmap='gray', vmin=0, vmax=1, origin='upper', format='png')
b64 = base64.b64encode(buf.getvalue()).decode()
Hpx, Wpx = img.shape
S = 16.0  # svg units per mm
def to_svg(pz, px):  # mm in volume -> svg coords (z down as image rows)
    return ((px - x0 * spz[H_AX]) * S, (pz - z0 * spz[0]) * S)
T = typical['T']
est = {0: T[0].copy()}
for k, (i, j) in enumerate(E6):
    if i == 0:
        est[j] = T[0] + typical['Y'][k]
Wv, Hv = Wpx * 0.25 * S, Hpx * 0.25 * S
import os
NO_CT = os.environ.get('LV_NO_CT') == '1'  # public build: no DIR-Lab image pixels, plain background
o = [f'<svg viewBox="0 0 {Wv + 230:.0f} {Hv + 4:.0f}" role="img" aria-label="True and estimated landmark trajectory">']
if NO_CT:
    o.append(f'<rect x="0" y="0" width="{Wv:.0f}" height="{Hv:.0f}" fill="#2a2a28"/>')
    for g in range(0, int(Wv) + 1, int(5 * S)):
        o.append(f'<line x1="{g}" x2="{g}" y1="0" y2="{Hv:.0f}" stroke="#3d3d3a"/>')
    for g in range(0, int(Hv) + 1, int(5 * S)):
        o.append(f'<line x1="0" x2="{Wv:.0f}" y1="{g}" y2="{g}" stroke="#3d3d3a"/>')
else:
    o.append(f'<image href="data:image/png;base64,{b64}" x="0" y="0" width="{Wv:.0f}" height="{Hv:.0f}" preserveAspectRatio="none"/>')
for name, pts, cls in [('true (annotated)', [T[i] for i in range(6)], 'trT'), ('window registrar', [est[i] for i in range(6)], 'trE')]:
    xy = [to_svg(p[0], p[H_AX]) for p in pts]
    o.append('<polyline points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in xy) + f'" class="trl {cls}"/>')
    for k, (a, b) in enumerate(xy):
        o.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{5 if k in (0, 5) else 3.5}" class="trp {cls}"><title>{name}, {PH[k]}</title></circle>')
    ex_, ey_ = xy[-1]
    if cls == 'trT':
        o.append(f'<text x="{ex_ - 8:.1f}" y="{ey_ - 10:.1f}" text-anchor="end" class="ctlab {cls}l">T50</text>')
    else:
        o.append(f'<text x="{ex_ + 12:.1f}" y="{ey_ + 4:.1f}" text-anchor="start" class="ctlab {cls}l">T50</text>')
sx_, sy_ = to_svg(T[0][0], T[0][H_AX]); o.append(f'<text x="{sx_ + 10:.1f}" y="{sy_ + 20:.1f}" class="ctlab trTl">T00 (shared start)</text>')
o.append(f'<line x1="{Wv - 14 - 5 * S:.0f}" x2="{Wv - 14:.0f}" y1="{Hv - 16:.0f}" y2="{Hv - 16:.0f}" class="scale"/><text x="{Wv - 14 - 2.5 * S:.0f}" y="{Hv - 24:.0f}" text-anchor="middle" class="scalet">5 mm</text>')
lx = Wv + 14
o.append(f'<circle cx="{lx + 6}" cy="30" r="5" class="trp trT"/><text x="{lx + 18}" y="34" class="lab">annotated landmark</text>'
         f'<circle cx="{lx + 6}" cy="56" r="5" class="trp trE"/><text x="{lx + 18}" y="60" class="lab">window registrar</text>'
         f'<text x="{lx}" y="90" class="tick">T00 to T50, {PLANE}</text><text x="{lx}" y="106" class="tick">projection; large dots</text><text x="{lx}" y="122" class="tick">mark T00 and T50</text>')
o.append('</svg>')
figs['trajectory'] = ''.join(o)
figs['trajectory_meta'] = dict(case=c, landmark=typical['lm'], plane=PLANE, t50_offset_zyx=off.tolist(),
                               offset_mm_by_phase=[float(np.linalg.norm(est[i] - T[i])) for i in range(6)])
figs['trajectory_has_ct'] = not NO_CT
(P / ('figs_noct.json' if NO_CT else 'figs.json')).write_text(json.dumps(figs))
print(json.dumps({k: v for k, v in figs.items() if k in ('rules', 'trajectory_meta')}, indent=1))
