"""Truth-blind Lagrangian re-measurement of the K10 graph (see PLAN_ADDENDUM.json)."""
import sys, json, hashlib, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = ROOT.parent / 'dvc-holonomy-pilot'
sys.path.insert(0, str(PILOT)); sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from measure import PHASES, EDGES  # noqa: E402
from correspondence.common import volume  # noqa: E402
from interior.selection import envelope  # noqa: E402
from dvc_pilot.acceptance_run import normalized_patch  # noqa: E402
from dvc_pilot.registration import register, registered_residual  # noqa: E402
from dvc_pilot.refinement import refine_translation  # noqa: E402

VOLS, META = {}, {}


def init():
    for p in PHASES:
        v, _, _ = volume(p); VOLS[p] = v; META[p] = envelope(v)[1]


def work(rec):
    if not VOLS:
        init()
    c = np.array(rec['center'])
    star = {}
    for e in rec['edges']:
        if e['i'] == 1: star[e['j']] = np.array(e['t'])
        if e['j'] == 1: star[e['i']] = -np.array(e['t'])
    star[1] = np.zeros(3)
    centers = {i: np.rint(c + star[i]).astype(int) for i in range(10)}
    new = []
    for old in rec['edges']:
        i, j = old['i'], old['j']
        ci = centers[i]
        try:
            a = normalized_patch(VOLS[PHASES[i]], ci, 32, META[PHASES[i]]['levels'])
            b = normalized_patch(VOLS[PHASES[j]], ci, 32, META[PHASES[j]]['levels'])
        except ValueError:
            new.append(dict(old, recentred=False)); continue
        r = register(a, b)
        if not r['valid']:
            new.append(dict(old, recentred=False)); continue
        f = refine_translation(a, b, r['t'])
        tr = f['t'] if f['success'] else r['t']
        # t is measured at c_i; it maps phase-i coords to phase-j coords there.
        new.append(dict(i=i, j=j, valid=True, recentred=True, center=ci.tolist(), t=r['t'], residual=r['residual'],
                        gradient_residual=r['gradient_residual'], ambiguity=r['ambiguity'],
                        valid_fraction=r['valid_fraction'], refine_success=f['success'], t_refined=tr,
                        residual_refined=registered_residual(a, b, tr)[0]))
    return dict(landmark=rec['landmark'], center=rec['center'], phase_centers={PHASES[i]: centers[i].tolist() for i in centers},
                edges=new)


def main():
    src = ROOT / 'results/measurements.jsonl'
    recs = [json.loads(l) for l in src.read_text().splitlines()]
    out = HERE / 'results'; out.mkdir(exist_ok=True)
    dest = out / 'measurements.jsonl'
    if dest.exists():
        raise SystemExit('exists')
    with dest.open('x') as f, ProcessPoolExecutor(max_workers=6) as pool:
        for n, r in enumerate(pool.map(work, recs)):
            f.write(json.dumps(r) + '\n'); print('measured', n + 1, flush=True)
    seal = json.loads((ROOT / 'results/measure_seal.json').read_text())
    seal.update(sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), source=str(src))
    (out / 'measure_seal.json').write_text(json.dumps(seal, indent=1))


if __name__ == '__main__':
    main()
