"""DIR-Lab 4DCT adapter. Dimensions and spacing copied from the DIR-Lab 4DCT page (2026-11-19)."""
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
DATA = D / 'data'
# case: (dims x,y,z), (spacing x,y,z mm)
CASES = {
    1: ((256, 256, 94), (0.97, 0.97, 2.5)), 2: ((256, 256, 112), (1.16, 1.16, 2.5)),
    3: ((256, 256, 104), (1.15, 1.15, 2.5)), 4: ((256, 256, 99), (1.13, 1.13, 2.5)),
    5: ((256, 256, 106), (1.10, 1.10, 2.5)), 6: ((512, 512, 128), (0.97, 0.97, 2.5)),
    7: ((512, 512, 136), (0.97, 0.97, 2.5)), 8: ((512, 512, 128), (0.97, 0.97, 2.5)),
    9: ((512, 512, 128), (0.97, 0.97, 2.5)), 10: ((512, 512, 120), (0.97, 0.97, 2.5)),
}
PUBLISHED_OBSERVER_MM = {1: 0.85, 2: 0.70, 3: 0.77, 4: 1.13, 5: 0.92, 6: 0.97, 7: 0.81, 8: 1.03, 9: 0.75, 10: 0.86}
PHASES = [f'T{i}0' for i in range(10)]


def case_dir(c):
    import re
    hits = [p for p in DATA.iterdir() if p.is_dir() and (m := re.match(r'case(\d+)', p.name, re.I)) and int(m.group(1)) == c]
    assert len(hits) == 1, (c, hits)
    return hits[0]


def image_path(c, phase):
    hits = [p for p in case_dir(c).rglob('*.img') if f'_{phase}' in p.name]
    assert len(hits) == 1, (c, phase, hits)
    return hits[0]


def landmark_path(c, phase, kind):
    """kind '300' (extreme phases) or '4D-75'."""
    pat = '300' if kind == '300' else '4D-75'
    hits = [p for p in case_dir(c).rglob('*.txt') if pat in p.name and f'_{phase}' in p.name]
    assert len(hits) == 1, (c, phase, kind, hits)
    return hits[0]


def load(c, phase, flip_y):
    (nx, ny, nz), sp = CASES[c]
    a = np.fromfile(image_path(c, phase), dtype='<i2')
    assert a.size == nx * ny * nz, (c, phase, a.size)
    v = a.reshape(nz, ny, nx)
    if flip_y:
        v = v[:, ::-1, :]
    return v, np.array(sp[::-1], float)  # zyx order


def landmarks_zyx(c, phase, kind='300'):
    """1-based xyz voxel indices -> 0-based zyx fractional voxel indices."""
    a = np.loadtxt(landmark_path(c, phase, kind))
    assert a.ndim == 2 and a.shape[1] == 3
    return a[:, ::-1] - 1.0
