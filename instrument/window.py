"""Local translation registrar used in the study (32-voxel window, FFT + spatial NCC refinement).

Copied verbatim in behaviour from the preceding pilot's registration, refinement and
selection modules; only the functions the study calls are included (the unused phase-
normalisation branch of register() is omitted). Axes are (z, y, x);
a translation t maps coordinates in window a to window b.
"""
from functools import lru_cache
import numpy as np
from scipy import ndimage, ndimage as ndi, optimize, fft
from scipy.signal.windows import tukey
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation

PARAMS = dict(mask_stride=4, mask_smoothing=1.5, mask_fraction_min=.03, mask_fraction_max=.90,
              parent_halfwidth_with_guard=40, bandpass_sd_min=3., octant_sd_min=2., textured_octants_min=6,
              gradient_min_share=.05, gradient_trace_min=.5)


# ---- window extraction and intensity scaling ----
def patch(volume, center, size):
    c = np.asarray(center, int)
    start = c - size // 2
    end = start + size
    if np.any(start < 0) or np.any(end > volume.shape):
        raise ValueError(f'Patch outside acquisition: {center}, {size}')
    return np.array(volume[tuple(slice(a, b) for a, b in zip(start, end))], dtype=np.float32)


def linear_levels(volume):
    lo, hi = np.percentile(volume[::4, ::4, ::4], [1, 99])
    if hi <= lo:
        raise RuntimeError('Unobservable acquisition intensity')
    return [float(lo), float(hi)]


def normalized_patch(v, center, size, levels):
    lo, hi = levels
    return (patch(v, center, size) - lo) * (255 / (hi - lo))


# ---- guards ----
def envelope(volume):
    """Coarse 3-D largest-component envelope; fill enclosed XY holes."""
    p = PARAMS; lo, hi = linear_levels(volume)
    small = (np.asarray(volume[::4, ::4, ::4], dtype=np.float32) - lo) * 255 / (hi - lo)
    smooth = ndi.gaussian_filter(small, p['mask_smoothing'])
    threshold = float(threshold_otsu(smooth)); bright = smooth > threshold
    labels, n = ndi.label(bright); counts = np.bincount(labels.ravel()); counts[0] = 0
    if n == 0:
        raise ValueError('No specimen component')
    mask = labels == int(counts.argmax())
    for z in range(len(mask)):
        mask[z] = ndi.binary_fill_holes(mask[z])
    frac = float(mask.mean())
    if not p['mask_fraction_min'] <= frac <= p['mask_fraction_max']:
        raise ValueError('Implausible specimen envelope fraction')
    return mask, dict(levels=[lo, hi], threshold=threshold, mask_fraction=frac, mask_shape=list(mask.shape))


def interior(mask, center):
    c = np.asarray(center); h = PARAMS['parent_halfwidth_with_guard']; s = PARAMS['mask_stride']
    lo = (c - h) // s; hi = (c + h + s - 1) // s
    return bool(np.all(lo >= 0) and np.all(hi <= mask.shape) and mask[tuple(slice(a, b) for a, b in zip(lo, hi))].all())


def texture(a):
    smooth = ndi.gaussian_filter(np.asarray(a, dtype=np.float64), 1.)
    band = smooth - ndi.gaussian_filter(np.asarray(a, dtype=np.float64), 3.)
    g = np.stack(np.gradient(smooth[3:-3, 3:-3, 3:-3])).reshape(3, -1)
    eig = np.linalg.eigvalsh(g @ g.T / g.shape[1]); trace = float(eig.sum())
    sd = float(band[3:-3, 3:-3, 3:-3].std()); tiles = []
    for z in [slice(3, 16), slice(16, 29)]:
        for y in [slice(3, 16), slice(16, 29)]:
            for x in [slice(3, 16), slice(16, 29)]:
                tiles.append(float(band[z, y, x].std()))
    p = PARAMS; count = sum(v >= p['octant_sd_min'] for v in tiles); share = float(eig[0] / max(trace, 1e-12))
    accepted = sd >= p['bandpass_sd_min'] and count >= p['textured_octants_min'] and trace >= p['gradient_trace_min'] and share >= p['gradient_min_share']
    return dict(accepted=bool(accepted), bandpass_sd=sd, textured_octants=count, gradient_trace=trace, gradient_min_share=share)


# ---- FFT translation ----
@lru_cache(maxsize=16)
def taper(shape):
    w = np.ones(shape, dtype=np.float32)
    for axis, length in enumerate(shape):
        dims = [1] * len(shape); dims[axis] = length
        w *= tukey(length, alpha=0.5).reshape(dims)
    return w


def ncc_residual(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-10:
        return 1.0
    r = float(np.clip(a @ b / den, -1, 1))
    return 1.0 - max(r, 0.0) ** 2


def registered_residual(a, b, t, margin=5):
    grid = np.indices(tuple(s - 2 * margin for s in a.shape), dtype=np.float32)
    grid += margin
    loc = grid + np.asarray(t)[:, None, None, None]
    valid = np.ones(grid.shape[1:], bool)
    for k, s in enumerate(a.shape):
        valid &= (loc[k] >= 1) & (loc[k] <= s - 2)
    if valid.mean() < 0.5:
        return 1.0, 1.0, float(valid.mean())
    moving = ndimage.map_coordinates(b, loc, order=1, mode='constant', cval=0)
    fixed = a[tuple(slice(margin, -margin) for _ in a.shape)]
    residual = ncc_residual(fixed[valid], moving[valid])
    ga = np.sqrt(sum(g * g for g in np.gradient(fixed)))
    gb = np.sqrt(sum(g * g for g in np.gradient(moving)))
    valid = ndimage.binary_erosion(valid, iterations=1)
    return residual, ncc_residual(ga[valid], gb[valid]), float(valid.mean())


def register(a, b, normalization=None, upsample=20):
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError('Equal-shaped 3-D windows required')
    if min(float(a.std()), float(b.std())) < 1e-6:
        return dict(valid=False, reason='unobservable_constant_window')
    w = taper(a.shape)
    aa = (a - a.mean()) * w; bb = (b - b.mean()) * w
    fa, fb = fft.fftn(aa), fft.fftn(bb)
    shift, _, _ = phase_cross_correlation(fa, fb, space='fourier', upsample_factor=upsample, normalization=normalization)
    t = -np.asarray(shift, dtype=float)
    cross = fa * fb.conj()
    corr = fft.ifftn(cross).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    main = float(corr[peak])
    keep = np.ones(corr.shape, bool)
    coords = [np.mod(np.arange(p - 1, p + 2), s) for p, s in zip(peak, corr.shape)]
    keep[np.ix_(*coords)] = False
    ambiguity = float(np.max(corr[keep]) / max(main, 1e-9))
    residual, grad_residual, valid_fraction = registered_residual(a, b, t)
    return dict(valid=True, t=t.tolist(), residual=residual, gradient_residual=grad_residual, ambiguity=ambiguity,
                valid_fraction=valid_fraction, large_shift=bool(np.max(abs(t)) > 5))


# ---- spatial NCC refinement ----
def refine_translation(a, b, initial_t, margin=6):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    initial_t = np.asarray(initial_t, dtype=float)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError('Equal shaped 3-D windows required')
    if np.max(abs(initial_t)) > margin - 2:
        return dict(success=False, t=initial_t.tolist(), reason='initializer_outside_refinement_support')
    fixed = a[(slice(margin, -margin),) * 3]
    if min(fixed.std(), b.std()) < 1e-8:
        return dict(success=False, t=initial_t.tolist(), reason='constant_window')
    grid = np.indices(fixed.shape, dtype=float) + margin
    coefficients = ndimage.spline_filter(b, order=3)

    def objective(t):
        moving = ndimage.map_coordinates(coefficients, grid + t[:, None, None, None], order=3, prefilter=False, mode='mirror')
        return ncc_residual(fixed, moving)
    bounds = [(max(-margin + 2, t - 1), min(margin - 2, t + 1)) for t in initial_t]
    fit = optimize.minimize(objective, initial_t, method='Powell', bounds=bounds, options={'xtol': 1e-4, 'ftol': 1e-8, 'maxiter': 30})
    return dict(success=bool(fit.success), t=fit.x.tolist(), objective=float(fit.fun),
                initial_objective=float(objective(initial_t)), evaluations=int(fit.nfev))
