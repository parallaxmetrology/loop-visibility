"""Whole-volume Diffeomorphic Demons wrapper used in the study (SimpleITK 2.5.8).

Same pyramid and settings as the preceding pilot's wrapper. fit(fixed, moving) returns a
displacement field on the fixed grid mapping fixed-image points to moving-image points.
"""
import time
import numpy as np
from scipy import ndimage
import SimpleITK as sitk

SETTINGS = dict(grid_spacing_mm=[8, 4, 2], image_smoothing_sigma_mm=[4, 2, 0.8], iterations=[80, 60, 40],
                total_field_sigma_pixels=1.0, update_field_sigma_pixels=0.5, maximum_update_step_pixels=0.5,
                maximum_rms_error=0.005, intensity_difference_threshold=0.007142857142857143, threads=4)


def image_from_array(array, spacing_xyz, origin=(0., 0., 0.)):
    image = sitk.GetImageFromArray(np.asarray(array, np.float32))
    image.SetSpacing(list(map(float, spacing_xyz))); image.SetOrigin(origin)
    return image


def ct_image(stored, spacing_xyz):
    """DIR-Lab stored values are HU + 1024; clip HU to [-1000, 400] and scale to [0, 1]."""
    hu = np.asarray(stored, np.float32) - 1024.0
    return image_from_array((np.clip(hu, -1000, 400) + 1000) / 1400, spacing_xyz)


def pyramid_image(image, spacing_mm, sigma_mm):
    extent = (np.array(image.GetSize()) - 1) * np.array(image.GetSpacing())
    size = (np.floor(extent / spacing_mm).astype(int) + 1).tolist()
    smoothed = sitk.SmoothingRecursiveGaussian(image, sigma_mm) if sigma_mm > 0 else image
    return sitk.Resample(smoothed, size, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear,
                         image.GetOrigin(), [float(spacing_mm)] * 3, image.GetDirection(), 0., sitk.sitkFloat32)


def fit(fixed, moving, settings=SETTINGS):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(settings['threads'])
    field = None; records = []
    for level, (sp, sigma, n) in enumerate(zip(settings['grid_spacing_mm'], settings['image_smoothing_sigma_mm'], settings['iterations'])):
        f = pyramid_image(fixed, sp, sigma); m = pyramid_image(moving, sp, sigma)
        if field is None:
            field = sitk.Image(f.GetSize(), sitk.sitkVectorFloat64, 3); field.CopyInformation(f)
        else:
            rs = sitk.ResampleImageFilter(); rs.SetReferenceImage(f); rs.SetInterpolator(sitk.sitkLinear)
            rs.SetUseNearestNeighborExtrapolator(True); field = rs.Execute(field)
        alg = sitk.DiffeomorphicDemonsRegistrationFilter()
        alg.SetNumberOfIterations(n); alg.SetMaximumRMSError(settings['maximum_rms_error'])
        alg.SetSmoothDisplacementField(True); alg.SetStandardDeviations(settings['total_field_sigma_pixels'])
        alg.SetSmoothUpdateField(True); alg.SetUpdateFieldStandardDeviations(settings['update_field_sigma_pixels'])
        alg.SetMaximumUpdateStepLength(settings['maximum_update_step_pixels'])
        alg.SetIntensityDifferenceThreshold(settings['intensity_difference_threshold'])
        alg.SetUseImageSpacing(True); alg.SetUseFirstOrderExp(False); alg.SetUseGradientType(alg.Symmetric)
        start = time.monotonic(); field = alg.Execute(f, m, field)
        records.append(dict(level=level, iterations=int(alg.GetElapsedIterations()), seconds=time.monotonic() - start))
    return field, records


def field_arrays(field):
    arr = sitk.GetArrayFromImage(field)
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field, True))
    return arr, jac


def sample(array, field, points_xyz):
    """Displacement (xyz, mm) at physical points; covered = inside the field grid."""
    pts = np.atleast_2d(points_xyz)
    direction = np.array(field.GetDirection()).reshape(3, 3)
    idx = ((pts - np.array(field.GetOrigin())) @ direction) / np.array(field.GetSpacing())
    coords = idx[:, ::-1].T
    values = np.column_stack([ndimage.map_coordinates(array[..., k], coords, order=1, mode='nearest') for k in range(3)])
    covered = np.all((idx >= 0) & (idx <= np.array(field.GetSize()) - 1), axis=1)
    return values, covered


def geometry_valid(field):
    arr, jac = field_arrays(field); inner = jac[1:-1, 1:-1, 1:-1]
    return bool(np.isfinite(arr).all() and np.isfinite(jac).all() and np.all(inner > 0))
