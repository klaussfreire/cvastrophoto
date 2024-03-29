# -*- coding: utf-8 -*-
import numpy
import numpy.linalg

from . import vectorize

if vectorize.with_numba:
    sigs = [
        'float32(float32, float32, float32, float32, float32, float32)',
        'float64(float64, float64, float64, float32, float32, float64)',
    ]

    import multiprocessing
    use_cuda = multiprocessing.cpu_count() < 6

    @vectorize.auto_vectorize(sigs, out_arg=5, size_arg=5, big_thresh=400000, cuda=use_cuda)
    def _to_srgb(gamma_recip, in_scale, out_scale, out_min, out_max, x):
        x *= in_scale
        if x >= 0.0031308:
            x = (1.055 * pow(x, gamma_recip) - 0.055) * out_scale
        else:
            x = x * out_scale * 12.92
        return max(out_min, min(out_max, x))

    @vectorize.auto_vectorize(sigs, out_arg=5, size_arg=5, big_thresh=400000, cuda=use_cuda)
    def _from_srgb(gamma, in_scale, out_scale, out_min, out_max, x):
        x *= in_scale
        if x >= 0.04045:
            x = pow((x+0.055) * (1.0/1.055), gamma) * out_scale
        else:
            x = x * out_scale * (1.0 / 12.92)
        return max(out_min, min(out_max, x))
else:
    # Pure-numpy fallbacks to use when numba is missing
    def _to_srgb(gamma_recip, in_scale, out_scale, out_min, out_max, raw_image):
        raw_image *= in_scale
        nonlinear_range = raw_image >= 0.0031308
        raw_image[raw_image < 0.0031308] *= 12.92
        raw_image[nonlinear_range] = 1.055 * numpy.power(
            raw_image[nonlinear_range],
            gamma_recip,
        ) - 0.055
        raw_image *= out_scale
        if out_min != 0.0 or out_max != float('inf'):
            raw_image = numpy.clip(raw_image, out_min, out_max, out=raw_image)
        return raw_image

    def _from_srgb(gamma, in_scale, out_scale, out_min, out_max, raw_image):
        raw_image *= in_scale
        nonlinear_range = raw_image >= 0.04045
        raw_image[raw_image < 0.04045] *= 1.0 / 12.92
        raw_image[nonlinear_range] = numpy.power(
            (raw_image[nonlinear_range]+0.055) * (1.0/1.055),
            gamma,
        )
        raw_image *= out_scale
        if out_min != 0.0 or out_max != float('inf'):
            raw_image = numpy.clip(raw_image, out_min, out_max, out=raw_image)
        return raw_image

def decode_srgb(raw_image, gamma=2.4, in_scale=None, out_scale=None, out_min=None, out_max=None):
    """
    Decodes srgb in-place in the normalized float raw_image
    """
    if raw_image.dtype.kind == 'u' and raw_image.dtype.itemsize <= 2:
        # For integer types, use lookup table, it's faster
        lut = numpy.arange(1 << (8 * raw_image.dtype.itemsize), dtype=numpy.float32)
        limits = numpy.iinfo(raw_image.dtype)
        if in_scale is None:
            in_scale = 1.0 / lut[-1]
        if out_scale is None:
            out_scale = lut[-1]
        if out_min is None:
            out_min = 0
        if out_max is None:
            out_max = limits.max
        out_max = min(out_max, lut[-1])
        lut = decode_srgb(lut, gamma, in_scale, out_scale, out_min, out_max).astype(raw_image.dtype)
        raw_image[:] = lut[raw_image]
    else:
        _from_srgb(gamma, in_scale or 1.0, out_scale or 1.0, out_min or 0.0, out_max or float('inf'), raw_image)
    return raw_image

def encode_srgb(raw_image, gamma=2.4, in_scale=None, out_scale=None, out_min=None, out_max=None):
    """
    Encodes srgb in-place in the normalized float raw_image
    """
    out_img = raw_image
    if raw_image.dtype.kind == 'i' and raw_image.dtype.itemsize <= 2 and raw_image.min() >= 0:
        # Cast to unsigned so we can use the LUT
        raw_image = raw_image.astype(raw_image.dtype.char.upper())
    if raw_image.dtype.kind == 'u' and raw_image.dtype.itemsize <= 2:
        # For integer types, use lookup table, it's faster
        lut = numpy.arange(1 << (8 * raw_image.dtype.itemsize), dtype=numpy.float32)
        limits = numpy.iinfo(raw_image.dtype)
        if in_scale is None:
            in_scale = 1.0 / lut[-1]
        if out_scale is None:
            out_scale = lut[-1]
        if out_min is None:
            out_min = 0
        if out_max is None:
            out_max = limits.max
        out_max = min(out_max, lut[-1])
        lut = encode_srgb(lut, gamma, in_scale, out_scale, out_min, out_max).astype(raw_image.dtype)
        raw_image[:] = lut[raw_image]
    else:
        if raw_image.dtype.kind != 'f':
            # vectorized _to_srgb does not support unsafe casting
            raw_image = raw_image.astype(numpy.float32)
        _to_srgb(1.0 / gamma, in_scale or 1.0, out_scale or 1.0, out_min or 0.0, out_max or float('inf'), raw_image)
        if raw_image is not out_img:
            out_img[:] = raw_image
    return out_img

def color_matrix(in_, matrix, out_, preserve_lum=False):
    if preserve_lum:
        from skimage import color
        dmax = in_.max()
        alum = numpy.average(in_)
        lum = color.rgb2lab(encode_srgb(in_ * (1.0 / dmax)))[:,:,0]

    out_[:,:,0] = in_[:,:,0] * matrix[0, 0]
    out_[:,:,0] += in_[:,:,1] * matrix[0, 1]
    out_[:,:,0] += in_[:,:,2] * matrix[0, 2]

    out_[:,:,1] = in_[:,:,0] * matrix[1, 0]
    out_[:,:,1] += in_[:,:,1] * matrix[1, 1]
    out_[:,:,1] += in_[:,:,2] * matrix[1, 2]

    out_[:,:,2] = in_[:,:,0] * matrix[2, 0]
    out_[:,:,2] += in_[:,:,1] * matrix[2, 1]
    out_[:,:,2] += in_[:,:,2] * matrix[2, 2]

    if preserve_lum:
        out_alum = numpy.average(out_)
        out_lch = color.rgb2lab(encode_srgb(out_ * (alum / (out_alum * dmax))))
        out_lch[:,:,0] = lum
        del lum
        out_[:] = decode_srgb(color.lab2rgb(out_lch)) * dmax
        del out_lch

    return out_

MATRIX_XYZ2RGB = numpy.array([
    [0.41847, -0.15866, -0.082835],
    [-0.091169, 0.25243, 0.015708],
    [0.0009209, -0.0025498, 0.17860],
], numpy.float32)

def xyz2rgb(in_, out_):
    return color_matrix(in_, MATRIX_XYZ2RGB, out_)

def camera2xyz(in_, rimg, out_):
    return color_matrix(in_, numpy.linalg.inv(rimg.rgb_xyz_matrix[:3,:3]), out_)

def camera2rgb(in_, rimg_or_matrix, out_, preserve_lum=False):
    if hasattr(rimg_or_matrix, 'rgb_xyz_matrix'):
        rgb_xyz_matrix = rimg_or_matrix.rgb_xyz_matrix
    else:
        rgb_xyz_matrix = rimg_or_matrix
    return color_matrix(
        in_,
        numpy.matmul(
            MATRIX_XYZ2RGB,
            numpy.linalg.inv(rgb_xyz_matrix[:3,:3]),
        ),
        out_,
        preserve_lum=preserve_lum)

def matrix_wb(matrix, wb, scale=1):
    matrix = matrix.copy()
    for i, f in enumerate(wb):
        matrix[i] *= f * scale
    return matrix
