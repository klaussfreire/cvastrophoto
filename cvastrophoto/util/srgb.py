# -*- coding: utf-8 -*-
import numpy
import numpy.linalg

from . import vectorize

if vectorize.with_numba:
    sigs = [
        'float32(float32, float32, float32, float32)',
        'float64(float64, float64, float64, float64)',
    ]

    import multiprocessing
    use_cuda = multiprocessing.cpu_count() < 6

    @vectorize.auto_vectorize(sigs, out_arg=3, size_arg=3, big_thresh=400000, cuda=use_cuda)
    def _to_srgb(gamma_recip, in_scale, out_scale, x):
        x *= in_scale
        if x >= 0.0031308:
            return (1.055 * pow(x, gamma_recip) - 0.055) * out_scale
        else:
            return x * out_scale * 12.92

    @vectorize.auto_vectorize(sigs, out_arg=3, size_arg=3, big_thresh=400000, cuda=use_cuda)
    def _from_srgb(gamma, in_scale, out_scale, x):
        x *= in_scale
        if x >= 0.04045:
            return pow((x+0.055) * (1.0/1.055), gamma) * out_scale
        else:
            return x * out_scale * (1.0 / 12.92)
else:
    # Pure-numpy fallbacks to use when numba is missing
    def _to_srgb(gamma_recip, in_scale, out_scale, raw_image):
        raw_image *= in_scale
        nonlinear_range = raw_image >= 0.0031308
        raw_image[raw_image < 0.0031308] *= 12.92
        raw_image[nonlinear_range] = 1.055 * numpy.power(
            raw_image[nonlinear_range],
            gamma_recip,
        ) - 0.055
        raw_image *= out_scale
        return raw_image

    def _from_srgb(gamma, in_scale, out_scale, raw_image):
        raw_image *= in_scale
        nonlinear_range = raw_image >= 0.04045
        raw_image[raw_image < 0.04045] *= 1.0 / 12.92
        raw_image[nonlinear_range] = numpy.power(
            (raw_image[nonlinear_range]+0.055) * (1.0/1.055),
            gamma,
        )
        raw_image *= out_scale
        return raw_image

def decode_srgb(raw_image, gamma=2.4, in_scale=None, out_scale=None):
    """
    Decodes srgb in-place in the normalized float raw_image
    """
    if raw_image.dtype.kind == 'u' and raw_image.dtype.itemsize <= 2:
        # For integer types, use lookup table, it's faster
        lut = numpy.arange(1 << (8 * raw_image.dtype.itemsize), dtype=numpy.float32)
        if in_scale is None:
            in_scale = 1.0 / lut[-1]
        if out_scale is None:
            out_scale = lut[-1]
        lut = numpy.clip(decode_srgb(lut, gamma, in_scale, out_scale), 0, lut[-1]).astype(raw_image.dtype)
        raw_image[:] = lut[raw_image]
    else:
        _from_srgb(gamma, in_scale or 1.0, out_scale or 1.0, raw_image)
    return raw_image

def encode_srgb(raw_image, gamma=2.4, in_scale=None, out_scale=None):
    """
    Encodes srgb in-place in the normalized float raw_image
    """
    if raw_image.dtype.kind == 'u' and raw_image.dtype.itemsize <= 2:
        # For integer types, use lookup table, it's faster
        lut = numpy.arange(1 << (8 * raw_image.dtype.itemsize), dtype=numpy.float32)
        if in_scale is None:
            in_scale = 1.0 / lut[-1]
        if out_scale is None:
            out_scale = lut[-1]
        lut = numpy.clip(encode_srgb(lut, gamma, in_scale, out_scale), 0, lut[-1]).astype(raw_image.dtype)
        raw_image[:] = lut[raw_image]
    else:
        _to_srgb(1.0 / gamma, in_scale or 1.0, out_scale or 1.0, raw_image)
    return raw_image

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
