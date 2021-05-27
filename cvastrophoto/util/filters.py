from __future__ import absolute_import

import math
import numpy
import scipy.ndimage
import skimage.morphology

from . import vectorize


if vectorize.with_numba:
    fsigs = [
        'float32(float32, float32)',
        'float64(float64, float64)',
    ]
    isigs = [
        'uint8(uint8, uint8)',
        'int8(int8, int8)',
        'uint16(uint16, uint16)',
        'int16(int16, int16)',
        'uint32(uint32, uint32)',
        'int32(int32, int32)',
        'uint64(uint64, uint64)',
        'int64(int64, int64)',
    ]
    @vectorize.auto_vectorize(['float32(float32, float32)'], cuda=False)
    def std_from_mean_and_sqmean(mean, sqmean):
        return math.sqrt(max(0, sqmean - mean * mean))

    @vectorize.auto_vectorize(
        [
            'uint8(uint8, float32, uint8, uint8)',
            'int8(int8, float32, int8, int8)',
            'uint16(uint16, float32, uint16, uint16)',
            'int16(int16, float32, int16, int16)',
            'uint32(uint32, float64, uint32, uint32)',
            'int32(int32, float64, int32, int32)',
            'uint64(uint64, float64, uint64, uint64)',
            'int64(int64, float64, int64, int64)',
            'float32(float32, float32, float32, float32)',
            'float64(float64, float64, float64, float64)',
        ],
        cuda=False,
        out_arg=0)
    def _scale_and_clip_flt(x, scale, mn, mx):
        return min(mx, max(mn, x * scale))

    @vectorize.auto_vectorize(
        [
            'uint8(uint8, uint64, uint8, uint8, uint8)',
            'uint16(uint16, uint64, uint8, uint16, uint16)',
            'uint32(uint32, uint64, uint8, uint32, uint32)',
        ],
        cuda=False,
        out_arg=0)
    def _scale_and_clip_fix(x, scale, shift, mn, mx):
        return min(mx, max(mn, (x * scale) >> shift))
else:
    def std_from_mean_and_sqmean(mean, sqmean, out=None):
        out = numpy.subtract(sqmean, numpy.square(mean), out=out)
        out = numpy.clip(out, 0, None, out=out)
        out = numpy.sqrt(out, out=out)
        return out

    def _scale_and_clip_flt(x, scale, mn, mx):
        if x.dtype.kind in 'ui':
            numpy.clip(x * scale, mn, mx, out=x)
        else:
            x *= scale
            numpy.clip(x, mn, mx, out=x)
        return x

    def _scale_and_clip_fix(x, scale, shift, mn, mx):
        return _scale_and_clip_flt(x, scale / float(1 << shift), mn, mx)


def local_moments(image, footprint, **kw):
    if isinstance(footprint, int):
        filter_fn = scipy.ndimage.uniform_filter
        footprint = footprint * 2 + 1
    else:
        filter_fn = scipy.ndimage.convolve
        footprint = footprint.astype(numpy.float32) / footprint.sum()
    center = filter_fn(image, footprint, output=numpy.float32, **kw)
    std = filter_fn(numpy.square(image), footprint, output=numpy.float32, **kw)
    std = std_from_mean_and_sqmean(center, std, out=std)
    return center, std


def quick_despeckle(image, footprint, **kw):
    """
    Computes an approximation of a median filter with a robust mean.
    Considerably faster than a median filter on large footprints.

    :param footprint: If given as an integer, it uses a square footprint of
        the given "radius", which enables use of a fast separable filter. If given
        as a boolean matrix, it will use the normalized matrix for a more
        accurate but slightly slower filter.
    """
    # Compute local average and stddev to remove outliers
    center, std = local_moments(image, footprint, **kw)

    # Neutralize outliers
    despeckled = image.copy()
    mask = image < (center - std) ; despeckled[mask] = center[mask] ; del mask
    mask = image > (center + std) ; despeckled[mask] = center[mask] ; del mask
    del center, std

    if isinstance(footprint, int):
        filter_fn = scipy.ndimage.uniform_filter
        footprint = footprint * 2 + 1
    else:
        filter_fn = scipy.ndimage.convolve
        footprint = footprint.astype(numpy.float32) / footprint.sum()
    return filter_fn(despeckled, footprint, output=image.dtype, **kw)


def scale_and_clip(x, scale, mn=None, mx=None, out=None):
    if out is x:
        x = out
    elif out is not None:
        out[:] = x
    else:
        out = x.copy()

    dtype = out.dtype
    dtkind = dtype.kind

    if dtkind in 'ui':
        limits = numpy.iinfo(dtype)
    elif dtkind == 'f':
        limits = numpy.finfo(dtype)
    else:
        limits = None
    emn = mn
    emx = mx
    if emn is None:
        emn = limits.min
    if emx is None:
        emx = limits.max

    if dtkind in 'fi':
        if dtkind == 'f' and mn is None and mx is None:
            # Fast path, float, no explicit limits
            out *= scale
        else:
            return _scale_and_clip_flt(out, scale, emn, emx)
    elif dtkind == 'u':
        # uints have fixed point implementations
        itemsize = out.dtype.itemsize
        if itemsize > 4:
            iscale = None
        else:
            shift = itemsize * 8
            iscale = int(scale * (1 << shift))
            if iscale > (0xFFFFFFFFFFFFFFFF >> shift):
                # Would overflow
                iscale = None

        if iscale is not None:
            return _scale_and_clip_fix(out, iscale, shift, emn, emx)
        else:
            # Fall back to float
            return _scale_and_clip_flt(out, scale, emn, emx)
    else:
        raise NotImplementedError(dtype)
