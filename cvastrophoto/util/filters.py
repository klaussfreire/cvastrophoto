from __future__ import absolute_import

import math
import numpy
import scipy.ndimage
import skimage.morphology

from . import vectorize


if vectorize.with_numba:
    @vectorize.auto_vectorize(['float32(float32, float32)'], cuda=False)
    def std_from_mean_and_sqmean(mean, sqmean):
        return math.sqrt(max(0, sqmean - mean * mean))
else:
    def std_from_mean_and_sqmean(mean, sqmean, out=None):
        out = numpy.subtract(sqmean, numpy.square(mean), out=out)
        out = numpy.clip(out, 0, None, out=out)
        out = numpy.sqrt(out, out=out)
        return out


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
