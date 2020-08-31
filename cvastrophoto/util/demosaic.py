from __future__ import absolute_import

from past.builtins import xrange
import numpy
import scipy.ndimage.filters


def demosaic(raw_image, raw_pattern):
    if raw_pattern.size == 1:
        # Luminance data
        return raw_image.reshape(raw_image.shape + (1,))
    elif raw_pattern.size == 3 and (raw_image.shape[1] % 3) == 0:
        # RGB data
        return raw_image.reshape((raw_image.shape[0], raw_image.shape[1] / 3, 3))

    # Otherwise, multichannel data
    channels = raw_pattern.max() + 1

    postprocessed = numpy.zeros(raw_image.shape + (channels,), raw_image.dtype)
    mask = numpy.zeros(postprocessed.shape, numpy.bool8)

    path, patw = raw_pattern.shape
    for y in xrange(path):
        for x in xrange(patw):
            postprocessed[y::path, x::patw, raw_pattern[y,x]] = raw_image[y::path, x::patw]
            mask[y::path, x::patw, raw_pattern[y,x]] = True

    filtered = numpy.empty(postprocessed.shape, numpy.float32)
    filtered_mask = numpy.empty(mask.shape, numpy.float32)
    for c in xrange(channels):
        filtered[:,:,c] = scipy.ndimage.filters.uniform_filter(
            postprocessed[:,:,c].astype(filtered.dtype), (path, patw), mode='constant')
        filtered_mask[:,:,c] = scipy.ndimage.filters.uniform_filter(
            mask[:,:,c].astype(filtered_mask.dtype), (path, patw), mode='constant')

    filtered_mask = numpy.clip(filtered_mask, 0.001, None, out=filtered_mask)
    filtered /= filtered_mask
    filtered = numpy.clip(filtered, raw_image.min(), raw_image.max(), out=filtered)

    postprocessed[~mask] = filtered[~mask]

    return postprocessed


def remosaic(image, raw_pattern, out=None):
    if raw_pattern.size == 1:
        # Luminance data
        image = image.reshape(image.shape[:2])
        if out is not None:
            out[:] = image
            return out
        else:
            return image
    elif raw_pattern.size == 3:
        image = image.reshape((image.shape[0], image.shape[1] * 3))
        if out is not None:
            out[:] = image
            return out
        else:
            return image

    # Otherwise, multichannel data
    if out is None:
        raw_image = numpy.empty(image.shape[:2])
    else:
        raw_image = out

    path, patw = raw_pattern.shape
    for y in xrange(path):
        for x in xrange(patw):
            raw_image[y::path, x::patw] = image[y::path, x::patw, raw_pattern[y,x]]

    return raw_image
