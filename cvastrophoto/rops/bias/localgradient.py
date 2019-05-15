# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage
import skimage.filters
import skimage.morphology
from ..base import BaseRop

class LocalGradientBiasRop(BaseRop):

    minfilter_size = 256
    gauss_size = 256
    pregauss_size = 4
    gain = 1
    offset = -1
    despeckle = True

    def detect(self, data, quick=False, **kw):
        path, patw = self._raw_pattern.shape
        if data.dtype.kind not in ('i', 'u'):
            dt = data.dtype
        else:
            dt = numpy.int32
        local_gradient = numpy.empty(data.shape, dt)
        data = self.raw.demargin(data.copy())
        def compute_local_gradient(task):
            data, local_gradient, y, x = task
            despeckled = data[y::path, x::patw]
            if self.despeckle:
                if quick:
                    despeckled = scipy.ndimage.maximum_filter(despeckled, 2)
                else:
                    despeckled = skimage.filters.median(despeckled, skimage.morphology.disk(1))
            if self.pregauss_size:
                despeckled = scipy.ndimage.gaussian_filter(despeckled, self.pregauss_size)
            grad = scipy.ndimage.gaussian_filter(
                scipy.ndimage.minimum_filter(despeckled, self.minfilter_size),
                min(8, self.gauss_size) if quick else self.gauss_size
            ) * self.gain

            local_gradient[y::path, x::patw] = grad
            local_gradient[y::path, x::patw] += self.offset

        if self.raw.default_pool is not None:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map

        tasks = []
        for y in xrange(path):
            for x in xrange(patw):
                tasks.append((data, local_gradient, y, x))
        for _ in map_(compute_local_gradient, tasks):
            pass

        return local_gradient

    def correct(self, data, local_gradient=None, quick=False, **kw):
        if local_gradient is None:
            local_gradient = self.detect(data, quick=quick)

        # Remove local gradient + offset into wider signed buffer to avoid over/underflow
        debiased = data.astype(local_gradient.dtype)
        debiased -= local_gradient

        # At this point, we may have out-of-bounds values due to
        # offset headroom. We have to clip the result, but carefully
        # to avoid numerical issues when close to data type limits.
        clip_min = 0
        clip_max = None
        if data.dtype.kind in ('i', 'u'):
            diinfo = numpy.iinfo(data.dtype)
            giinfo = numpy.iinfo(local_gradient.dtype)
            if diinfo.max < giinfo.max:
                clip_max = diinfo.max
        debiased = numpy.clip(debiased, clip_min, clip_max, out=debiased)

        # Copy into data buffer, casting back to data.dtype in the process
        data[:] = debiased
        return data


class PerFrameLocalGradientBiasRop(LocalGradientBiasRop):

    offset = -300
    despeckle = False
