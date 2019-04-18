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
        local_gradient = numpy.empty(data.shape, data.dtype)
        data = self.raw.demargin(data)
        def compute_local_gradient(task):
            data, local_gradient, y, x = task
            if self.despeckle:
                if quick:
                    despeckled = scipy.ndimage.maximum_filter(data[y::path, x::patw], 2)
                else:
                    despeckled = skimage.filters.median(data[y::path, x::patw], skimage.morphology.disk(1))
            else:
                despeckled = data[y::path, x::patw]
            if self.pregauss_size:
                despeckled = scipy.ndimage.gaussian_filter(despeckled, self.pregauss_size)
            local_gradient[y::path, x::patw] = scipy.ndimage.gaussian_filter(
                scipy.ndimage.minimum_filter(despeckled, self.minfilter_size),
                min(8, self.gauss_size) if quick else self.gauss_size
            ) * self.gain

            offset = self.offset
            if local_gradient.dtype.kind not in ('i', 'u'):
                local_gradient[y::path, x::patw] += offset
            elif offset < 0:
                iinfo = numpy.iinfo(local_gradient.dtype)
                local_gradient[y::path, x::patw][local_gradient[y::path, x::patw] <= iinfo.min-offset] = 0
                local_gradient[y::path, x::patw][local_gradient[y::path, x::patw] > iinfo.min-offset] -= -offset
            elif offset > 0:
                iinfo = numpy.iinfo(local_gradient.dtype)
                local_gradient[y::path, x::patw][local_gradient[y::path, x::patw] < iinfo.max-offset] += offset
                local_gradient[y::path, x::patw][local_gradient[y::path, x::patw] >= iinfo.max-offset] = iinfo.max

        if self.raw.default_pool is not None:
            map_ = self.raw.default_pool.map
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
        data -= numpy.minimum(data, local_gradient)
        return data
