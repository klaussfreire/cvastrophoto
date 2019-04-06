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

    def detect(self, data, quick=False, **kw):
        path, patw = self._raw_pattern.shape
        local_gradient = numpy.empty(data.shape, data.dtype)
        data = self.raw.demargin(data)
        for y in xrange(path):
            for x in xrange(patw):
                if quick:
                    despeckled = scipy.ndimage.maximum_filter(data[y::path, x::patw], 2)
                else:
                    despeckled = skimage.filters.median(data[y::path, x::patw], skimage.morphology.disk(1))
                local_gradient[y::path, x::patw] = scipy.ndimage.gaussian_filter(
                    scipy.ndimage.minimum_filter(
                        scipy.ndimage.gaussian_filter(despeckled, self.pregauss_size),
                        self.minfilter_size
                    ),
                    min(8, self.gauss_size) if quick else self.gauss_size
                ) * self.gain
        return local_gradient

    def correct(self, data, local_gradient=None, quick=False, **kw):
        if local_gradient is None:
            local_gradient = self.detect(data, quick=quick)
        data -= numpy.minimum(data, local_gradient)
        return data
