# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage
from ..base import BaseRop

class LocalGradientBiasRop(BaseRop):

    minfilter_size = 256
    gauss_size = 256
    pregauss_size = 4
    gain = 1

    def detect(self, data, quick=False, **kw):
        path, patw = self._raw_pattern.shape
        local_gradient = numpy.empty(data.shape, data.dtype)
        for y in xrange(path):
            for x in xrange(patw):
                local_gradient[y::path, x::patw] = scipy.ndimage.gaussian_filter(
                    scipy.ndimage.minimum_filter(
                        scipy.ndimage.gaussian_filter(data[y::path, x::patw], self.pregauss_size),
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
