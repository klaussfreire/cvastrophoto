from __future__ import absolute_import

import numpy
from ..base import BaseRop

class UniformBiasRop(BaseRop):

    iterations = 2

    def detect(self, data):
        return (
            self._detect_bias(data, self.rmask_image),
            self._detect_bias(data, self.gmask_image),
            self._detect_bias(data, self.bmask_image),
        )

    def _detect_bias(self, data, mask):
        noise = data[mask]
        bias = numpy.average(noise)
        for i in xrange(self.iterations):
            bias = numpy.average(noise[noise < bias])
        return bias

    def correct(self, data, bias=None):
        if bias is None:
            bias = self.detect(data)
        data[self.rmask_image] -= numpy.minimum(data[self.rmask_image], bias[0])
        data[self.gmask_image] -= numpy.minimum(data[self.gmask_image], bias[1])
        data[self.bmask_image] -= numpy.minimum(data[self.bmask_image], bias[2])
        return data