from __future__ import absolute_import

import numpy
from ..base import BaseRop

class UniformBiasRop(BaseRop):

    iterations = 8

    def detect(self, data, **kw):
        return (
            self._detect_bias(data, self.rmask_image),
            self._detect_bias(data, self.gmask_image),
            self._detect_bias(data, self.bmask_image),
        )

    def _detect_bias(self, data, mask):
        sizes = self.raw.rimg.sizes
        data = data[
            sizes.top_margin:sizes.top_margin+sizes.iheight,
            sizes.left_margin:sizes.left_margin+sizes.iwidth]
        mask = mask[
            sizes.top_margin:sizes.top_margin+sizes.iheight,
            sizes.left_margin:sizes.left_margin+sizes.iwidth]
        noise = data[mask]
        bias = numpy.average(noise)
        for i in xrange(self.iterations):
            darks = noise < bias
            if not darks.any():
                return
            bias = numpy.average(noise[darks]).astype(noise.dtype)
        return bias

    def correct(self, data, bias=None, **kw):
        if bias is None:
            bias = self.detect(data)
        if bias is not None:
            if bias[0] is not None:
                data[self.rmask_image] -= numpy.minimum(data[self.rmask_image], bias[0])
            if bias[1] is not None:
                data[self.gmask_image] -= numpy.minimum(data[self.gmask_image], bias[1])
            if bias[2] is not None:
                data[self.bmask_image] -= numpy.minimum(data[self.bmask_image], bias[2])
        return data