from __future__ import absolute_import

import numpy
import skimage.filters
import skimage.morphology
import scipy.ndimage

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


class UniformFloorRemovalRop(UniformBiasRop):

    pregauss_size = 2
    offset = -1

    def detect(self, data, **kw):
        path, patw = self._raw_pattern.shape
        despeckled = self.raw.demargin(data.copy())
        for y in xrange(path):
            for x in xrange(patw):
                component = despeckled[y::path, x::patw]
                component[:] = skimage.filters.median(component, skimage.morphology.disk(1))
                component[:] = scipy.ndimage.gaussian_filter(component, self.pregauss_size, mode='nearest')

        return (
            self._detect_bias(despeckled, self.rmask_image),
            self._detect_bias(despeckled, self.gmask_image),
            self._detect_bias(despeckled, self.bmask_image),
        )

    def _detect_bias(self, data, mask):
        sizes = self.raw.rimg.sizes
        data = data[
            sizes.top_margin:sizes.top_margin+sizes.iheight,
            sizes.left_margin:sizes.left_margin+sizes.iwidth]
        mask = mask[
            sizes.top_margin:sizes.top_margin+sizes.iheight,
            sizes.left_margin:sizes.left_margin+sizes.iwidth]
        rv = data[mask].min()
        if rv < 0:
            return rv + self.offset
        else:
            return max(0, rv + self.offset)
