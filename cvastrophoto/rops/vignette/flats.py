from __future__ import absolute_import

from ..base import BaseRop

import numpy

class FlatImageRop(BaseRop):

    def __init__(self, raw=None, flat=None, color=False):
        self.flat = flat
        super(FlatImageRop, self).__init__(raw)

    def detect(self, data):
        pass

    def correct(self, data, flat=None):
        if flat is None:
            flat = self.flat

        return self.flatten(data, flat)

    def flatten(self, light, flat, dtype=numpy.float32, scale=None):
        flattened = light.astype(numpy.float32) / self.raw.luma_image(flat)
        flattened *= 1.0 / flattened.max()
        flattened = numpy.clip(flattened, 0, 1, out=flattened)

        if scale is not None:
            flattened *= scale
        if dtype is not numpy.float32:
            flattened = flattened.astype(dtype)

        return flattened
