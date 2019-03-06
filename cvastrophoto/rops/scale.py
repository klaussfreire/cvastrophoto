# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy

from .base import BaseRop

class ScaleRop(BaseRop):

    def __init__(self, raw, scale, dtype, clamp_min=None, clamp_max=None):
        self.scale = scale
        self.dtype = dtype
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        super(ScaleRop, self).__init__(raw)

    def detect(self, data):
        pass

    def correct(self, data, detected=None):
        data *= self.scale
        if self.clamp_min or self.clamp_max:
            data = numpy.clip(data, self.clamp_min, self.clamp_max, out=data)
        return data.astype(self.dtype)
