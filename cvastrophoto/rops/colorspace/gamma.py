# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy

from ..base import PerChannelRop


class GammaRop(PerChannelRop):

    gamma = 1.0

    def process_channel(self, data, detected=None, channel=None, **kw):
        origdt = data.dtype
        dmax = data.max()
        if dmax == 0:
            dmax = 1
        data = data.astype(numpy.float64, copy=False)
        data *= 1.0 / dmax
        data = numpy.power(data, 1.0 / self.gamma, out=data, where=data > 0)
        data *= dmax
        return data.astype(origdt, copy=False)
