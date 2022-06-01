# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy

from ..base import BaseRop
from cvastrophoto.util import demosaic


class SaturationRop(BaseRop):

    sat = 1.0

    def correct(self, data, detected=None, **kw):
        ppdata = demosaic.demosaic(data, self._raw_pattern).astype(numpy.float32, copy=False)

        if ppdata.shape[2] > 1:
            dmax = numpy.max(ppdata, axis=2)
            dmax[dmax == 0] = 1
            for c in range(ppdata.shape[2]):
                ppdata[:,:,c] /= dmax
            ppdata = numpy.clip(ppdata, 0, 1, out=ppdata)
            ppdata = numpy.power(ppdata, self.sat, out=ppdata)
            for c in range(ppdata.shape[2]):
                ppdata[:,:,c] *= dmax

        return demosaic.remosaic(ppdata, self._raw_pattern, out=data)
