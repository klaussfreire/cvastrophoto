# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy

from .. import base

logger = logging.getLogger(__name__)

class LinearStretchRop(base.BaseRop):

    bright = 4.0

    def correct(self, data, detected=None, dmax=None, **kw):
        if dmax is None:
            dmax = data.max()
        if data.dtype.kind in 'fd':
            data *= self.bright
            data = numpy.clip(data, 0, dmax, out=data)
        else:
            data = numpy.clip(data * self.bright, 0, dmax, out=data)

        return data
