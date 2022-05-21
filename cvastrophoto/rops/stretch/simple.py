# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy

from .. import base

from cvastrophoto.util import demosaic

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


class ColorimetricStretchRop(base.BaseRop):

    bright = 4.0

    @staticmethod
    def colorimetric_stretch(data, bright, dmax, reclip):
        if bright != 1:
            data *= bright
        r = data[:,:,0]
        g = data[:,:,1]
        b = data[:,:,2]
        mx = numpy.maximum(numpy.maximum(r, g), b)
        clipmask = mx > dmax
        if numpy.any(clipmask):
            mxclip = mx[clipmask]
            cliplev = numpy.log2(mxclip / dmax)
            cliplev += 1
            rr = r[clipmask] / mxclip
            rg = g[clipmask] / mxclip
            rb = b[clipmask] / mxclip
            r[clipmask] = dmax * (1.0 - (1-0 - rr) / cliplev)
            g[clipmask] = dmax * (1.0 - (1-0 - rg) / cliplev)
            b[clipmask] = dmax * (1.0 - (1-0 - rb) / cliplev)
        if reclip:
            data = numpy.clip(data, 0, dmax, out=data)
        return data

    def correct(self, data, detected=None, dmax=None, **kw):
        odata = data
        raw_pattern = self._raw_pattern
        data = demosaic.demosaic(data, raw_pattern)
        same_base = data.base is odata
        if dmax is None:
            dmax = data.max()

        if data.dtype.kind not in 'fd':
            data = data.astype(numpy.float32)
            dmax = float(dmax)
            same_base = False

        data = self.colorimetric_stretch(data, self.bright, dmax, True)

        if not same_base:
            demosaic.remosaic(data, raw_pattern, out=odata)
        return odata
