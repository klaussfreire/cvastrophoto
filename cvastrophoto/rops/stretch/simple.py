# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import logging
import numpy
import math

from .. import base

from cvastrophoto.util import demosaic, vectorize, filters


logger = logging.getLogger(__name__)


if vectorize.with_numba:
    @vectorize.auto_guvectorize(
        [
            'float32[:], float32, float32, float32, float32, float32[:]',
            'float64[:], float32, float32, float32, float32, float64[:]',
        ],
        '(n), (), (), (), () -> (n)',
        tile_param=0,
    )
    def _colorimetric_stretch(data, bright, dmax, clipmin, clipmax, out):
        r = data[0] * bright
        g = data[1] * bright
        b = data[2] * bright
        mx = max(r, g, b)
        if mx > dmax:
            cliplev = math.log(mx / dmax) / math.log(2) + 1
            desat = 1.0 / cliplev
            rr = r / mx
            rg = g / mx
            rb = b / mx
            r = r * (1.0 - desat) + (dmax * rr) * desat
            g = g * (1.0 - desat) + (dmax * rg) * desat
            b = b * (1.0 - desat) + (dmax * rb) * desat
        out[0] = min(clipmax, max(clipmin, r))
        out[1] = min(clipmax, max(clipmin, g))
        out[2] = min(clipmax, max(clipmin, b))
else:
    def _colorimetric_stretch(data, bright, dmax, clipmin, clipmax, out):
        if bright != 1:
            data = numpy.multiply(data, bright, out=out)
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
        return numpy.clip(data, clipmin, clipmax, out=data)


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
        if reclip:
            clipmin = 0.0
            clipmax = float(dmax)
        else:
            clipmin = float('-inf')
            clipmax = float('inf')
        return _colorimetric_stretch(data, float(bright), float(dmax), clipmin, clipmax, out=data)

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


class AutoStretchRop(base.BaseRop):

    bright = 1.0
    p_hi = 99.5
    p_lo = 2.0
    white = 65535.0

    def correct(self, data, detected=None, dmax=None, **kw):
        if dmax is None:
            if data.dtype.kind in 'iu':
                dmax = numpy.iinfo(data.dtype).max
            else:
                dmax = max(65535, data.max())
        if self._raw_pattern.max() > 0:
            ldata = self.raw.luma_image(data, dtype=numpy.float32, renormalize=True, raw_pattern=self._raw_pattern)
            orig_ldata = ldata.copy()
        else:
            ldata = data

        v_lo, v_hi = numpy.percentile(ldata, (self.p_lo, self.p_hi))
        scale = self.white * self.bright / (v_hi - v_lo)
        if ldata.dtype.kind in 'fd':
            ldata -= v_lo
        else:
            ldata -= numpy.minimum(ldata, v_lo).astype(ldata.dtype, copy=False)
        ldata = filters.scale_and_clip(ldata, scale, 0, dmax, out=ldata)

        if ldata is not data:
            ldata = numpy.divide(ldata, orig_ldata, out=ldata, where=orig_ldata > 0)
            ldata[orig_ldata <= 0] = 0
            data = numpy.clip(data * ldata, 0, dmax, out=data)

        return data


class AutoBlackRop(base.BaseRop):

    p_lo = 0.0
    r_lo = 0.0

    def correct(self, data, detected=None, dmax=None, **kw):
        if self._raw_pattern.max() > 0:
            ldata = self.raw.luma_image(data, dtype=numpy.float32, renormalize=True, raw_pattern=self._raw_pattern)
            orig_ldata = ldata.copy()
        else:
            ldata = data

        if self.r_lo:
            v_lo = ldata.max() * self.r_lo
        else:
            v_lo = numpy.percentile(ldata, self.p_lo)
        if ldata.dtype.kind in 'fd':
            ldata -= v_lo
        else:
            ldata -= numpy.minimum(ldata, v_lo).astype(ldata.dtype, copy=False)

        if ldata is not data:
            ldata = numpy.divide(ldata, orig_ldata, out=ldata, where=orig_ldata > 0)
            ldata[orig_ldata <= 0] = 0
            data = numpy.clip(data * ldata, 0, dmax, out=data)

        return data
