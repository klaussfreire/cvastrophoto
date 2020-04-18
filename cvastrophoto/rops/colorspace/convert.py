# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
from skimage import color

from ..base import BaseRop
from cvastrophoto.util import demosaic


def rgb2l(d):
    out = numpy.empty(d.shape, d.dtype)
    out[:,:,0] = color.rgb2gray(d)
    for c in xrange(1, out.shape[2]):
        out[:,:,c] = out[:,:,0]
    return out


def rgb2ciel(d):
    out = color.rgb2lab(d)
    for c in xrange(1, out.shape[2]):
        out[:,:,c] = out[:,:,0]
    return out


def ciel2rgb(d):
    d = d.copy()
    d[:,:,1:] = 0
    d = color.lab2rgb(d)
    return d


class ColorspaceConversionRop(BaseRop):

    ccfrom = 'RGB'
    ccto = 'CIE'

    CCMAP = {
        'CIE-RGB': 'RGB CIE',
        'LAB': 'CIE-LAB',
        'LCH': 'CIE-LCH',
    }

    SPECIAL = {
        ('RGB', 'CIE-LAB'): color.rgb2lab,
        ('CIE-LAB', 'RGB'): color.lab2rgb,
        ('RGB', 'CIE-LCH'): lambda d: color.lab2lch(color.rgb2lab(d)),
        ('CIE-LCH', 'RGB'): lambda d: color.lab2rgb(color.lch2lab(d)),
        ('RGB', 'L'): rgb2l,
        ('L', 'RGB'): lambda d: d,
        ('RGB', 'CIEL'): rgb2ciel,
        ('CIEL', 'RGB'): ciel2rgb,
    }

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        raw_pattern = self._raw_pattern

        roi = kw.get('roi')

        ccfrom = self.ccfrom
        ccfrom = self.CCMAP.get(ccfrom.upper(), ccfrom)
        ccto = self.ccto
        ccto = self.CCMAP.get(ccto.upper(), ccto)

        def process_data(data):
            if roi is not None:
                data, eff_roi = self.roi_precrop(roi, data)

            ppdata = demosaic.demosaic(data, raw_pattern)

            if ppdata.dtype.kind == 'f':
                # skimage requires normalized float data
                scale = ppdata.max()
                if scale != 0:
                    ppdata = ppdata * (1.0 / scale)
            else:
                scale = None

            if (ccfrom, ccto) in self.SPECIAL:
                ppdata = self.SPECIAL[(ccfrom, ccto)](ppdata)
            else:
                ppdata = color.convert_colorspace(ppdata, ccfrom, ccto)

            if scale:
                ppdata *= scale
            data = demosaic.remosaic(ppdata, raw_pattern, out=data)

            if roi is not None:
                data = self.roi_postcrop(roi, eff_roi, data)

            return data

        rv = data

        if not isinstance(data, list):
            data = [data]

        for sdata in data:
            if sdata is None:
                continue

            sdata = process_data(sdata)

        return rv