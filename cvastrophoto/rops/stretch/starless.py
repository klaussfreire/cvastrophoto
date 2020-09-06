# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy

from ..tracking.extraction import ExtractPureStarsRop
from .simple import LinearStretchRop
from .hdr import HDRStretchRop
from cvastrophoto.util import gaussian


class StarlessLinearStretchRop(LinearStretchRop):

    mask_sigma = None
    shrink = 0.75

    def __init__(self, raw, **kw):
        self._extract_stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(ExtractPureStarsRop, k)}
        super(StarlessLinearStretchRop, self).__init__(raw, **kw)

    def correct(self, data, *p, **kw):
        dmax = data.max()
        stars_rop = ExtractPureStarsRop(self.raw, **self._extract_stars_kw)
        stars = stars_rop.correct(data.copy())

        # Mask for smooth transition to avoid "soft clipping"
        dnz = data > 0
        weights = stars.astype(numpy.float32)
        weights[dnz] /= data[dnz]
        weights[~dnz] = 1
        star_bright = (1 + numpy.clip(1 - weights, 0, 1) * (self.bright - 1))
        if self.shrink < 1:
            star_bright = gaussian.fast_gaussian(star_bright, int(stars_rop.star_size * (1 - self.shrink)))
        del weights

        data -= numpy.clip(stars, None, data, out=stars)
        data = super(StarlessLinearStretchRop, self).correct(data, *p, dmax=dmax, **kw)
        data += numpy.clip(stars * star_bright, None, dmax - data)
        return data


class StarlessHDRStretchRop(HDRStretchRop):

    mask_sigma = None
    rescale = False

    shrink = 0.75
    bg_bright = 1.0
    star_bright = 1.0

    def __init__(self, raw, **kw):
        self._extract_stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(ExtractPureStarsRop, k)}
        super(StarlessHDRStretchRop, self).__init__(raw, **kw)

    def correct(self, data, *p, **kw):
        dmax = data.max()
        stars_rop = ExtractPureStarsRop(self.raw, **self._extract_stars_kw)
        stars = stars_rop.correct(data.copy())

        if self.bg_bright > 1:
            # Mask for smooth transition to avoid "soft clipping"
            dnz = data > 0
            weights = stars.astype(numpy.float32)
            weights[dnz] /= data[dnz]
            weights[~dnz] = 1
            hdr_star_bright = 1
            star_bright = (self.star_bright + numpy.clip(1 - weights, 0, 1) * (self.bg_bright - self.star_bright))
            if self.shrink < 1:
                star_bright = gaussian.fast_gaussian(star_bright, int(stars_rop.star_size * (1 - self.shrink)))
            del weights
        else:
            hdr_star_bright = self.star_bright
            star_bright = 1

        data -= numpy.clip(stars, None, data, out=stars)
        data = super(StarlessHDRStretchRop, self).correct(
            data, *p, dmax=dmax, bright=self.bg_bright, **kw)
        stars = super(StarlessHDRStretchRop, self).correct(
            stars * star_bright, *p, dmax=dmax, bright=hdr_star_bright, **kw)
        data += numpy.clip(stars, None, dmax - data)
        return data
