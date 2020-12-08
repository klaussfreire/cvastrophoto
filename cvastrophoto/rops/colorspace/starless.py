# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy

from ..tracking.extraction import ExtractPureStarsRop
from .whitebalance import WhiteBalanceRop


class StarlessWhiteBalanceRop(WhiteBalanceRop):

    target = 'bg'

    def __init__(self, raw, **kw):
        self._extract_stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(ExtractPureStarsRop, k)}
        self._extract_stars_kw.setdefault('copy', False)
        super(StarlessWhiteBalanceRop, self).__init__(raw, **kw)

    def correct(self, data, *p, **kw):
        dmax = data.max()
        stars_rop = ExtractPureStarsRop(self.raw, **self._extract_stars_kw)
        stars = stars_rop.correct(data.copy())

        data -= numpy.clip(stars, None, data, out=stars)
        if self.target == 'bg':
            data = super(StarlessWhiteBalanceRop, self).correct(data, *p, dmax=dmax, **kw)
        elif self.target == 'stars':
            stars = super(StarlessWhiteBalanceRop, self).correct(stars, *p, dmax=dmax, **kw)
        else:
            raise ValueError("Invalid target %s" % (self.target,))
        data += numpy.clip(stars, None, dmax - data)
        return data


class StarWhiteBalanceRop(StarlessWhiteBalanceRop):
    target = 'stars'
