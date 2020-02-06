# -*- coding: utf-8 -*-
from __future__ import absolute_import

import scipy.ndimage

from ..denoise import median
from ..compound import CompoundRop
from ..base import PerChannelRop


class WhiteTophatFilterRop(PerChannelRop):

    size = 1

    def process_channel(self, data, detected=None):
        return scipy.ndimage.white_tophat(data, self.size)


class ExtractStarsRop(CompoundRop):

    def __init__(self, raw, **kw):
        median_size = kw.pop('median_size', 3)
        median_sigma = kw.pop('median_sigma', 1.0)
        star_size = kw.pop('star_size', 16)
        super(ExtractStarsRop, self).__init__(
            raw,
            WhiteTophatFilterRop(raw, size=star_size, **kw),
            median.MaskedMedianFilterRop(raw, size=median_size, sigma=median_sigma, **kw)
        )
