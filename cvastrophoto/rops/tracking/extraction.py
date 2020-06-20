# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage

from ..denoise import median
from ..compound import CompoundRop
from ..base import PerChannelRop
from ..bias import localgradient


class BackgroundRemovalRop(localgradient.LocalGradientBiasRop):

    minfilter_size = 32
    gauss_size = 32
    pregauss_size = 0
    despeckle_size = 0
    chroma_filter_size = None
    luma_minfilter_size = None
    luma_gauss_size = None
    close_factor = 1.0
    despeckle = False
    aggressive = False
    protect_white = False
    gain = 1
    offset = 0


class WhiteTophatFilterRop(PerChannelRop):

    size = 1

    def process_channel(self, data, detected=None):
        return scipy.ndimage.white_tophat(data, self.size)


class ExtractStarsRop(CompoundRop):

    quick = False

    def __init__(self, raw, **kw):
        median_size = kw.pop('median_size', 3)
        median_sigma = kw.pop('median_sigma', 1.0)
        star_size = kw.pop('star_size', 32)

        if self.quick:
            extract_rop = WhiteTophatFilterRop(raw, size=star_size, **kw)
        else:
            extract_rop = BackgroundRemovalRop(raw, minfilter_size=star_size, gauss_size=star_size, **kw)

        super(ExtractStarsRop, self).__init__(
            raw,
            extract_rop,
            median.MaskedMedianFilterRop(raw, size=median_size, sigma=median_sigma, **kw)
        )


class ExtractPureStarsRop(ExtractStarsRop):

    quick = False

    def __init__(self, raw, **kw):
        kw.setdefault('despeckle', True)
        kw.setdefault('despeckle_size', 3)
        kw.setdefault('pregauss_size', 3)
        kw.setdefault('aggressive', True)
        super(ExtractPureStarsRop, self).__init__(raw, **kw)


class RemoveStarsRop(ExtractPureStarsRop):

    def correct(self, data, *p, **kw):
        stars = super(RemoveStarsRop, self).correct(data.copy(), *p, **kw)
        data -= numpy.clip(stars, None, data)
        return data
