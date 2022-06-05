# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage

from ..base import PerChannelRop
from cvastrophoto.util import gaussian


class GaussianFilterRop(PerChannelRop):

    pre_demargin = True
    sigma = 8
    mode = 'nearest'
    quick = False

    @property
    def PROCESSING_MARGIN(self):
        return self.sigma * 4

    def process_channel(self, data, detected=None, channel=None):
        if self.quick:
            kw = {'truncate': 2.5}
        else:
            kw = {}
        return gaussian.fast_gaussian(data, self.sigma, mode=self.mode, **kw)
