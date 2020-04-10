# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy
import scipy.ndimage

from . import base

logger = logging.getLogger(__name__)

class FocusMeasureRop(base.PerChannelMeasureRop):

    size = 0.05
    quick = False

    def measure_channel(self, channel_data, detected=None, channel=None):
        isize = int(max(channel_data.shape) * self.size)
        edge = scipy.ndimage.sobel(channel_data.astype(numpy.float32, copy=False))
        edge *= 1.0 / 65535
        edge = (
            scipy.ndimage.maximum_filter(edge, isize)
            - scipy.ndimage.minimum_filter(edge, isize)
        )
        if not self.quick:
            edge = scipy.ndimage.gaussian_filter(edge, isize)
        return edge
