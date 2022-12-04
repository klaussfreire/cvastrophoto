# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage
import skimage.morphology

from ..base import PerChannelRop

from cvastrophoto.accel.skimage.filters import median_filter


class MedianFilterRop(PerChannelRop):

    pre_demargin = True
    size = 1

    @property
    def PROCESSING_MARGIN(self):
        return self.size

    def process_channel(self, data, detected=None, channel=None):
        return median_filter(
            data,
            footprint=skimage.morphology.disk(self.size),
            mode='nearest')


class MaskedMedianFilterRop(MedianFilterRop):

    sigma = 1.0
    low_clip = False

    def get_mask(self, data):
        avg = numpy.average(data)
        dark_part = data[data <= avg]
        noise_avg = numpy.average(dark_part)
        noise_std = numpy.std(dark_part)
        noise_ceil = noise_avg + noise_std * self.sigma
        return data < noise_ceil, noise_ceil, noise_avg, noise_std

    def process_channel(self, data, detected=None, channel=None):
        dark_mask, thr, avg, std = self.get_mask(data)

        dark_part = data.copy()
        dark_part[~dark_mask] = thr

        data[dark_mask] = super(MaskedMedianFilterRop, self).process_channel(
            dark_part, detected)[dark_mask]

        if self.low_clip:
            floor = avg - std * self.sigma
            data[data < floor] = floor

        return data
