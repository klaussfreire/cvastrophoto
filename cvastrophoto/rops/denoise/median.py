# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage
import skimage.morphology

from ..base import PerChannelRop


class MedianFilterRop(PerChannelRop):

    size = 1

    def process_channel(self, data, detected=None):
        return scipy.ndimage.median_filter(
            data,
            footprint=skimage.morphology.disk(self.size),
            mode='nearest')


class MaskedMedianFilterRop(MedianFilterRop):

    sigma = 1.0

    def get_mask(self, data):
        avg = numpy.average(data)
        dark_part = data[data <= avg]
        noise_avg = numpy.average(dark_part)
        noise_std = numpy.std(dark_part)
        noise_ceil = noise_avg + noise_std * self.sigma
        return data < noise_ceil, noise_ceil

    def process_channel(self, data, detected=None):
        dark_mask, thr = self.get_mask(data)

        dark_part = data.copy()
        dark_part[~dark_mask] = thr

        data[dark_mask] = super(MaskedMedianFilterRop, self).process_channel(
            dark_part, detected)[dark_mask]

        return data
