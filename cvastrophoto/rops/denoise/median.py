# -*- coding: utf-8 -*-
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
