# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage

from ..base import PerChannelRop


class MinfilterRop(PerChannelRop):

    size = 64
    mode = 'nearest'

    def process_channel(self, data, detected=None, channel=None):
        return scipy.ndimage.minimum_filter(data, self.size, mode=self.mode)


class MaxfilterRop(PerChannelRop):

    size = 64
    mode = 'nearest'

    def process_channel(self, data, detected=None, channel=None):
        return scipy.ndimage.maximum_filter(data, self.size, mode=self.mode)


class OpeningRop(PerChannelRop):

    size = 64
    mode = 'nearest'

    def process_channel(self, data, detected=None, channel=None):
        data = scipy.ndimage.minimum_filter(data, self.size, mode=self.mode)
        data = scipy.ndimage.maximum_filter(data, self.size, mode=self.mode)
        return data


class ClosingRop(PerChannelRop):

    size = 64
    mode = 'nearest'

    def process_channel(self, data, detected=None, channel=None):
        data = scipy.ndimage.maximum_filter(data, self.size, mode=self.mode)
        data = scipy.ndimage.minimum_filter(data, self.size, mode=self.mode)
        return data
