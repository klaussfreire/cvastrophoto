# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage

from ..base import PerChannelRop


class MinfilterRop(PerChannelRop):

    pre_demargin = True
    size = 64
    mode = 'nearest'

    def process_channel(self, data, detected=None, channel=None, **kw):
        return scipy.ndimage.minimum_filter(data, self.size, mode=self.mode)


class MaxfilterRop(PerChannelRop):

    pre_demargin = True
    size = 64
    mode = 'nearest'

    def process_channel(self, data, detected=None, channel=None, **kw):
        return scipy.ndimage.maximum_filter(data, self.size, mode=self.mode)


class OpeningRop(PerChannelRop):

    pre_demargin = True
    size = 64
    mode = 'nearest'

    def process_channel(self, data, detected=None, channel=None, **kw):
        data = scipy.ndimage.minimum_filter(data, self.size, mode=self.mode)
        data = scipy.ndimage.maximum_filter(data, self.size, mode=self.mode)
        return data


class ClosingRop(PerChannelRop):

    pre_demargin = True
    size = 64
    mode = 'nearest'

    def process_channel(self, data, detected=None, channel=None, **kw):
        data = scipy.ndimage.maximum_filter(data, self.size, mode=self.mode)
        data = scipy.ndimage.minimum_filter(data, self.size, mode=self.mode)
        return data
