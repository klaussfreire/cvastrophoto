# -*- coding: utf-8 -*-
from __future__ import absolute_import

import scipy.ndimage

from ..base import PerChannelRop


class ShiftRop(PerChannelRop):

    x = 0.0
    y = 0.0
    mode = 'mirror'

    def process_channel(self, channel_data, detected, channel, **kw):
        return scipy.ndimage.shift(channel_data, (self.y, self.x), mode=self.mode)
