# -*- coding: utf-8 -*-
from __future__ import absolute_import

from past.builtins import xrange
import numpy

from ..base import PerChannelRop
from cvastrophoto.util import demosaic, srgb


class ClipMaxRop(PerChannelRop):

    _clip_min = None

    def detect_channel(self, channel_data, **kw):
        return channel_data.max()

    def process_channel(self, data, detected=None, channel=None, **kw):
        if detected is not None:
            detected = min(detected.values())

        if detected is not None:
            data = numpy.clip(data, self._clip_min, detected, out=data)

        return data


class ClipBothRop(ClipMaxRop):

    _clip_min = 0
