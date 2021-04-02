# -*- coding: utf-8 -*-
from __future__ import absolute_import

from past.builtins import xrange
import numpy

from ..base import PerChannelRop
from cvastrophoto.util import demosaic, srgb


class SCNRRop(PerChannelRop):

    target_channel = 1

    def detect_channel(self, channel_data, **kw):
        return channel_data

    def process_channel(self, data, detected=None, channel=None, **kw):
        raw_pattern = self._raw_pattern

        if detected is not None:
            m = detected.get('m')
            if m is None:
                for (y, x), cdata in list(detected.items()):
                    if raw_pattern[y,x] != self.target_channel:
                        if m is not None:
                            m = numpy.maximum(m, cdata)
                        else:
                            m = cdata
                detected['m'] = m

            y, x = channel
            if raw_pattern[y, x] == self.target_channel:
                data = numpy.minimum(data, m, data)

        return data
