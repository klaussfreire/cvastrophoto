# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy

from ..base import PerChannelRop


class SCNRRop(PerChannelRop):

    target_channel = 1
    invert = False

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
                            if self.invert:
                                m = numpy.minimum(m, cdata)
                            else:
                                m = numpy.maximum(m, cdata)
                        else:
                            m = cdata
                detected['m'] = m

            y, x = channel
            if raw_pattern[y, x] == self.target_channel:
                if self.invert:
                    data = numpy.maximum(data, m, data)
                else:
                    data = numpy.minimum(data, m, data)

        return data
