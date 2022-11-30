# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy

from .. import base

logger = logging.getLogger(__name__)

class BaseMeasureRop(base.BaseRop):

    scalar_from_image = staticmethod(numpy.average)

    def measure_image(self, data, *p, **kw):
        raise NotImplementedError

    def measure_scalar(self, data, *p, **kw):
        data = self.measure_image(data, *p, **kw)
        return self.scalar_from_image(data)


class PerChannelMeasureRop(BaseMeasureRop, base.PerChannelRop):

    _measure_rv_method = None

    scalar = False
    measure_dtype = None

    def measure_image(self, data, *p, **kw):
        kw['process_method'] = self.measure_channel
        kw['rv_method'] = self._measure_rv_method

        if self.measure_dtype is not None:
            data = data.astype(self.measure_dtype)
        else:
            data = data.copy()

        if self.scalar:
            data[:] = self.measure_scalar(data, *p, **kw)
        else:
            data = base.PerChannelRop.correct(self, data, *p, **kw)

        return data

    def measure_channel(self, channel_data, detected=None, channel=None):
        raise NotImplementedError
