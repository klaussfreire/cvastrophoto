# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy

from .. import base

logger = logging.getLogger(__name__)

class BaseMeasureRop(base.BaseRop):

    scalar_from_image = staticmethod(numpy.average)

    def measure_image(self, data, *p, **kw):
        return self.correct(data.copy(), *p, **kw)

    def measure_scalar(self, data, *p, **kw):
        data = self.measure_image(data, *p, **kw)
        return self.scalar_from_image(data)


class PerChannelMeasureRop(BaseMeasureRop, base.PerChannelRop):
    pass
