from __future__ import absolute_import

import functools

from ..measures.base import BaseMeasureRop
from ..base import PerChannelRop


class BaseNormalizationRop(BaseMeasureRop):

    def detect(self, data, **kw):
        return self.measure_image(data, **kw)


class PerChannelNormalizationRop(BaseNormalizationRop, PerChannelRop):

    _measure_rv_method = None

    def measure_image(self, data, *p, **kw):
        kw['process_method'] = self.measure_channel
        kw['rv_method'] = self._measure_rv_method
        return PerChannelRop.correct(data.copy(), *p, **kw)

    def process_channel(self, channel_data, detected=None, channel=None, **kw):
        raise NotImplementedError

    def measure_channel(self, channel_data, detected=None, channel=None):
        raise NotImplementedError
