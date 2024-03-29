from __future__ import absolute_import

import functools
import numpy
import scipy.ndimage
import skimage.morphology

from . import base

from cvastrophoto.util import gaussian
from cvastrophoto.accel.skimage.filters import median_filter

class StatsMeasureBase(base.PerChannelMeasureRop):

    median_size = 2
    maxfilter_size = 0
    minfilter_size = 16
    gauss_size = 16

    @staticmethod
    def scalar_from_image(x):
        return numpy.average(list(x.values()))

    @staticmethod
    def _measure_rv_method(rvdata, data, y, x, processed):
        rvdata[y, x] = processed

    def detect(self, data, **kw):
        return None

    def measure_image(self, data, *p, **kw):
        rvdata = {}
        kw['process_method'] = self.measure_channel
        kw['rv_method'] = functools.partial(self._measure_rv_method, rvdata)
        base.PerChannelMeasureRop.correct(self, data.copy(), *p, **kw)
        return rvdata

    def measure_channel(self, channel_data, detected=None, channel=None):
        bg_data = median_filter(
            channel_data,
            footprint=skimage.morphology.disk(self.median_size),
            mode='nearest')

        if self.maxfilter_size:
            bg_data = scipy.ndimage.maximum_filter(bg_data, self.maxfilter_size, mode='nearest', output=bg_data)
        bg_data = scipy.ndimage.minimum_filter(
            bg_data, self.maxfilter_size + self.minfilter_size, mode='nearest', output=bg_data)
        bg_data = gaussian.fast_gaussian(bg_data, self.gauss_size, mode='nearest')
        bg_data = scipy.ndimage.maximum_filter(bg_data, self.minfilter_size, mode='nearest', output=bg_data)

        bg_avg = numpy.average(bg_data)
        bg_std = numpy.std(bg_data)

        if channel_data.dtype.kind == 'u':
            channel_data = channel_data.astype(numpy.int32)
        channel_data -= bg_data

        signal_avg = numpy.average(channel_data)
        signal_std = numpy.std(channel_data)

        return (bg_avg, bg_std, signal_avg, signal_std)


class SNRMeasureRop(StatsMeasureBase):

    maxfilter_size = 8

    def measure_image(self, data, *p, **kw):
        kw['process_method'] = self.measure_channel
        return base.PerChannelMeasureRop.correct(self, data.astype(numpy.float32), *p, **kw)

    @staticmethod
    def scalar_from_image(x):
        return numpy.average(x)

    def measure_channel(self, channel_data, detected=None, channel=None):
        bg_avg, bg_std, signal_avg, signal_std = super(SNRMeasureRop, self).measure_channel(channel_data, detected)
        snr = ((abs(signal_avg) + signal_std) / (abs(bg_avg) + bg_std)) ** 2
        return snr


class BgAvgMeasureRop(SNRMeasureRop):

    def measure_channel(self, channel_data, detected=None, channel=None):
        bg_avg, bg_std, signal_avg, signal_std = super(SNRMeasureRop, self).measure_channel(channel_data, detected)
        return bg_avg


class SimpleMeasureBase(StatsMeasureBase):

    _stat = None

    def measure_channel(self, channel_data, detected=None, channel=None):
        return self._stat(channel_data)


class AvgMeasureRop(SimpleMeasureBase):

    _stat = staticmethod(numpy.average)


class StdDevMeasureRop(SimpleMeasureBase):

    _stat = staticmethod(numpy.std)


class MedianMeasureRop(SimpleMeasureBase):

    _stat = staticmethod(numpy.median)


class MaxMeasureRop(SimpleMeasureBase):

    _stat = staticmethod(numpy.max)


class MinMeasureRop(SimpleMeasureBase):

    _stat = staticmethod(numpy.min)
