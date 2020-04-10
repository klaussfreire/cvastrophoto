from __future__ import absolute_import

import functools
import numpy
import scipy.ndimage
import skimage.morphology

from . import base

class StatsMeasureBase(base.PerChannelMeasureRop):

    median_size = 2
    minfilter_size = 16
    gauss_size = 16

    @staticmethod
    def scalar_from_image(x):
        return numpy.average(x.values())

    @staticmethod
    def _measure_rv_method(rvdata, data, y, x, processed):
        rvdata[y, x] = processed

    def measure_image(self, data, *p, **kw):
        rvdata = {}
        kw['process_method'] = self.measure_channel
        kw['rv_method'] = functools.partial(self._measure_rv_method, rvdata)
        base.PerChannelMeasureRop.correct(self, data.copy(), *p, **kw)
        return rvdata

    def measure_channel(self, channel_data, detected=None, channel=None):
        bg_data = scipy.ndimage.median_filter(
            channel_data,
            footprint=skimage.morphology.disk(self.median_size),
            mode='nearest')
        bg_data = scipy.ndimage.minimum_filter(bg_data, self.minfilter_size, mode='nearest')
        bg_data = scipy.ndimage.gaussian_filter(bg_data, self.gauss_size, mode='nearest')
        bg_data = scipy.ndimage.maximum_filter(bg_data, self.minfilter_size, mode='nearest')

        bg_avg = numpy.average(bg_data)
        bg_std = numpy.std(bg_data)

        if channel_data.dtype.kind == 'u':
            channel_data = channel_data.astype(numpy.int32)
        channel_data -= bg_data

        signal_avg = numpy.average(channel_data)
        signal_std = numpy.std(channel_data)

        return (bg_avg, bg_std, signal_avg, signal_std)


class SNRMeasureRop(StatsMeasureBase):

    def measure_image(self, data, *p, **kw):
        kw['process_method'] = self.measure_channel
        return base.PerChannelMeasureRop.correct(self, data.astype(numpy.float32), *p, **kw)

    def measure_channel(self, channel_data, detected=None, channel=None):
        bg_avg, bg_std, signal_avg, signal_std = super(SNRMeasureRop, self).measure_channel(channel_data, detected)
        snr = ((abs(signal_avg) + signal_std) / (abs(bg_avg) + bg_std)) ** 2
        return snr
