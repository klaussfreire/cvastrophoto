# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy.fft

from ..base import PerChannelRop


class DebandingFilterRop(PerChannelRop):

    hipass = 5
    mask_sigma = 1.0

    def process_channel(self, data, detected=None, channel=None):
        avg = numpy.average(data)
        std = numpy.std(data)
        thr = avg + self.mask_sigma * std
        tdata = data.copy()
        tdata[tdata > thr] = avg
        fdata = numpy.fft.fft2(tdata)
        del tdata

        fdata[1:, 1:] = 0
        fdata[:self.hipass, :self.hipass] = 0
        banding = numpy.fft.ifft2(fdata).real
        del fdata

        banding -= numpy.average(banding)
        debanded = data - banding
        del banding

        return numpy.clip(debanded, 0, data.max(), out=data)
