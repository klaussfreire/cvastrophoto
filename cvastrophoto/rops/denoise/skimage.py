# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage
import skimage.restoration

from ..base import PerChannelRop


class SigmaDenoiseMixin(object):

    sigma = 0.0

    def estimate_noise(self, data):
        hfc = scipy.ndimage.white_tophat(data, int(self.prefilter_size))
        mask = hfc <= ((numpy.average(hfc) + numpy.max(hfc)) * 0.5)
        return numpy.std(hfc[mask])

    def normalize_get_sigma(self, data, dtype=numpy.float32):
        mxdata = data.max()
        data = data.astype(dtype)
        data *= (1.0 / mxdata)

        sigma = self.sigma
        if not sigma:
            sigma = None
        else:
            # Compute noise standard deviation and scale by sigma
            sigma *= self.estimate_noise(data)

        return data, mxdata, sigma


class TVDenoiseRop(PerChannelRop):

    weight = 0.001
    eps = 0.0002
    steps = 200

    def process_channel(self, data, detected=None, channel=None):
        mxdata = data.max()
        rv = skimage.restoration.denoise_tv_chambolle(
            data * (1.0 / mxdata),
            weight=self.weight, eps=self.eps, n_iter_max=self.steps)
        rv *= mxdata
        return rv


class WaveletDenoiseRop(SigmaDenoiseMixin, PerChannelRop):

    sigma = 2.0
    prefilter_size = 2.0
    mode = 'soft'
    wavelet = 'bior2.2'
    levels = 0

    def process_channel(self, data, detected=None, channel=None):
        data, mxdata, sigma = self.normalize_get_sigma(data)

        rv = skimage.restoration.denoise_wavelet(
            data,
            sigma=sigma, mode=self.mode, wavelet_levels=self.levels or None,
            wavelet=self.wavelet)
        rv *= mxdata
        return rv


class BilateralDenoiseRop(SigmaDenoiseMixin, PerChannelRop):

    win_size = 0
    bins = 10000
    sigma_spatial = 2
    mode = 'edge'

    def process_channel(self, data, detected=None, channel=None):
        data, mxdata, sigma = self.normalize_get_sigma(data, dtype=numpy.double)

        rv = skimage.restoration.denoise_bilateral(
            data,
            sigma_color=sigma, mode=self.mode, sigma_spatial=self.sigma_spatial,
            win_size=self.win_size or None,
            multichannel=False)
        rv *= mxdata
        return rv