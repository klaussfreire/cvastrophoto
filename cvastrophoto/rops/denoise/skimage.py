# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage
import skimage.restoration
import skimage.transform

from cvastrophoto.util import decomposition
from cvastrophoto.image import rgb

from ..base import PerChannelRop
from ..tracking.extraction import ExtractPureStarsRop, ExtractPureBackgroundRop


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

    weight = 0.01
    normalize = False
    eps = 0.0002
    steps = 200
    iters = 1
    levels = 1
    level_scale = 1
    scale_levels = False
    gamma = 1.8
    offset = 0.003
    aoffset = 0

    def estimate_noise(self, data):
        bgdata = ExtractPureBackgroundRop(rgb.Templates.LUMINANCE, copy=False).correct(data.copy())
        return numpy.var(bgdata) / numpy.var(data)

    def process_channel(self, data, detected=None, channel=None):
        mxdata = data.max()
        scale = (1.0 / mxdata)
        rv = data * scale
        aoffset = self.aoffset * scale

        weight = self.weight
        if self.normalize:
            weight *= self.estimate_noise(data)

        if self.gamma != 0:
            rv += self.offset + aoffset
            rv = numpy.power(rv, self.gamma, out=rv, where=rv > 0)
            weight = pow(weight, self.gamma)

        levels = decomposition.gaussian_decompose(rv, self.levels, scale=self.level_scale)
        del rv

        for nlevel, level in enumerate(levels):
            for i in range(self.iters):
                level[:] = skimage.restoration.denoise_tv_chambolle(
                    level,
                    weight=weight * ((1 << nlevel) if self.scale_levels else 1),
                    eps=self.eps, n_iter_max=self.steps)
        rv = decomposition.gaussian_recompose(levels)

        if self.gamma != 0:
            rv = numpy.power(rv, 1.0 / self.gamma, out=rv, where=rv > 0)
            rv -= self.offset + aoffset

        rv *= mxdata
        return rv


class WaveletDenoiseRop(SigmaDenoiseMixin, PerChannelRop):

    sigma = 2.0
    prefilter_size = 2.0
    mode = 'soft'
    method = 'VisuShrink'
    wavelet = 'coif1'
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
    sigma_spatial = 2.0
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


class StarlessDenoiseRopBase(PerChannelRop):

    def __init__(self, raw, **kw):
        self._extract_stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(ExtractPureStarsRop, k)}
        self._extract_stars_kw.setdefault('copy', False)
        super(StarlessDenoiseRopBase, self).__init__(raw, **kw)

    def correct(self, data, *p, **kw):
        dmax = data.max()
        stars_rop = ExtractPureStarsRop(self.raw, **self._extract_stars_kw)
        stars = stars_rop.correct(data.copy())

        data -= numpy.clip(stars, None, data, out=stars)
        data = super(StarlessDenoiseRopBase, self).correct(data, *p, dmax=dmax, **kw)
        data += numpy.clip(stars, None, dmax - data)
        return data


class StarlessBilateralDenoiseRop(BilateralDenoiseRop, StarlessDenoiseRopBase):
    pass


class StarlessTVDenoiseRop(TVDenoiseRop, StarlessDenoiseRopBase):
    pass


class StarlessWaveletDenoiseRop(WaveletDenoiseRop, StarlessDenoiseRopBase):
    pass
