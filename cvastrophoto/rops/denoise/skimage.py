# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy
import math
import scipy.ndimage
import skimage.restoration
import skimage.transform

from cvastrophoto.util import decomposition, vectorize, gaussian
from cvastrophoto.image import rgb, fits, metaimage
from cvastrophoto.accel.skimage import filters

from ..base import PerChannelRop
from ..tracking.extraction import ExtractPureStarsRop, ExtractPureBackgroundRop


logger = logging.getLogger(__name__)


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


if vectorize.with_cuda:
    from numba import cuda
    import numba
    from cvastrophoto.util.vectorize import cuda_sum, cuda_sum2, cuda_mul

    orig_denoise_tv_chambolle_nd = skimage.restoration._denoise._denoise_tv_chambolle_nd

    TV_TPB = 16

    @cuda.jit(fastmath=True)
    def _cuda_denoise_tv_chambolle_divergence(d, p, img, out):
        x, y = cuda.grid(2)

        if x < out.shape[0] and y < out.shape[1]:
            D = -(p[x, y, 0] + p[x, y, 1])

            if x >= 1:
                D += p[x-1,y,0]
            if y >= 1:
                D += p[x,y-1,1]
            d[x, y] = D
            out[x, y] = img[x,y] + D

    @cuda.jit(fastmath=True)
    def _cuda_denoise_tv_chambolle_gradupdate(p, g, out, norm, tau, tau_over_weight):
        x, y = cuda.grid(2)

        if x < out.shape[0] and y < out.shape[1]:
            if x < (out.shape[0]-1):
                g[x,y,0] = g0 = out[x+1,y] - out[x,y]
            else:
                g0 = 0
            if y < (out.shape[1]-1):
                g[x,y,1] = g1 = out[x,y+1] - out[x,y]
            else:
                g1 = 0

            norm[x,y] = n = math.sqrt(g0 * g0 + g1 * g1)

            n = n * tau_over_weight + 1.
            p[x,y,0] = (p[x,y,0] - tau * g[x,y,0]) / n
            p[x,y,1] = (p[x,y,1] - tau * g[x,y,1]) / n

    @cuda.jit(fastmath=True)
    def _cuda_denoise_tv_chambolle_gradupdate_weightarray(p, g, out, norm, tau, tau_over_weight):
        x, y = cuda.grid(2)

        if x < out.shape[0] and y < out.shape[1]:
            if x < (out.shape[0]-1):
                g[x,y,0] = g0 = out[x+1,y] - out[x,y]
            else:
                g0 = 0
            if y < (out.shape[1]-1):
                g[x,y,1] = g1 = out[x,y+1] - out[x,y]
            else:
                g1 = 0

            norm[x,y] = n = math.sqrt(g0 * g0 + g1 * g1)

            n = n * tau_over_weight[x,y] + 1.
            p[x,y,0] = (p[x,y,0] - tau * g[x,y,0]) / n
            p[x,y,1] = (p[x,y,1] - tau * g[x,y,1]) / n

    def _denoise_tv_chambolle_nd(image, weight=0.1, eps=2.e-4, n_iter_max=200):
        ndim = image.ndim
        if ndim != 2:
            return orig_denoise_tv_chambolle_nd(image, weight=weight, eps=eps, n_iter_max=n_iter_max)

        try:
            return _cuda_denoise_tv_chambolle_nd(image, weight, eps, n_iter_max)
        except (vectorize.CUDA_ERRORS + (MemoryError,)) as e:
            logger.warning("Error doing CUDA TV denoise, falling back to CPU: %s", e)
            return orig_denoise_tv_chambolle_nd(image, weight=weight, eps=eps, n_iter_max=n_iter_max)

    def _cuda_denoise_tv_chambolle_nd(image, weight=0.1, eps=2.e-4, n_iter_max=200):
        return vectorize.in_cuda_pool(
            image.size * 8 * image.dtype.itemsize,
            _cuda_denoise_tv_chambolle_nd_impl,
            image, weight, eps, n_iter_max).get()

    def _cuda_denoise_tv_chambolle_nd_impl(image, weight=0.1, eps=2.e-4, n_iter_max=200):
        p = cuda.device_array(image.shape + (2,), image.dtype)
        g = cuda.device_array_like(p)
        norm = cuda.device_array_like(image)
        d = cuda.device_array_like(image)
        d2tmp = cuda_sum2.mktemp(d)
        img = cuda.to_device(image)
        out = cuda.device_array_like(img)

        normflat = norm.reshape(norm.size)
        tau = 1. / (2.*2)

        if isinstance(weight, numpy.ndarray):
            weight_array = True
            weightflat = cuda.to_device(weight).reshape(weight.size)
            wnormflat = cuda.device_array_like(normflat)
            tau_over_weight = cuda.to_device(tau / weight)
        else:
            weight_array = False
            weightflat = weight
            tau_over_weight = tau / weight

        vectorize.cuda_fill(d, 0)
        vectorize.cuda_fill(p, 0)
        vectorize.cuda_fill(g, 0)

        blockconf = vectorize.cuda_block_config(image.shape, (TV_TPB, TV_TPB))
        divergence = _cuda_denoise_tv_chambolle_divergence[blockconf]

        if weight_array:
            gradupdate = _cuda_denoise_tv_chambolle_gradupdate_weightarray[blockconf]
        else:
            gradupdate = _cuda_denoise_tv_chambolle_gradupdate[blockconf]

        i = 0
        while i < n_iter_max:
            if i > 0:
                # d will be the (negative) divergence of p
                divergence(d, p, img, out)
            else:
                out.copy_to_device(img)

            # g stores the gradients of out along each axis
            # e.g. g[0] is the first order finite difference along axis 0
            gradupdate(p, g, out, norm, tau, tau_over_weight)

            E = cuda_sum2(d, partials=d2tmp) if i > 0 else 0
            E += cuda_sum(cuda_mul(weightflat, normflat, wnormflat)) if weight_array else weight * cuda_sum(normflat)
            E /= float(image.size)
            if i == 0:
                E_init = E
                E_previous = E
            else:
                if numpy.abs(E_previous - E) < eps * E_init:
                    break
                else:
                    E_previous = E
            i += 1
        return out.copy_to_host()

    skimage.restoration._denoise._denoise_tv_chambolle_nd = _denoise_tv_chambolle_nd


class TVDenoiseRop(PerChannelRop):

    weight = 0.25
    normalize = True
    eps = 0.000005
    steps = 200
    iters = 1
    levels = 3
    level_scale = 2
    level_limit = 0
    scale_levels = False
    scale_base = 0.25
    gamma = 1.0
    offset = 0.003
    aoffset = 0
    use_meta = True
    meta_std_smoothing = 2
    meta_std_open_size = 0

    # CUDA does parallelization on its own, it serves no purpose to further parallelize it
    parallel_channels = not vectorize.with_cuda

    def estimate_noise(self, data, mim=None, weight=1.0, channel=None):
        if mim is not None and self.use_meta:
            std_data = mim.std_data
            if std_data is not None:
                if channel is not None and std_data.shape != data.shape:
                    raw_pattern = self._raw_pattern
                    path, patw = raw_pattern.shape
                    if len(std_data.shape) == 3:
                        std_data = std_data[:, :, raw_pattern[channel]]
                    else:
                        y, x = channel
                        std_data = std_data[y::path, x::patw]
                mim_max = mim.mainimage.max()
                if self.meta_std_open_size > 0:
                    std_data = scipy.ndimage.minimum_filter(std_data, self.meta_std_open_size, mode='nearest')
                if self.meta_std_smoothing > 0:
                    std_data = gaussian.fast_gaussian(std_data, self.meta_std_smoothing)
                weight = (weight / mim_max) * std_data
                weight *= 1.0 / math.sqrt(mim.mainaccum.num_images)
                return weight

        bgdata = ExtractPureBackgroundRop(rgb.Templates.LUMINANCE, copy=False).correct(data.copy())
        bgdata = scipy.ndimage.sobel(bgdata)
        weight = 0.35 * (weight / data.max()) * numpy.std(bgdata)
        return weight

    def process_channel(self, data, detected=None, channel=None, img=None, **kw):
        mxdata = data.max()
        scale = (1.0 / mxdata)
        rvdt = numpy.float32 if data.dtype.kind in 'ui' else data.dtype
        rv = (data * scale).astype(rvdt, copy=False)
        aoffset = self.aoffset * scale

        mim = metaimage.MetaImage.get_metaimage(img)
        mim_close = mim is img

        weight = self.weight
        if self.normalize:
            weight = self.estimate_noise(data, mim=mim, weight=weight, channel=channel)

        if mim_close:
            mim.close()

        if self.gamma != 0:
            rv += self.offset + aoffset
            if self.gamma != 1.0:
                rv = numpy.power(rv, self.gamma, out=rv, where=rv > 0)
                weight = pow(weight, self.gamma)

        levels = decomposition.gaussian_decompose(rv, self.levels, scale=self.level_scale)
        del rv

        for nlevel, level in enumerate(levels):
            if nlevel >= len(levels) - self.level_limit:
                continue
            for i in range(self.iters):
                level[:] = skimage.restoration.denoise_tv_chambolle(
                    level,
                    weight=weight * (self.scale_base ** nlevel if self.scale_levels else 1),
                    eps=self.eps, n_iter_max=self.steps)

        rv = decomposition.gaussian_recompose(levels)
        del levels

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

    def process_channel(self, data, detected=None, channel=None, **kw):
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

    def process_channel(self, data, detected=None, channel=None, **kw):
        data, mxdata, sigma = self.normalize_get_sigma(data, dtype=numpy.double)

        rv = skimage.restoration.denoise_bilateral(
            data,
            sigma_color=sigma, mode=self.mode, sigma_spatial=self.sigma_spatial,
            win_size=self.win_size or None,
            multichannel=False)
        rv *= mxdata
        return rv


class StarlessDenoiseRopBase(PerChannelRop):

    # background, stars or both
    denoise_layer = 'background'

    def __init__(self, raw, **kw):
        self._extract_stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(ExtractPureStarsRop, k)}
        self._extract_stars_kw.setdefault('copy', False)
        super(StarlessDenoiseRopBase, self).__init__(raw, **kw)

    def correct(self, data, *p, **kw):
        dmax = data.max()
        stars_rop = ExtractPureStarsRop(self.raw, **self._extract_stars_kw)
        stars = stars_rop.correct(data.copy())

        data -= numpy.clip(stars, None, data, out=stars)

        if self.denoise_layer in ('background', 'both'):
            data = super(StarlessDenoiseRopBase, self).correct(data, *p, dmax=dmax, **kw)
        if self.denoise_layer in ('stars', 'both'):
            stars = super(StarlessDenoiseRopBase, self).correct(stars, *p, dmax=dmax, **kw)
        data += numpy.clip(stars, None, dmax - data)
        return data


class StarlessBilateralDenoiseRop(BilateralDenoiseRop, StarlessDenoiseRopBase):
    pass


class StarlessTVDenoiseRop(TVDenoiseRop, StarlessDenoiseRopBase):
    pass


class StarlessWaveletDenoiseRop(WaveletDenoiseRop, StarlessDenoiseRopBase):
    pass
