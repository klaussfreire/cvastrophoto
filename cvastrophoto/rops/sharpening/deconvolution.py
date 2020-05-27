# -*- coding: utf-8 -*-
from __future__ import absolute_import

import math
import numpy
import random
import scipy.ndimage
import skimage.restoration
import skimage.morphology
import skimage.feature
import PIL.Image

from ..base import PerChannelRop


class BaseDeconvolutionRop(PerChannelRop):

    method = 'wiener'
    normalize_mode = 'max'
    offset = 0.0005

    show_k = False

    METHODS = {
        'rl': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False),
        'rls': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False, iterations=100),
        'rlm': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False, iterations=25),
        'rlw': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False, iterations=10),
        'rlf': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False, iterations=5),
        'wiener': lambda data, k: skimage.restoration.unsupervised_wiener(data, k)[0],
    }

    NORMALIZERS = {
        'max': lambda k: k.max(),
        'sum': lambda k: k.sum(),
    }

    def get_kernel(self, data, detected=None):
        raise NotImplementedError

    def process_channel(self, data, detected=None, channel=None):
        # Compute kernel
        k = self.get_kernel(data, detected=detected)

        if self.normalize_mode in self.NORMALIZERS:
            knorm = self.NORMALIZERS[self.normalize_mode](k)
            if knorm > 0:
                k /= knorm

        if self.show_k:
            PIL.Image.fromarray((numpy.sqrt(k / k.max()) * 255).astype(numpy.uint8)).show()

        # Apply deconvolution
        mxdata = data.max()
        if mxdata > 1:
            data = data.astype(numpy.float32, copy=True)
            data /= mxdata
        else:
            data = data.astype(numpy.float32, copy=self.offset != 0)

        if self.offset != 0:
            data += self.offset
        rv = self.METHODS[self.method](data, k)
        if self.offset != 0:
            rv -= self.offset
        if mxdata > 1:
            rv *= mxdata

        return rv


class DrizzleDeconvolutionRop(BaseDeconvolutionRop):

    scale = 2

    def get_kernel(self, data, detected=None):
        scale = self.scale
        size = scale + (scale - 1) * 2
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[scale-1:2*scale-2, scale-1:2*scale-2] = 1
        return scipy.ndimage.uniform_filter(k, scale)


class GaussianDeconvolutionRop(BaseDeconvolutionRop):

    sigma = 1.0
    size = 3.0

    def get_kernel(self, data, detected=None):
        sigma = self.sigma
        scale = int(max(1, self.sigma) * self.size)
        size = scale + (scale - 1) * 2
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[size/2, size/2] = 1
        return scipy.ndimage.gaussian_filter(k, sigma)


class DoubleGaussianDeconvolutionRop(BaseDeconvolutionRop):

    sigma = 1.0
    sigma2 = 3.0
    w1 = 1.0
    w2 = 0.15
    offx = 0
    offy = 0
    size = 3.0

    def get_kernel(self, data, detected=None):
        scale = int(max(1, self.sigma, self.sigma2) * self.size)
        size = 1 + (scale - 1) * 2
        offx = self.offx
        offy = self.offy
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[size/2, size/2] = 1
        k1 = scipy.ndimage.gaussian_filter(k, self.sigma)
        k2 = scipy.ndimage.gaussian_filter(k, self.sigma2)
        k = k1 * self.w1
        k[offy:,offx:] += k2[:k.shape[0]-offy,:k.shape[1]-offx] * self.w2
        return k


class AiryDeconvolutionRop(BaseDeconvolutionRop):

    sigma = 0.5
    r1 = 1.25
    r0 = 0.5
    m0 = 0.2
    scale = 1.0
    size = 5.0

    def get_kernel(self, data, detected=None):
        # A gaussian is a good approximation of the airy power envelope,
        # but it's missing the rings. We get the rings by multiplying by
        # |cos(d)|. We apply the gaussian again to hide aliasing.
        sigma = self.sigma * self.scale
        r1 = self.r1 * self.scale
        r0 = self.r0 * self.scale
        scale = int(max(1, sigma, r1 + r0) * self.size)
        size = 1 + (scale - 1) * 2
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[size/2, size/2] = 1
        k = scipy.ndimage.gaussian_filter(k, sigma)
        x = numpy.arange(size, dtype=k.dtype)
        x, y = numpy.meshgrid(x, x)
        x -= size/2
        y -= size/2
        d = numpy.sqrt(x*x + y*y)
        k *= self.m0 + numpy.abs(numpy.cos(numpy.clip((d - r0) * (math.pi / 2 / r1), 0, None)))
        return k


class SampledDeconvolutionRop(BaseDeconvolutionRop):

    anisotropic = False
    sigma = 0.5
    gamma = 1.0
    doff = 0.0
    size = 64
    weight = 1.0
    envelope = 1.0
    max_samples = 1024
    threshold_rel = 0.01
    sample_threshold_rel = 10
    sx = 0
    sy = 0

    def get_kernel(self, data, detected=None):
        # A gaussian is a good approximation of the airy power envelope,
        # but it's missing the rings. We get the rings by multiplying by
        # |cos(d)|. We apply the gaussian again to hide aliasing.
        luma = self.raw.luma_image(data, renormalize=False, same_shape=False, dtype=numpy.float32)

        ksize = 1 + int(self.size + self.doff) * 2
        size = int(self.size + self.doff)

        pool = self.raw.default_pool
        if pool is not None:
            map_ = pool.map
        else:
            map_ = map

        peaks = []
        footprints = []
        dirs = []
        krange = numpy.arange(size)

        if not self.sx and not self.sy:
            for dy, dx in (
                    (0, 1), (0, -1), (1, 0), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)):
                footprint = numpy.zeros((ksize, ksize), numpy.bool8)
                footprint[ksize/2 + dy*krange, ksize/2 + dx*krange] = True
                footprints.append(footprint)
                dirs.append([dy, dx])

            def find_peaks(footprint):
                return skimage.feature.peak_local_max(
                    luma, footprint=footprint, threshold_rel=self.threshold_rel, num_peaks=self.max_samples*8)

            for speaks in map_(find_peaks, footprints):
                peaks.append(list(speaks))
        else:
            peaks.append([[self.sy, self.sx]] * 4)

        for speaks in peaks:
            random.shuffle(speaks)

        if not self.anisotropic:
            npeaks = []
            for speaks in peaks:
                npeaks.extend(speaks)
            peaks = [npeaks] * len(dirs)

        lkerns = []

        for (dy, dx), speaks in zip(dirs, peaks):
            lkern = numpy.zeros(size * 2, numpy.float32)
            lkern[:size] = 1
            nsamples = 0

            for y, x in speaks:
                lsample = luma[y + dy*krange, x + dx*krange]

                if lsample.min() <= 1:
                    continue
                if len(lsample) != size or lsample[1:].ptp() <= lsample.min() * self.sample_threshold_rel:
                    continue
                if lsample.max() != lsample[0]:
                    continue
                if lsample[1:].max() >= lsample[0]:
                    continue

                lsample -= lsample.min()
                if not lsample[0]:
                    continue

                lsample /= lsample[0]
                newmin = numpy.minimum(lkern[:size], lsample)
                lkern[:size][newmin > 0] = newmin[newmin > 0]
                nsamples += 1

                if nsamples > self.max_samples:
                    break

            lkerns.append(lkern)

        del peaks, speaks

        ks = []

        x = numpy.arange(ksize, dtype=numpy.float32)
        x, y = numpy.meshgrid(x, x)
        x -= ksize/2
        y -= ksize/2

        d = numpy.sqrt(x*x + y*y)
        kdirx = x / numpy.clip(d, 0.5, None)
        kdiry = y / numpy.clip(d, 0.5, None)

        for (dy, dx), lkern in zip(dirs, lkerns):
            doff = d / math.sqrt(dy*dy + dx*dx) + self.doff
            di = doff.astype(numpy.uint16)
            df = doff - di

            thresh = numpy.percentile(lkern[lkern != 0], 10)
            tail = lkern < thresh
            lkern[~tail] -= thresh
            lkern[tail] = 0

            lkern = numpy.power(lkern, self.gamma)

            k = lkern[di] * (1.0 - df) + lkern[di + 1] * df

            ks.append(k)

        if self.anisotropic:
            def dweight(i):
                rv = (
                    numpy.clip(kdirx * dirs[i][1], 0, None)
                    + numpy.clip(kdiry * dirs[i][0], 0, None)
                )
                rv[ksize/2, ksize/2] = 1
                return rv
            k = dweight(0) * ks[0]
            for i in xrange(1, len(ks)):
                k += dweight(i) * ks[i]
        else:
            k = ks[0]
            for i in xrange(1, len(ks)):
                k = numpy.minimum(k, ks[i])

        if k.sum():
            k /= k.sum()
            k *= self.weight

        if self.sigma:
            k0 = numpy.zeros(k.shape, k.dtype)
            k0[ksize/2, ksize/2] = 1
            k0 = scipy.ndimage.gaussian_filter(k0, self.sigma)
            k += k0

        return k
