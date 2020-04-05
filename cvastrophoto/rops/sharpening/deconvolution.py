# -*- coding: utf-8 -*-
from __future__ import absolute_import

import math
import numpy
import scipy.ndimage
import skimage.restoration
import skimage.morphology
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

    def process_channel(self, data, detected=None):
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
    size = 3.0

    def get_kernel(self, data, detected=None):
        scale = int(max(1, self.sigma, self.sigma2) * self.size)
        size = 1 + (scale - 1) * 2
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[size/2, size/2] = 1
        k1 = scipy.ndimage.gaussian_filter(k, self.sigma)
        k2 = scipy.ndimage.gaussian_filter(k, self.sigma2)
        k = k1 * self.w1 + k2 * self.w2
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
