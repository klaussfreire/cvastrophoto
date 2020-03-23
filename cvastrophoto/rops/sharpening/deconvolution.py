# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage
import skimage.restoration
import skimage.morphology

from ..base import PerChannelRop


class BaseDeconvolutionRop(PerChannelRop):

    method = 'wiener'

    METHODS = {
        'rl': skimage.restoration.richardson_lucy,
        'wiener': lambda data, k: skimage.restoration.unsupervised_wiener(data, k)[0],
    }

    def get_kernel(self, data, detected=None):
        raise NotImplementedError

    def process_channel(self, data, detected=None):
        # Compute kernel
        k = self.get_kernel(data, detected=detected)

        # Apply deconvolution
        mxdata = data.max()
        if mxdata > 1:
            data = data.astype(numpy.float32, copy=True)
            data /= mxdata
        else:
            data = data.astype(numpy.float32, copy=False)

        rv = self.METHODS[self.method](data, k)
        if mxdata > 1:
            rv *= mxdata

        return rv


class DrizzleDeconvolutionRop(BaseDeconvolutionRop):

    scale = 2

    def get_kernel(self, data, detected=None):
        scale = self.scale
        size = scale + (scale - 1) * 2
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[scale-1:2*scale-1, scale-1:2*scale-1] = 1
        return scipy.ndimage.uniform_filter(k, scale)


class GaussianDeconvolutionRop(BaseDeconvolutionRop):

    sigma = 2.0
    size = 3.0

    def get_kernel(self, data, detected=None):
        sigma = self.sigma
        scale = int(self.sigma * self.size)
        size = scale + (scale - 1) * 2
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[scale-1:2*scale-1, scale-1:2*scale-1] = 1
        return scipy.ndimage.gaussian_filter(k, sigma)
