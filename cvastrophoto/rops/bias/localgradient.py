# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy
import scipy.ndimage
import skimage.filters
import skimage.morphology
from ..base import BaseRop

logger = logging.getLogger(__name__)

class LocalGradientBiasRop(BaseRop):

    minfilter_size = 256
    gauss_size = 256
    pregauss_size = 2
    despeckle_size = 2
    gain = 1
    offset = -1
    despeckle = True
    aggressive = False

    def detect(self, data, quick=False, **kw):
        path, patw = self._raw_pattern.shape
        if data.dtype.kind not in ('i', 'u'):
            dt = data.dtype
        else:
            dt = numpy.int32
        local_gradient = numpy.empty(data.shape, dt)
        data = self.raw.demargin(data.copy())
        def compute_local_gradient(task):
            try:
                data, local_gradient, y, x = task
                despeckled = data[y::path, x::patw]

                # Remove small-scale artifacts that could bias actual sky level
                if self.despeckle:
                    if quick or self.aggressive:
                        despeckled = scipy.ndimage.maximum_filter(despeckled, self.despeckle_size)
                    else:
                        despeckled = scipy.ndimage.median_filter(
                            despeckled,
                            footprint=skimage.morphology.disk(self.despeckle_size),
                            mode='nearest')
                if self.pregauss_size:
                    despeckled = scipy.ndimage.gaussian_filter(despeckled, self.pregauss_size)

                # Compute sky baseline levels
                grad = scipy.ndimage.minimum_filter(despeckled, self.minfilter_size, mode='nearest')
                del despeckled

                # Regularization (smoothen)
                grad = scipy.ndimage.gaussian_filter(
                    grad,
                    min(8, self.gauss_size) if quick else self.gauss_size,
                    mode='nearest'
                )

                # Compensate for minfilter erosion effect
                grad = scipy.ndimage.maximum_filter(grad, self.minfilter_size, mode='nearest')

                # Apply gain and save to channel buffer
                grad *= self.gain
                local_gradient[y::path, x::patw] = grad
                local_gradient[y::path, x::patw] += self.offset
            except Exception:
                logger.exception("Error computing local gradient")
                raise

        if self.raw.default_pool is not None:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map

        tasks = []
        for y in xrange(path):
            for x in xrange(patw):
                tasks.append((data, local_gradient, y, x))
        for _ in map_(compute_local_gradient, tasks):
            pass

        return local_gradient

    def correct(self, data, local_gradient=None, quick=False, **kw):
        if local_gradient is None:
            local_gradient = self.detect(data, quick=quick)

        # Remove local gradient + offset into wider signed buffer to avoid over/underflow
        debiased = data.astype(local_gradient.dtype)
        debiased -= local_gradient

        # At this point, we may have out-of-bounds values due to
        # offset headroom. We have to clip the result, but carefully
        # to avoid numerical issues when close to data type limits.
        clip_min = 0
        clip_max = None
        if data.dtype.kind in ('i', 'u'):
            diinfo = numpy.iinfo(data.dtype)
            giinfo = numpy.iinfo(local_gradient.dtype)
            if diinfo.max < giinfo.max:
                clip_max = diinfo.max
        debiased = numpy.clip(debiased, clip_min, clip_max, out=debiased)

        # Copy into data buffer, casting back to data.dtype in the process
        data[:] = debiased
        return data


class PerFrameLocalGradientBiasRop(LocalGradientBiasRop):

    offset = -300
    despeckle = False


class PoissonGradientBiasRop(BaseRop):

    preavg_size = 256

    def detect(self, data, quick=False, **kw):
        path, patw = self._raw_pattern.shape
        dt = numpy.float64
        local_gradient = numpy.empty(data.shape, dt)
        data = self.raw.demargin(data.copy())
        def compute_local_gradient(task):
            try:
                data, local_gradient, y, x = task
                cdata = data[y::path, x::patw].astype(dt)

                # Compute sky baseline levels
                # We assume cdata is a sum of 2 Poisson: sky + target
                grad = scipy.ndimage.uniform_filter(cdata, self.preavg_size, mode='nearest')

                # Since sky tends to be much stronger than target, the uniform filter
                # should produce an approximation of the sky. After recentering the data,
                # the sky signal will cancel out, leaving mostly the target above 0.
                cdata -= grad
                grad2avg = scipy.ndimage.uniform_filter(cdata, self.preavg_size, mode='nearest')

                cdata -= grad2avg
                cdata *= cdata
                grad2sq = scipy.ndimage.uniform_filter(cdata, self.preavg_size, mode='nearest')
                grad2std = numpy.sqrt(grad2sq, out=grad2sq)
                del grad2sq, cdata

                # Actual target lies is a Poisson(grad2std)
                # If grad was correct, grad2avg == grad2std
                # So we correct gradient by grad2std - grad2avg
                grad -= grad2std

                grad = numpy.clip(grad, 0, None, out=grad)

                # Save to channel buffer
                local_gradient[y::path, x::patw] = grad
            except Exception:
                logger.exception("Error computing local gradient")
                raise

        if self.raw.default_pool is not None:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map

        tasks = []
        for y in xrange(path):
            for x in xrange(patw):
                tasks.append((data, local_gradient, y, x))
        for _ in map_(compute_local_gradient, tasks):
            pass

        return local_gradient

    def correct(self, data, local_gradient=None, quick=False, **kw):
        if local_gradient is None:
            local_gradient = self.detect(data, quick=quick)

        # Remove local gradient into wider signed buffer to avoid over/underflow
        debiased = data.astype(local_gradient.dtype)
        debiased -= local_gradient

        # At this point, we may have out-of-bounds values due to
        # offset headroom. We have to clip the result, but carefully
        # to avoid numerical issues when close to data type limits.
        clip_min = 0
        clip_max = None
        if data.dtype.kind in ('i', 'u'):
            diinfo = numpy.iinfo(data.dtype)
            clip_max = diinfo.max
        debiased = numpy.clip(debiased, clip_min, clip_max, out=debiased)

        # Copy into data buffer, casting back to data.dtype in the process
        data[:] = debiased
        return data