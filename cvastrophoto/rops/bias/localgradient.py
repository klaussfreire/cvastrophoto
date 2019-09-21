# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy
import scipy.ndimage
import skimage.filters
import skimage.morphology
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
from ..base import BaseRop

logger = logging.getLogger(__name__)

class LocalGradientBiasRop(BaseRop):

    minfilter_size = 64
    gauss_size = 64
    pregauss_size = 8
    despeckle_size = 3
    iteration_factors = (1,)
    gain = 1
    offset = -1
    despeckle = True
    aggressive = False
    svr_regularization = False
    svr_margin = 0.1
    svr_maxsamples = 250000
    svr_params = dict(alphas=numpy.logspace(-4, 4, 13))
    svr_model = staticmethod(
        lambda degree=3, **kw: sklearn.pipeline.Pipeline([
            ('poly', sklearn.preprocessing.PolynomialFeatures(degree=degree)),
            ('linear', sklearn.linear_model.RidgeCV(**kw))
        ])
    )

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
                logger.info("Computing sky level at %d,%d", y, x)
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

                grad = despeckled
                del despeckled
                for scale in (self.iteration_factors[:1] if quick else self.iteration_factors):
                    # Compute sky baseline levels
                    grad = scipy.ndimage.minimum_filter(grad, self.minfilter_size * scale, mode='nearest')

                    # Regularization (smoothen)
                    grad = scipy.ndimage.gaussian_filter(
                        grad,
                        min(8, self.gauss_size) if quick else self.gauss_size * scale,
                        mode='nearest'
                    )

                    # Compensate for minfilter erosion effect
                    grad = scipy.ndimage.maximum_filter(grad, self.minfilter_size * scale, mode='nearest')

                if self.svr_regularization and not quick:
                    # Fit to a linear gradient
                    # Sky level can't be any more complex than that, but the above filters
                    # could pick large-scale structures and thus damage them. Fitting a
                    # a linear gradient removes them leaving only the base sky levels
                    logger.info("Applying sky level regularization at %d,%d", y, x)

                    # We construct a coordinate grid and select the inner portion
                    # only, discarding an svr_margin fraction of it, since the edges
                    # usually contain copious artifacts we don't want fitted
                    fgrad = grad.astype(numpy.float)
                    ygrid = numpy.linspace(-1, 1, fgrad.shape[0], dtype=numpy.float)
                    xgrid = numpy.linspace(-1, 1, fgrad.shape[1], dtype=numpy.float)
                    ymargin = max(1, int(self.svr_margin * len(ygrid)))
                    xmargin = max(1, int(self.svr_margin * len(xgrid)))
                    grid = numpy.array([
                            g.flatten()
                            for g in numpy.meshgrid(xgrid[xmargin:-xmargin], ygrid[ymargin:-ymargin])
                        ]).transpose()
                    mgrad = fgrad[ymargin:-ymargin,xmargin:-xmargin]

                    # Center the values at 0 to make them easier to fit
                    gradavg = numpy.average(mgrad)
                    gradstd = max(1, numpy.std(mgrad))
                    mgrad -= gradavg
                    mgrad *= (1.0 / gradstd)

                    # Pick a reduced sample.
                    # It should be more than sufficient anyway since a linear model
                    # is simple enough.
                    sampling = max(1, mgrad.size / self.svr_maxsamples)
                    reg = self.svr_model(**self.svr_params)
                    reg.fit(grid[::sampling], mgrad.reshape(mgrad.size)[::sampling])
                    del mgrad

                    # Finally, evaluate the model on the full grid (no margins)
                    # to produce a regularized sky level
                    logger.info("Computing regularized sky level at %d,%d", y, x)
                    grid = numpy.array([
                            g.flatten()
                            for g in numpy.meshgrid(xgrid, ygrid)
                        ]).transpose()
                    pred = reg.predict(grid).reshape(grad.shape)
                    pred *= gradstd
                    pred += gradavg
                    grad[:] = pred
                    del grid, reg, pred

                # Apply gain and save to channel buffer
                grad *= self.gain
                local_gradient[y::path, x::patw] = grad
                local_gradient[y::path, x::patw] += self.offset
                logger.info("Computed sky level at %d,%d", y, x)
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