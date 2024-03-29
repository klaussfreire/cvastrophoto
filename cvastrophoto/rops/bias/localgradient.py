# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
import collections
import logging
import numpy
import functools
import scipy.ndimage
import skimage.filters
import skimage.morphology
import skimage.transform
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline

from ..base import BaseRop
from cvastrophoto.util import gaussian, demosaic
from cvastrophoto.util.filters import quick_despeckle
from cvastrophoto.image import Image

logger = logging.getLogger(__name__)

def raw2yuv(raw_data, raw_pattern, wb, dtype=numpy.float32, scale=None, maxgreen=False):
    pat = raw_pattern
    path, patw = pat.shape

    rw = numpy.count_nonzero(pat == 0)
    gw = numpy.count_nonzero(pat == 1)
    bw = numpy.count_nonzero(pat == 2)
    cw = (rw, gw, bw)

    if scale is None:
        scale = raw_data.max()
    rgb_data = numpy.zeros((raw_data.shape[0] // path, raw_data.shape[1] // patw, 3), dtype)

    for y in xrange(path):
        for x in xrange(patw):
            c = pat[y,x]
            if cw[c] > 1:
                if maxgreen:
                    numpy.maximum(rgb_data[:,:,c], raw_data[y::path, x::patw], out=rgb_data[:,:,c])
                else:
                    rgb_data[:,:,c] += raw_data[y::path, x::patw]
            else:
                rgb_data[:,:,c] = raw_data[y::path, x::patw]

    for c in xrange(3):
        cwc = cw[c] if not maxgreen else 1
        rgb_data[:,:,c] *= wb[c] / (cwc * scale)

    return skimage.color.rgb2yuv(rgb_data), scale

def yuv2raw(yuv_data, raw_pattern, wb, scale, raw_data=None, dtype=numpy.int32):
    pat = raw_pattern
    path, patw = pat.shape
    rgb_data = skimage.color.yuv2rgb(yuv_data)

    if raw_data is None:
        raw_data = numpy.empty((rgb_data.shape[0] * path, rgb_data.shape[1] * patw), dtype)

    for y in xrange(path):
        for x in xrange(patw):
            c = pat[y,x]
            raw_data[y::path, x::patw] = numpy.clip(rgb_data[:,:,c] / wb[c], 0, 1) * scale

    return raw_data

class LocalGradientBiasRop(BaseRop):

    quick = False
    minfilter_size = 256
    gauss_size = 256
    pregauss_size = 8
    despeckle_size = 3
    chroma_filter_size = 0
    luma_minfilter_size = 0
    luma_gauss_size = 0
    iteration_factors = (1,)
    close_factor = 0.8
    opening_size = 0
    closing_size = 0
    gain = 1.0
    offset = -1
    clip = False
    despeckle = True
    aggressive = False
    noisecap = False
    auto_offset = False
    auto_offset_max_clip = 0.01
    residual_iterations = 0
    residual_size_factor = 0.5
    residual_protection = 1.0
    differential_scale = 0.0
    differential_despeckle_scale = 1.0
    differential_noise_threshold = 1.5
    zmask = -1.0
    gain_mask = ''
    svr_regularization = False
    svr_marginfix = False
    svr_margin = 0.1
    svr_mask = ''
    svr_core_exclude = 0.0
    svr_maxsamples = 250000
    svr_params = dict(alphas=numpy.logspace(-4, 4, 13), degree=1)
    svr_model = staticmethod(
        lambda degree, **kw: sklearn.pipeline.Pipeline([
            ('poly', sklearn.preprocessing.PolynomialFeatures(degree=degree)),
            ('linear', sklearn.linear_model.RidgeCV(**kw))
        ])
    )
    div_wbneutral = False
    mode = 'sub'
    edge_mode = 'nearest'
    fft_dtype = None
    single_channel = -1

    preprocessing_rop = None

    @property
    def PROCESSING_MARGIN(self):
        return max(
            self.minfilter_size or 0,
            self.gauss_size or 0,
            self.chroma_filter_size or 0,
            self.luma_minfilter_size or 0,
            self.luma_gauss_size or 0,
            self.preprocessing_rop.PROCESSING_MARGIN if self.preprocessing_rop is not None else 0,
        )

    @property
    def svr_degree(self):
        return self.svr_params['degree']

    @svr_degree.setter
    def svr_degree(self, degree):
        self.svr_params['degree'] = degree

    @property
    def scale(self):
        return self.minfilter_size

    @scale.setter
    def scale(self, value):
        self.minfilter_size = value
        self.gauss_size = value
        self.luma_minfilter_size = value // 4
        self.luma_gauss_size = value // 4

    def detect(self, data, quick=False, roi=None, **kw):
        local_gradient = self._detect(data, quick=quick, roi=roi, **kw)

        # At this point, we may have out-of-bounds values due to
        # offset headroom. We have to clip the result, but carefully
        # to avoid numerical issues when close to data type limits.
        path, patw = self._raw_pattern.shape
        clip_min = 0
        clip_max = None
        if data.dtype.kind in ('i', 'u'):
            diinfo = numpy.iinfo(data.dtype)
            giinfo = numpy.iinfo(local_gradient.dtype)
            if diinfo.max < giinfo.max:
                clip_max = diinfo.max

        if self.residual_iterations:
            for i in xrange(self.residual_iterations):
                residual_data = data.astype(local_gradient.dtype)
                residual_data -= local_gradient
                residual_data = numpy.clip(residual_data, clip_min, clip_max, out=residual_data)
                residual_gradient = self._detect(
                    residual_data,
                    quick=quick, roi=roi,
                    pregauss_size=self.gauss_size, size_factor=self.residual_size_factor,
                    **kw)
                del residual_data

                for y in xrange(path):
                    for x in xrange(patw):
                        local_gradient[y::path, x::patw] += (
                            residual_gradient[y::path, x::patw]
                            - numpy.average(residual_gradient[y::path, x::patw]) * self.residual_protection
                        ).astype(local_gradient.dtype, copy=False)

        return local_gradient

    def _detect(self, data, quick=False, roi=None, pregauss_size=None, size_factor=1, **kw):
        raw_pattern = self._raw_pattern
        path, patw = raw_pattern.shape
        patsize = raw_pattern.size
        if data.dtype.kind not in ('i', 'u'):
            dt = data.dtype
            is_int_dt = False
        else:
            dt = numpy.int32
            is_int_dt = True
        if len(data.shape) == 3:
            data = demosaic.remosaic(data, raw_pattern)
            demosaic_gradient = True
        else:
            data = data.copy()
            demosaic_gradient = False
        local_gradient = numpy.empty(data.shape, dt)
        if self.raw.demargin_safe:
            data = self.demargin(data)
        wb = self.raw.rimg.daylight_whitebalance

        if pregauss_size is None:
            pregauss_size = self.pregauss_size
        pregauss_size = int(pregauss_size * size_factor)

        if roi is not None:
            eff_roi, data = self.roi_precrop(roi, data)

        if self.preprocessing_rop is not None:
            data = self.preprocessing_rop.correct(data)

        gauss_kw = dict(mode=self.edge_mode, fft_dtype=self.fft_dtype)
        if quick or self.quick:
            gauss_kw['truncate'] = 2.5

        def soft_gray_opening(gradcell, minfilter_size, gauss_size, close_factor, opening_size, closing_size):
            # Weird hack to avoid keeping a reference to a needless temporary
            grad, = gradcell
            del gradcell[:]
            reclose_size = int(minfilter_size * close_factor)

            # Compute sky baseline levels
            # 4 optional steps, somewhat merged:
            # - Closing (to remove dark artifacts)
            # - Minfilter (to find background)
            # - Opening (to remove large diffuse objects)
            # - Reclose (to avoid phase offsets caused by minfilter)
            if closing_size:
                grad = scipy.ndimage.maximum_filter(grad, closing_size, mode='nearest', output=grad)

            grad = scipy.ndimage.minimum_filter(
                grad, minfilter_size + opening_size + closing_size, mode='nearest', output=grad)

            if opening_size:
                grad = scipy.ndimage.maximum_filter(grad, opening_size, mode='nearest', output=grad)

            # Regularization (smoothen)
            grad = gaussian.fast_gaussian(grad, gauss_size, **gauss_kw)

            # Compensate for minfilter erosion effect
            if reclose_size:
                grad = scipy.ndimage.maximum_filter(grad, reclose_size, mode='nearest', output=grad)

            return grad

        def despeckle_data(data, scale=1):
            despeckled = data

            if self.despeckle:
                if quick or self.aggressive:
                    despeckled = scipy.ndimage.maximum_filter(
                        despeckled, max(1, self.despeckle_size*scale), output=despeckled)
                else:
                    despeckled = quick_despeckle(
                        despeckled,
                        max(1, self.despeckle_size*scale),
                        mode='nearest')
            else:
                despeckled = data.copy()
            if self.zmask != -1.0:
                # Fill masked area with a neutral value that won't affect its surrounding gradient
                dmin = despeckled.min()
                davg = numpy.average(despeckled)
                thr = dmin + (davg - dmin) * self.zmask
                zmask = despeckled <= thr
                if despeckled is data:
                    despeckled = despeckled.copy()
                despeckled[zmask] = davg
                despeckled[zmask] = scipy.ndimage.maximum_filter(
                    despeckled, self.minfilter_size + 2 * self.gauss_size)[zmask]
            if pregauss_size:
                despeckled = gaussian.fast_gaussian(
                    despeckled, max(1, pregauss_size*scale),
                    **gauss_kw)

            return despeckled

        def compute_local_gradient(task, scale=1, despeckle_scale=1):
            try:
                data, local_gradient, y, x = task
                grad = data[y::path, x::patw]

                logger.info("Computing sky level at %d,%d scale %s", y, x, scale)

                if self.noisecap:
                    gradavg = numpy.average(grad)
                    gradstd = numpy.std(grad[grad <= gradavg])
                    logger.debug("Noise levels avg %s + std %s = %s", gradavg, gradstd, gradavg + gradstd)

                # Remove small-scale artifacts that could bias actual sky level
                grad = despeckle_data(grad, scale=despeckle_scale)

                for iscale in (self.iteration_factors[:1] if quick else self.iteration_factors):
                    iscale *= scale * size_factor
                    grad = [grad]  # Weird hack to avoid keeping a reference to a needless temporary
                    grad = soft_gray_opening(
                        grad,
                        int(self.minfilter_size * iscale),
                        int(min(8, self.gauss_size * iscale) if quick else self.gauss_size * iscale),
                        self.close_factor,
                        int(self.opening_size * iscale),
                        int(self.closing_size * iscale))

                if self.noisecap:
                    grad = numpy.minimum(grad, gradavg + gradstd, out=grad)

                # Apply gain and save to channel buffer
                local_gradient[y::path, x::patw] = grad
                logger.info("Computed sky gradient at %d,%d", y, x)
            except Exception:
                logger.exception("Error computing local gradient")
                raise

        def svr_regularize(grad, log_scope=""):
            logger.info("Applying sky level regularization %s", log_scope)

            # We construct a coordinate grid and select the inner portion
            # only, discarding an svr_margin fraction of it, since the edges
            # usually contain copious artifacts we don't want fitted
            ftype = numpy.float64 if grad.dtype.char == 'd' else numpy.float32
            fgrad = grad.astype(ftype)
            ygrid = numpy.linspace(-1, 1, fgrad.shape[0], dtype=ftype)
            xgrid = numpy.linspace(-1, 1, fgrad.shape[1], dtype=ftype)
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
            gradmin = mgrad.min()
            gradmax = mgrad.max()
            mgrad -= gradavg
            mgrad *= (1.0 / gradstd)

            # Pick a reduced sample.
            # It should be more than sufficient anyway since a linear model
            # is simple enough.
            sampling = max(1, mgrad.size // self.svr_maxsamples)

            # Build training samples - exclude exclusion zones
            if self.svr_mask:
                svr_mask = Image.open(self.svr_mask).luma_image(same_shape=False) > 0
                svr_mask = skimage.transform.resize(svr_mask, fgrad.shape).astype(svr_mask.dtype)
                svr_mask = svr_mask[ymargin:-ymargin,xmargin:-xmargin].flatten()
            else:
                svr_mask = slice(None)
            X = grid[svr_mask][::sampling]
            Y = mgrad.reshape(mgrad.size)[svr_mask][::sampling]
            trainmask = ~(
                (-self.svr_core_exclude < X[:,0]) & (X[:,0] < self.svr_core_exclude)
                & (-self.svr_core_exclude < X[:,1]) & (X[:,1] < self.svr_core_exclude)
            )
            X = X[trainmask]
            Y = Y[trainmask]
            del trainmask

            reg = self.svr_model(**self.svr_params)
            reg.fit(X, Y)
            del mgrad
            del grid

            # Finally, evaluate the model on the full grid (no margins)
            # to produce a regularized sky level
            logger.info("Computing regularized sky level %s", log_scope)
            if self.svr_marginfix:
                gtop = grad[:ymargin].copy()
                gbottom = grad[-ymargin:].copy()
                gleft = grad[:,:xmargin].copy()
                gright = grad[:,-xmargin:].copy()
            for ystart in xrange(0, len(ygrid), 128):
                grid = numpy.array([
                        g.ravel()
                        for g in numpy.meshgrid(xgrid, ygrid[ystart:ystart+128])
                    ]).transpose()
                pred = reg.predict(grid).reshape(grad[ystart:ystart+128].shape)
                pred *= gradstd
                pred += gradavg
                pred = numpy.clip(pred, gradmin, gradmax, pred)
                grad[ystart:ystart+128] = pred
            if self.svr_marginfix:
                grad[:ymargin] = gtop
                grad[-ymargin:] = gbottom
                grad[:,:xmargin] = gleft
                grad[:,-xmargin:] = gright
                del gtop, gbottom, gleft, gright
            del grid, reg, pred

        def smooth_local_gradient(task):
            try:
                data, local_gradient, y, x = task
                grad = local_gradient[y::path, x::patw]
                orig_data = data[y::path, x::patw]

                if self.svr_regularization and not quick and not regularized:
                    # Fit to a linear gradient
                    # Sky level can't be any more complex than that, but the above filters
                    # could pick large-scale structures and thus damage them. Fitting a
                    # a linear gradient removes them leaving only the base sky levels
                    svr_regularize(grad, 'at %d,%d' % (y, x))

                if self.gain != 1:
                    if is_int_dt:
                        grad[:] = grad * self.gain
                    else:
                        grad *= self.gain

                if gain_mask is not None:
                    channel_gain_mask = gain_mask[y::path, x::patw]
                    if is_int_dt:
                        grad[:] = grad * channel_gain_mask
                    else:
                        grad *= channel_gain_mask

                if self.auto_offset:
                    grad_plow = numpy.percentile(orig_data - grad, self.auto_offset_max_clip * 100)
                    if grad_plow < 0:
                        grad += grad_plow.astype(grad.dtype)
                if self.offset != 0:
                    if self.offset < 0 and self.clip:
                        grad += self.offset
                        grad = numpy.clip(grad, 0, None, out=grad)
                    else:
                        grad += self.offset
                logger.info("Computed sky level at %d,%d", y, x)
            except Exception:
                logger.exception("Error computing local gradient")
                raise

        if self.raw.default_pool is not None and patsize > 1:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map

        def parallel_task(fn, tasks=None):
            if tasks is None:
                tasks = []
                for y in xrange(path):
                    for x in xrange(patw):
                        if self.single_channel == -1 or raw_pattern[y,x] == self.single_channel:
                            tasks.append((data, local_gradient, y, x))
            for _ in map_(fn, tasks):
                pass

        # First stage, compute base sky level
        parallel_task(compute_local_gradient)

        if self.differential_scale:
            fine_grad = local_gradient.copy()
            parallel_task(functools.partial(compute_local_gradient,
                scale=self.differential_scale,
                despeckle_scale=self.differential_despeckle_scale))
            coarse_grad = local_gradient.copy()

            for y in xrange(path):
                for x in xrange(patw):
                    ccoarse = coarse_grad[y::path, x::patw]
                    cfine = fine_grad[y::path, x::patw]
                    cdata = data[y::path, x::patw]

                    gradavg = numpy.average(cdata)
                    gradstd = numpy.std(cdata[cdata <= gradavg])
                    gradthr = self.differential_noise_threshold

                    cweight = numpy.abs(cfine - ccoarse).astype(numpy.float32) / (gradthr * gradstd)
                    cweight = numpy.clip(cweight, 0, 1, out=cweight)
                    local_gradient[y::path, x::patw] = cfine * (1 - cweight) + ccoarse * cweight
            del fine_grad, coarse_grad

        multichannel = raw_pattern.max() > 0 and self.single_channel != -1
        if multichannel and (self.chroma_filter_size or (self.luma_minfilter_size and self.luma_gauss_size)):
            scale = max(local_gradient.max(), data.max())

            yuv_grad, scale = raw2yuv(
                local_gradient, raw_pattern, wb,
                scale=scale, maxgreen=self.aggressive, dtype=local_gradient.dtype)

            def smooth_chroma(yuv_grad, c):
                if self.chroma_filter_size == 'median':
                    yuv_grad[:,:,c] = numpy.median(yuv_grad[:,:,c])
                else:
                    yuv_grad[:,:,c] = gaussian.fast_gaussian(
                        yuv_grad[:,:,c], self.chroma_filter_size,
                        **gauss_kw)

            def open_luma(yuv_grad, c):
                yuv_grad[:,:,c] = soft_gray_opening(
                    [yuv_grad[:,:,c]],
                    self.luma_minfilter_size,
                    min(8, self.luma_gauss_size) if quick else self.luma_gauss_size,
                    self.close_factor,
                    0, 0)

            def process_channel(c):
                if c == 0:
                    if self.luma_minfilter_size and self.luma_gauss_size:
                        open_luma(yuv_grad, c)
                else:
                    if self.chroma_filter_size:
                        smooth_chroma(yuv_grad, c)

            parallel_task(process_channel, range(3))

            if self.svr_regularization:
                regularized = True
                for _ in map_(svr_regularize, [yuv_grad[:,:,i] for i in xrange(yuv_grad.shape[2])]):
                    pass

            yuv2raw(yuv_grad, raw_pattern, wb, scale, local_gradient, local_gradient.dtype)
        else:
            regularized = False

        # Second stage, apply smoothing and regularization
        if self.gain_mask:
            gain_mask = Image.open(self.gain_mask).luma_image(same_shape=False).astype(numpy.float32)
            gain_mask *= (1.0 / gain_mask.max())
            gain_mask = skimage.transform.resize(gain_mask, local_gradient.shape).astype(gain_mask.dtype)
        else:
            gain_mask = None

        parallel_task(smooth_local_gradient)

        if self.single_channel != -1:
            for y in xrange(path):
                for x in xrange(patw):
                    if raw_pattern[y,x] != self.single_channel:
                        local_gradient[y::path, x::patw] = 0
        if len(set(raw_pattern.flatten())) != raw_pattern.size:
            # Some components appear multiple times (like RGGB)
            # Average the components in each cell to avoid moire artifacts
            avg_local_gradient = collections.defaultdict(int)
            pattern_weights = collections.defaultdict(int)
            for y in xrange(path):
                for x in xrange(patw):
                    avg_local_gradient[raw_pattern[y, x]] += local_gradient[y::path, x::patw]
                    pattern_weights[raw_pattern[y, x]] += 1
            for c, component_gradient in avg_local_gradient.items():
                if pattern_weights[c]:
                    if component_gradient.dtype.kind in 'iu':
                        # Integer division is both faster and more precise
                        component_gradient //= pattern_weights[c]
                    else:
                        # For floats multiply by reciprocal is faster
                        component_gradient *= 1.0 / pattern_weights[c]
            for y in xrange(path):
                for x in xrange(patw):
                    local_gradient[y::path, x::patw] = avg_local_gradient[raw_pattern[y, x]]
            del avg_local_gradient, pattern_weights

        # No negatives
        local_gradient = numpy.clip(local_gradient, 0, None, out=local_gradient)

        if roi is not None:
            local_gradient = self.roi_postcrop(roi, eff_roi, local_gradient)

        if demosaic_gradient:
            local_gradient = demosaic.demosaic(local_gradient, self._raw_pattern)

        return local_gradient

    def correct(self, data, local_gradient=None, quick=False, roi=None, **kw):
        if roi is not None:
            eff_roi, data = self.roi_precrop(roi, data)

        if local_gradient is None:
            local_gradient = self.detect(data, quick=quick)

        debiased = data.astype(local_gradient.dtype)

        if self.mode == 'div':
            if self.div_wbneutral:
                raw_pattern = self._raw_pattern
                path, patw = raw_pattern.shape
                wb_shift = numpy.empty(raw_pattern.shape, dtype=numpy.float32)
                for y in xrange(path):
                    for x in xrange(patw):
                        wb_shift[y,x] = numpy.average(local_gradient[y::path, x::patw])
                wb_shift /= wb_shift.max()
            dbmax = debiased.max()
            debiased[:] = numpy.clip(debiased / (local_gradient * (1.0 / local_gradient.max())), None, dbmax)
            if self.div_wbneutral:
                for y in xrange(path):
                    for x in xrange(patw):
                        debiased[y::path,x::patw] *= wb_shift[y,x]
        elif self.mode == 'ldiv':
            local_gradient = self.raw.luma_image(
                local_gradient, renormalize=True, same_shape=True, dtype=local_gradient.dtype,
                raw_pattern = self._raw_pattern)
            dbmax = debiased.max()
            debiased[:] = numpy.clip(debiased / (local_gradient * (1.0 / local_gradient.max())), None, dbmax)
        elif self.mode == 'set':
            debiased[:] = local_gradient
        else:
            # Remove local gradient + offset into wider signed buffer to avoid over/underflow
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

        if roi is not None:
            data = self.roi_postcrop(roi, eff_roi, data)

        return data


class PerFrameLocalGradientBiasRop(LocalGradientBiasRop):

    minfilter_size = 32
    gauss_size = 16
    pregauss_size = 8
    despeckle_size = 3
    iteration_factors = (1,) * 8
    svr_regularization = True


class MoonGradientBiasRop(LocalGradientBiasRop):
    minfilter_size = 128
    gauss_size = 128
    pregauss_size = 16
    despeckle_size = 1
    chroma_filter_size = 32
    luma_minfilter_size = 32
    luma_gauss_size = 32
    despeckle = False
    svr_regularization = True


class QuickGradientBiasRop(LocalGradientBiasRop):
    minfilter_size = 64
    gauss_size = 64
    pregauss_size = 8
    despeckle_size = 2
    chroma_filter_size = None
    luma_minfilter_size = None
    luma_gauss_size = None
    despeckle = True
    svr_regularization = False
    fft_dtype = numpy.complex64
    quick = True


class PoissonGradientBiasRop(BaseRop):

    preavg_size = 256

    def detect(self, data, quick=False, **kw):
        path, patw = self._raw_pattern.shape
        dt = numpy.float64
        local_gradient = numpy.empty(data.shape, dt)
        data = self.demargin(data.copy())
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
