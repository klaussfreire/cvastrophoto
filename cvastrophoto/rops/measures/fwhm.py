# -*- coding: utf-8 -*-
from __future__ import absolute_import

from past.builtins import xrange
import logging
import numpy
import numpy.fft

import scipy.ndimage
import skimage.morphology
import skimage.feature
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline

from . import base
from ..tracking.extraction import ExtractPureStarsRop

from cvastrophoto.util import gaussian

logger = logging.getLogger(__name__)

class FWHMMeasureRop(base.PerChannelMeasureRop):

    min_sigmas = 2.0
    min_spacing = 1
    exclude_saturated = True
    saturation_margin = 0.9
    degree = 2
    quick = False
    quick_roi = 1024

    outlier_filter = 0.005

    measure_dtype = numpy.float32

    def __init__(self, raw, **kw):
        quick = kw.pop('quick', self.quick)
        extract_stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(ExtractPureStarsRop, k)}
        self._extract_stars_rop = ExtractPureStarsRop(raw, **extract_stars_kw)
        super(FWHMMeasureRop, self).__init__(raw, quick=quick, **kw)

    def measure_image(self, data, *p, **kw):
        stars = self._extract_stars_rop.correct(data)
        return super(FWHMMeasureRop, self).measure_image(stars, *p, **kw)

    def _scalar_from_stars(self, value, labels, C, full_stats=False, quadrants=False, start=1):
        if quadrants:
            Y = C[:,0].max() / 3
            X = C[:,1].max() / 3
            q = []
            for y in xrange(3):
                row = []
                q.append(row)
                for x in xrange(3):
                    vmask = (x*X <= C[:,1]) & (C[:,1] <= (x+1)*X) & (y*Y <= C[:,0]) & (C[:,0] <= (y+1)*Y)
                    vmask[0] = False
                    row.append(self._scalar_from_stars(value[vmask], labels, C[vmask], full_stats=full_stats, start=0))
            if not full_stats:
                q = numpy.array(q)
            return q
        elif full_stats:
            return dict(
                median=numpy.median(value[start:]),
                min=value[start:].min(),
                max=value[start:].max(),
                mean=value[start:].mean(),
            )
        else:
            return numpy.median(value[1:])

    def _scalar_from_channels(self, cdata):
        return numpy.average(list(cdata.values()), axis=0)

    def measure_scalar(self, data, *p, **kw):
        scalars = {}
        full_stats = kw.pop('full_stats', False)
        quadrants = kw.pop('quadrants', False)

        def gather_scalars(data, y, x, processed):
            scalars[(y,x)] = self._scalar_from_stars(*processed, full_stats=full_stats, quadrants=quadrants)

        kw['process_method'] = self._measure_channel_stars
        kw['rv_method'] = gather_scalars

        if self.measure_dtype is not None:
            data = data.astype(self.measure_dtype)
        else:
            data = data.copy()

        data = self._extract_stars_rop.correct(data)
        data = base.PerChannelMeasureRop.correct(self, data, *p, **kw)

        if full_stats:
            return scalars
        else:
            return self._scalar_from_channels(scalars)

    def _get_star_map(self, channel_data):
        if self.quick:
            # Find the brightest spot to build a tracking window around it
            quick_roi = self.quick_roi
            margin = min(quick_roi, min(channel_data.shape) // 4)
            mluma = channel_data[margin:-margin, margin:-margin]
            pos = numpy.argmax(mluma)

            ymax = pos // mluma.shape[1]
            xmax = pos - ymax * mluma.shape[1]
            ymax += margin
            xmax += margin

            wleft = min(xmax, quick_roi)
            wright = min(channel_data.shape[1] - xmax, quick_roi)
            wup = min(ymax, quick_roi)
            wdown = min(channel_data.shape[0] - ymax, quick_roi)
            channel_data = channel_data[ymax-wup:ymax+wdown, xmax-wleft:xmax+wright]

        # Build a noise floor to filter out dim stars
        size = self._extract_stars_rop.star_size
        nfloor = scipy.ndimage.uniform_filter(channel_data, size * 4)
        nfloor = nfloor + self.min_sigmas * numpy.sqrt(
            scipy.ndimage.uniform_filter(numpy.square(channel_data - nfloor), size * 4))
        nfloor = gaussian.fast_gaussian(
            nfloor,
            size * (1 if self.quick else 4),
            mode='wrap' if self.quick else 'reflect')

        # Find stars by building a mask around local maxima
        lmax = scipy.ndimage.maximum_filter(channel_data, size)

        potential_star_mask = channel_data > nfloor
        if self.exclude_saturated:
            potential_star_mask &= lmax < (channel_data.max() * self.saturation_margin)
        potential_star_mask = scipy.ndimage.binary_opening(
            potential_star_mask,
            skimage.morphology.disk(self.min_spacing))
        star_edge_mask = channel_data >= (lmax / 2)
        star_mask = potential_star_mask & star_edge_mask
        nstar_mask = scipy.ndimage.binary_opening(
            star_mask,
            skimage.morphology.disk(self.min_spacing))
        if nstar_mask.any():
            star_mask = nstar_mask
        del star_edge_mask, potential_star_mask, nfloor, lmax, nstar_mask

        # Remove outlier features, that are probably not stars, or overly bright and overblown stars
        outlier_pixels = star_pixels = star_mask.sum()
        m = star_mask
        outlier_radius = 0
        while outlier_pixels > star_pixels * self.outlier_filter:
            nm = scipy.ndimage.binary_erosion(m)
            noutlier_pixels = m.sum()
            if noutlier_pixels:
                m = nm
                outlier_pixels = noutlier_pixels
                outlier_radius += 1
                del nm
            else:
                break
        if outlier_pixels <= star_pixels * self.outlier_filter:
            for i in xrange(outlier_radius):
                m = scipy.ndimage.binary_dilation(m)
            star_mask &= ~m
        del m

        labels, n_stars = scipy.ndimage.label(star_mask)

        # Compute the center of masses of each star
        weights = channel_data
        index = numpy.arange(n_stars+1)
        centers = scipy.ndimage.center_of_mass(weights, labels, index)

        # Compute a coordinate grid with distance from the center
        X = numpy.arange(channel_data.shape[1], dtype=numpy.float32)
        Y = numpy.arange(channel_data.shape[0], dtype=numpy.float32)
        X, Y = numpy.meshgrid(X, Y)

        C = numpy.array(centers)

        X -= C[labels, 1]
        Y -= C[labels, 0]

        return labels, n_stars, index, C, X, Y

    def _dmax(self, X, Y, labels, index):
        D = numpy.square(X, out=X)
        D += numpy.square(Y, out=Y)
        D = numpy.sqrt(D, out=D)
        del X, Y

        # Compute FWHM as max distance
        return scipy.ndimage.maximum(D, labels, index) * 2

    def _measure_channel_stars(self, channel_data, detected=None, channel=None):
        labels, n_stars, index, C, X, Y = self._get_star_map(channel_data)
        Dmax = self._dmax(X, Y, labels, index)
        return Dmax, labels, C

    def measure_channel(self, channel_data, detected=None, channel=None):
        Dmax, labels, C = self._measure_channel_stars(channel_data)

        scaler = sklearn.preprocessing.StandardScaler()
        model = sklearn.pipeline.Pipeline([
            ('poly', sklearn.preprocessing.PolynomialFeatures(degree=self.degree)),
            ('linear', sklearn.linear_model.RidgeCV(alphas=numpy.logspace(-4, 4, 13)))
        ])
        Cn = C.copy()
        Cn[:,0] -= channel_data.shape[0]/2
        Cn[:,1] -= channel_data.shape[1]/2
        Dnorm = scaler.fit_transform(Dmax.reshape(-1, 1))
        model.fit(Cn[1:], Dnorm[1:])

        # Finally, evaluate the model on the full grid to produce a regularized parameter map
        X = numpy.arange(channel_data.shape[1], dtype=numpy.float32) - channel_data.shape[1]/2
        Y = numpy.arange(channel_data.shape[0], dtype=numpy.float32) - channel_data.shape[0]/2
        score = numpy.empty(channel_data.shape, self.measure_dtype)
        for ystart in xrange(0, len(Y), 128):
            grid = numpy.array([
                    g.ravel()
                    for g in numpy.meshgrid(X, Y[ystart:ystart+128])
                ]).transpose()
            score[ystart:ystart+128] = scaler.inverse_transform(
                model.predict(grid)).reshape(channel_data[ystart:ystart+128].shape)

        return score


class ElongationAngleMeasureRop(FWHMMeasureRop):

    def _elongation(self, X, Y, labels, index):
        # Compute angle as median angle - longest axis will win
        theta = Y.copy()
        safe = X != 0
        theta[safe] /= X[safe]
        theta[~safe] *= 100000
        theta = numpy.arctan(theta)
        del safe

        return scipy.ndimage.median(theta, labels, index)

    def _measure_channel_stars(self, channel_data, detected=None, channel=None):
        labels, n_stars, index, C, X, Y = self._get_star_map(channel_data)
        theta = self._elongation(X, Y, labels, index)
        return theta, labels, C


class TiltMeasureRop(ElongationAngleMeasureRop):

    def _measure_channel_stars(self, channel_data, detected=None, channel=None):
        labels, n_stars, index, C, X, Y = self._get_star_map(channel_data)

        Cx = C[:,1]
        Cy = C[:,0]
        Cnorm = numpy.sqrt(Cx*Cx + Cy*Cy)
        Cx = Cx / Cnorm
        Cy = Cy / Cnorm
        theta = self._elongation(X, Y, labels, index)
        longx = numpy.cos(theta)
        longy = numpy.sin(theta)
        tiltside = longx * Cx + longy * Cy
        tiltside *= tiltside
        tiltside = tiltside * 2 - 1
        dmax = self._dmax(X, Y, labels, index)

        tilt = (dmax - dmax.min()) * tiltside

        return tilt, labels, C


class InvFWHMMeasureRop(FWHMMeasureRop):

    def measure_scalar(self, data, *p, **kw):
        full_stats = kw.get('full_stats', False)
        scalars = super(InvFWHMMeasureRop, self).measure_scalar(data, *p, **kw)

        if full_stats:
            for k, fwhm in scalars.itervalues():
                scalars[k] = 1.0 / fwhm
        else:
            scalars = 1.0 / scalars

        return scalars
