# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy

import scipy.ndimage
import skimage.morphology
import skimage.feature

from . import base
from ..tracking.extraction import ExtractPureStarsRop

logger = logging.getLogger(__name__)

class FWHMMeasureRop(base.PerChannelMeasureRop):

    min_sigmas = 2.0
    min_spacing = 1

    measure_dtype = numpy.float32

    def __init__(self, raw, **kw):
        extract_stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(ExtractPureStarsRop, k)}
        self._extract_stars_rop = ExtractPureStarsRop(raw, **extract_stars_kw)
        super(FWHMMeasureRop, self).__init__(raw, **kw)

    def measure_image(self, data, *p, **kw):
        stars = self._extract_stars_rop.correct(data.copy())
        return super(FWHMMeasureRop, self).measure_image(stars, *p, **kw)

    def _measure_channel_stars(self, channel_data):
        # Build a noise floor to filter out dim stars
        size = self._extract_stars_rop.star_size
        nfloor = scipy.ndimage.uniform_filter(channel_data, size * 4)
        nfloor = nfloor + self.min_sigmas * numpy.sqrt(
            scipy.ndimage.uniform_filter(numpy.square(channel_data - nfloor), size * 4))
        nfloor = scipy.ndimage.gaussian_filter(nfloor, size * 4)

        # Find stars by building a mask around local maxima
        lmax = scipy.ndimage.maximum_filter(channel_data, size)

        potential_star_mask = scipy.ndimage.binary_opening(
            channel_data > nfloor,
            skimage.morphology.disk(self.min_spacing))
        star_edge_mask = channel_data >= (lmax / 2)
        star_mask = scipy.ndimage.binary_opening(
            potential_star_mask & star_edge_mask,
            skimage.morphology.disk(self.min_spacing))
        del star_edge_mask, potential_star_mask, nfloor

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
        D = numpy.square(X, out=X)
        D += numpy.square(Y, out=Y)
        D = numpy.sqrt(D, out=D)
        del X, Y

        # Compute FWHM as max distance
        Dmax = scipy.ndimage.maximum(D, labels, index)

        return Dmax, labels

    def measure_channel(self, channel_data, detected=None, channel=None):
        Dmax, labels = self._measure_channel_stars(channel_data)

        size = self._extract_stars_rop.star_size
        maxscore = Dmax[1:].max()
        minscore = Dmax[1:].min()
        score = (((Dmax - minscore) / max(0.00001, maxscore - minscore)) * 255).astype(numpy.uint8)[labels]
        score[labels == 0] = numpy.median(Dmax[1:])
        score = skimage.filters.rank.median(score, skimage.morphology.disk(size*4), mask=labels != 0)
        score = minscore + score * (maxscore / 255.0)
        score = scipy.ndimage.gaussian_filter(score, size*4)

        return score
