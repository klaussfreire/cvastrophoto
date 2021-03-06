# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy

import scipy.ndimage
import skimage.feature

from . import base
from . import focus

from cvastrophoto.util import gaussian

logger = logging.getLogger(__name__)

class SeeingMeasureRop(base.PerChannelMeasureRop):

    size = 32
    quick = False

    measure_dtype = numpy.float32

    def measure_channel(self, channel_data, detected=None, channel=None, size=None):
        if size is None:
            size = self.size

        # Build a segmentation of bright features
        edges = skimage.feature.canny(channel_data)
        if edges.max() == 0:
            thr = 0
        else:
            thr = channel_data[edges].min()
        mask = channel_data >= thr
        mask = scipy.ndimage.binary_closing(mask)
        mask = scipy.ndimage.binary_opening(mask)
        mask = scipy.ndimage.binary_dilation(mask, iterations=size)
        labels, nlabels = scipy.ndimage.label(mask)
        del mask

        # Compute the center of masses of each feature
        weights = channel_data
        index = numpy.arange(nlabels+1)
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
        del X, Y

        # Compute dispersion as weighed distance
        Davg = scipy.ndimage.sum(D * weights, labels, index)
        Dw = scipy.ndimage.sum(weights, labels, index)
        Davg[Dw > 0] /= Dw[Dw > 0]

        nzlabels = Davg != 0
        score = numpy.zeros_like(index, dtype=self.measure_dtype)
        score[nzlabels] = scipy.ndimage.mean(D, labels, index[nzlabels]) / Davg[nzlabels]
        score = score[labels]

        if not self.quick:
            score = gaussian.fast_gaussian(score, size)

        return score


class SeeingFocusRankingRop(SeeingMeasureRop, focus.FocusMeasureRop):

    focus_size = focus.FocusMeasureRop.size
    seeing_size = SeeingMeasureRop.size

    def measure_channel(self, channel_data, detected=None, channel=None):
        seeing_rank = SeeingMeasureRop.measure_channel(
            self, channel_data,
            detected=detected, channel=channel,
            size=self.seeing_size)
        focus_rank = focus.FocusMeasureRop.measure_channel(
            self, channel_data, detected=detected, channel=channel,
            size=self.focus_size)
        return numpy.square(seeing_rank) * focus_rank
