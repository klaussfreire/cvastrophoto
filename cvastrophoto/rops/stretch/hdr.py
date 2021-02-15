# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
import logging
import numpy
import scipy.ndimage
import skimage.morphology
import skimage.filters.rank
from cvastrophoto.util import entropy, demosaic

from .. import base

logger = logging.getLogger(__name__)

class HDRStretchRop(base.BaseRop):

    bright = 1.0
    gamma = 2.4
    size = 32
    erode = 0
    steps = 6
    white = 1.0
    rescale = True

    @property
    def step_scales(self):
        if isinstance(self.steps, int):
            return [2**i for i in range(self.steps)]
        else:
            return self.steps

    def get_hdr_step(self, data, scale):
        return numpy.clip(data.astype(numpy.float32, copy=False) * scale, 0, 65535)

    def get_hdr_set(self, data, dmax=None, bright=None):
        if dmax is None:
            dmax = data.max()
        if bright is None:
            bright = self.bright
        scale = 65535.0 * bright / max(0.01, dmax)

        if len(data.shape) == 3:
            data = demosaic.remosaic(data, self._raw_pattern)

        data = data.astype(numpy.float32)

        # Get the different exposure steps
        iset = []
        for step in self.step_scales:
            self.raw.set_raw_image(
                self.get_hdr_step(data, scale * step),
                add_bias=True)
            img = self.raw.postprocessed.copy()
            iset.append((step, scale * step, img))

        # Compute local entropy weights
        selem = skimage.morphology.disk(self.size)
        if self.erode:
            erode_disk = skimage.morphology.disk(self.erode)
        else:
            erode_disk = None

        def append_entropy(entry):
            step, scale, img = entry
            luma = self.raw.postprocessed_luma(numpy.float32, copy=True, postprocessed=img)
            ent = entropy.local_entropy(luma, selem=selem, gamma=self.gamma, copy=False, white=self.white)
            if erode_disk is not None:
                ent = scipy.ndimage.minimum_filter(ent, footprint=erode_disk)
            return (step, scale, ent)

        if self.raw.default_pool is not None:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map
        iset = list(map_(append_entropy, iset))

        # Fix all-zero weights
        max_ent = iset[0][2].copy()
        for step, img, ent in iset[1:]:
            max_ent = numpy.maximum(max_ent, ent)

        if max_ent.min() <= 0:
            # All-zero weights happen with always-saturated pixels
            clippers = max_ent <= 0
            for step, img, ent in iset:
                ent[clippers] = 1

        return iset

    def detect(self, data, dmax=None, bright=bright, **kw):
        return self.get_hdr_set(data, dmax=dmax, bright=bright)

    def correct(self, data, detected=None, dmax=None, bright=None, **kw):

        if detected is None:
            detected = self.detect(data, dmax=dmax, bright=bright)

        iset = detected
        path, patw = self._raw_pattern.shape
        sizes = self._raw_sizes

        if len(data.shape) == 3:
            data = demosaic.remosaic(data, self._raw_pattern)
            demosaic_result = True
        else:
            demosaic_result = False

        # Do the entropy-weighted average
        step, scale, ent = iset[0]
        raw_hdr_img = numpy.zeros(data.shape, numpy.float32)
        hdr_img = raw_hdr_img[
            sizes.top_margin:sizes.top_margin+sizes.height,
            sizes.left_margin:sizes.left_margin+sizes.width]
        ent_sum = numpy.zeros(ent.shape, ent.dtype)
        raw_shape = hdr_img.shape
        nchannels = hdr_img.size // ent.size
        if nchannels == (patw * path):
            hdr_img = hdr_img.reshape((raw_shape[0] // path, raw_shape[1] // patw, nchannels))
        else:
            hdr_img = hdr_img.reshape((raw_shape[0], raw_shape[1] // nchannels, nchannels))
        for step, scale, ent in iset:
            img = self.get_hdr_step(data, scale)[
                sizes.top_margin:sizes.top_margin+sizes.height,
                sizes.left_margin:sizes.left_margin+sizes.width].reshape(hdr_img.shape)
            for c in xrange(nchannels):
                hdr_img[:,:,c] += img[:,:,c] * ent
            ent_sum += ent
        if ent_sum.min() <= 0:
            ent_sum[ent_sum <= 0] = 1
        for c in xrange(nchannels):
            hdr_img[:,:,c] /= ent_sum
        if self.rescale:
            hdr_img *= 65535.0 / max(1, hdr_img.max())
        hdr_img = numpy.clip(hdr_img, 0, 65535, out=hdr_img)

        if demosaic_result:
            raw_hdr_img = demosaic.demosaic(raw_hdr_img, self._raw_pattern)

        return raw_hdr_img
