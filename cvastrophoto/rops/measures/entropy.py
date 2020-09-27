# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy
import scipy.ndimage
import skimage.morphology

from . import base

from cvastrophoto.util import entropy


logger = logging.getLogger(__name__)


class LocalEntropyMeasureRop(base.BaseMeasureRop):

    gamma = 2.4
    size = 32
    erode = 16

    def measure_image(self, data, detected=None):
        selem = skimage.morphology.disk(self.size)
        luma = self.raw.luma_image(data, dtype=numpy.float32, same_shape=True)
        ent = entropy.local_entropy(luma, selem=selem, gamma=self.gamma, copy=False)

        if self.erode:
            erode_disk = skimage.morphology.disk(self.erode)
            ent = scipy.ndimage.minimum_filter(ent, footprint=erode_disk)

        return ent
