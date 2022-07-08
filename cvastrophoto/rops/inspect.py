# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import logging
import skimage.exposure

from .base import BaseRop


logger = logging.getLogger(__name__)


class ShowImageRop(BaseRop):

    stretch = True

    def __init__(self, raw):
        super(ShowImageRop, self).__init__(raw)

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        logger.info(
            "Showing image (max=%r, min=%r, avg=%r): %r",
            data.max(), data.min(), numpy.average(data),
            data)
        odata = data
        if self.stretch:
            data = data.copy()
            path, patw = self._raw_pattern.shape
            for y in range(path):
                for x in range(patw):
                    sdata = data[y::path, x::patw]
                    lo, hi = numpy.percentile(sdata, (2, 98))
                    sdata[:] = skimage.exposure.rescale_intensity(
                        sdata,
                        in_range=(lo, hi), out_range=(0, 65535))

        self.raw.set_raw_image(data, add_bias=True)
        self.raw.show()
        return odata
