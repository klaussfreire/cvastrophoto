from __future__ import absolute_import

from ..base import BaseRop

import numpy
import scipy.ndimage

class FlatImageRop(BaseRop):

    scale = None
    dtype = numpy.float32
    gauss_size = 8
    min_luma = 5
    min_luma_ratio = 0.05

    def __init__(self, raw=None, flat=None, color=False):
        super(FlatImageRop, self).__init__(raw)
        self.set_flat(flat)

    def set_flat(self, flat):
        self.flat = flat
        self.flat_luma = self._flat_luma(flat)

    def _flat_luma(self, flat):
        if flat is None:
            return None

        if flat.max() > 65535:
            flat = flat * (65535.0 / flat.max())
        self.raw.set_raw_image(flat)
        luma = numpy.sum(self.raw.postprocessed, axis=2, dtype=numpy.uint32)

        min_luma = max(self.min_luma, self.min_luma_ratio * numpy.average(luma))
        if luma.min() <= min_luma:
            # cover holes
            bad_luma = luma <= min_luma
            luma[bad_luma] = luma[bad_luma].min()

        raw_luma = flat.copy()
        sizes = self.raw.rimg.sizes
        raw_luma[
            sizes.top_margin:sizes.top_margin+sizes.iheight,
            sizes.left_margin:sizes.left_margin+sizes.iwidth] = luma
        raw_luma = self.raw.demargin(raw_luma)

        return raw_luma

    def detect(self, data, **kw):
        pass

    def correct(self, data, flat=None, **kw):
        if flat is None:
            flat_luma = self.flat_luma
        else:
            flat_luma = self._flat_luma(flat)

        return self.flatten(data, flat_luma)

    def flatten(self, light, luma, dtype=None, scale=None):
        flattened = light.astype(numpy.float32) / luma
        flattened *= 1.0 / flattened.max()
        flattened = numpy.clip(flattened, 0, 1, out=flattened)

        if scale is None:
            scale = self.scale
        if dtype is None:
            dtype = self.dtype
        if scale is not None:
            flattened *= scale
        if dtype is not numpy.float32:
            flattened = flattened.astype(dtype)

        return flattened
