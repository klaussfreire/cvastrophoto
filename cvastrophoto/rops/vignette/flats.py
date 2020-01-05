from __future__ import absolute_import

import logging

from ..base import BaseRop

import numpy
import scipy.ndimage

logger = logging.getLogger(__name__)

class FlatImageRop(BaseRop):

    scale = None
    dtype = numpy.float32
    gauss_size = None
    min_luma = 5
    min_luma_ratio = 0.05
    remove_bias = False

    def __init__(self, raw=None, flat=None, color=False, flat_rop=None):
        super(FlatImageRop, self).__init__(raw)
        self.flat_rop = flat_rop
        self.set_flat(flat)

    def set_flat(self, flat):
        self.flat = flat
        self.flat_luma = self._flat_luma(flat)

    def _flat_luma(self, flat):
        if flat is None:
            return None

        if flat.max() > 65535:
            flat = flat * (65535.0 / flat.max())
        self.raw.set_raw_image(flat, add_bias=True)
        flatpp = self.raw.postprocessed
        if (flatpp == 65535).any():
            # Saturated flat fields are bad. Some cameras however have more
            # dynamic range in the raw than accessible through postprocessed
            # values, so try to shrink it until it's no longer saturated
            logger.warning(
                "Overexposed flat field, shifting exposure down to try to recover. "
                "Overexposure should be avoided in flat fields in any case.")
            shrink = 1
            while (flatpp == 65535).any():
                # Shrink and retry
                shrink *= 2
                self.raw.set_raw_image(
                    numpy.clip(flat.astype(numpy.int32) / shrink, 0, 65535),
                    add_bias=True)
                flatpp = self.raw.postprocessed
        luma = numpy.sum(flatpp, axis=2, dtype=numpy.uint32)

        min_luma = max(self.min_luma, self.min_luma_ratio * numpy.average(luma))
        if luma.min() <= min_luma:
            # cover holes
            bad_luma = luma <= min_luma
            luma[bad_luma] = luma[bad_luma].min()

        vshape = self.raw.rimg.raw_image_visible.shape
        lshape = luma.shape
        lyscale = vshape[0] / lshape[0]
        lxscale = vshape[1] / lshape[1]

        if self.gauss_size:
            luma = scipy.ndimage.gaussian_filter(luma, self.gauss_size, mode='nearest')

        raw_luma = flat.copy()
        sizes = self.raw.rimg.sizes
        if lyscale > 1 or lxscale > 1:
            for yoffs in xrange(lyscale):
                for xoffs in xrange(lxscale):
                    raw_luma[
                        sizes.top_margin+yoffs:sizes.top_margin+sizes.iheight:lyscale,
                        sizes.left_margin+xoffs:sizes.left_margin+sizes.iwidth:lxscale] = luma
        else:
            raw_luma[
                sizes.top_margin:sizes.top_margin+sizes.iheight,
                sizes.left_margin:sizes.left_margin+sizes.iwidth] = luma
        raw_luma = self.demargin(raw_luma)

        if self.flat_rop is not None:
            raw_luma = self.flat_rop.correct(raw_luma)
            raw_luma = self.demargin(raw_luma)

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
        flattened = light.astype(numpy.float32)
        if self.remove_bias:
            flattened -= self.raw.black_level
        flattened /= luma
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
