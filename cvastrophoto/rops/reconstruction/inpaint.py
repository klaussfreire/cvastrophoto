from __future__ import absolute_import

import numpy
import skimage.restoration
import skimage.transform

from cvastrophoto.rops.base import BaseRop
from cvastrophoto.util import demosaic
from cvastrophoto import image


class InpaintRop(BaseRop):
    method = 'biharmonic'
    mask = ''
    dqmask = 3

    METHODS = {
        'biharmonic': lambda data, mask: skimage.restoration.inpaint_biharmonic(data, mask, multichannel=True),
    }

    AUTOMASKS = {
        'zero': lambda data: data == 0,
        'sat': lambda data: data == data.max(),
    }

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, img=None, **kw):
        raw_pattern = self._raw_pattern

        roi = kw.get('roi')

        mask_fn = None
        if self.mask in self.AUTOMASKS:
            mask_fn = self.AUTOMASKS[self.mask]
        elif self.mask:
            mask = image.Image.open(self.mask).luma_image(same_shape=False) > 0
            def mask_fn(data):
                if data.shape != mask.shape:
                    return skimage.transform.resize(mask, data.shape).astype(numpy.bool8)
                else:
                    return mask
        elif img is not None:
            # Get bad pixel map if available
            extensions = getattr(img.rimg, 'extensions')
            if 'DQ' in extensions:
                # STCI data quality flags
                mask = (extensions['DQ'].data & self.dqmask) != 0
                mask_fn = lambda : mask

        if mask_fn is None:
            raise RuntimeError("Missing inpainting mask")

        def process_data(data):
            rmask = mask_fn(data)

            if roi is not None:
                eff_roi, data = self.roi_precrop(roi, data)
                _, rmask = self.roi_precrop(roi, rmask)

            ppdata = demosaic.demosaic(data, raw_pattern)

            ppdata = self.METHODS[self.method](ppdata, rmask)

            data = demosaic.remosaic(ppdata, raw_pattern, out=data)

            if roi is not None:
                data = self.roi_postcrop(roi, eff_roi, data)

            return data

        rv = data

        if not isinstance(data, list):
            data = [data]

        for sdata in data:
            if sdata is None:
                continue

            sdata = process_data(sdata)

        return rv
